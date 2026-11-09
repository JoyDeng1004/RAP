#!/usr/bin/env python3
"""Fill only the 46 audited navtrain camera gaps, using a separate staging root."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


MANIFEST = Path(__file__).with_name("navtrain_camera_gaps.json")
CAMERAS = {"CAM_B0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"}


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def checked_path(root, relative):
    target = root / relative
    target.resolve().relative_to(root.resolve())
    return target


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_image(path):
    from PIL import Image

    with Image.open(path) as picture:
        if picture.format != "JPEG":
            raise ValueError("Expected a JPEG: " + str(path))
        picture.verify()
    with Image.open(path) as picture:
        picture.load()


def publish(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".navtrain-gap-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
            verify_image(temporary)
            os.chmod(temporary, 0o644)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def fetch(url, target, size, checksum):
    if not url.startswith("https://"):
        raise ValueError("Download URLs must use HTTPS")
    if size <= 0 or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("A positive size and SHA-256 are required")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == size and digest(target) == checksum:
            return target
        raise ValueError("Existing cached file failed verification: " + str(target))
    partial = target.with_name(target.name + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--proto", "=https", "--proto-redir", "=https",
        "--retry", "3", "--connect-timeout", "30", "--max-time", "21600",
        "--continue-at", "-", "--output", str(partial), url,
    ], check=True)
    if partial.stat().st_size != size or digest(partial) != checksum:
        raise ValueError("Downloaded file failed size/SHA-256 verification: " + str(partial))
    os.link(partial, target)
    partial.unlink()
    return target


def extract_selected(archive, wanted, staging):
    remaining = set(wanted)
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts or len(parts) < 3:
                continue
            relative = "/".join(parts[-3:])
            if relative not in remaining:
                continue
            if not member.isfile() or member.size <= 0:
                raise ValueError("Invalid archive image member: " + member.name)
            target = checked_path(staging, relative)
            with bundle.extractfile(member) as source:
                publish(source, target)
            remaining.remove(relative)
            if not remaining:
                break
    if remaining:
        raise ValueError("Archive did not contain all requested images: " + str(len(remaining)))


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    paths = manifest["paths"]
    logs = {archive["log"] for archive in manifest["archives"]}
    if len(paths) != 46 or len(set(paths)) != 46:
        raise ValueError("Expected the fixed 46-file audit allowlist")
    for relative in paths:
        parts = PurePosixPath(relative).parts
        if (len(parts) != 3 or parts[0] not in logs or parts[1] not in CAMERAS
                or not parts[2].endswith(".jpg") or ".." in parts):
            raise ValueError("Unexpected path in camera allowlist: " + relative)
    return manifest


def run(args):
    manifest = load_manifest()
    sensor_root = Path(args.sensor_root or manifest["sensor_root"]).resolve()
    staging = Path(args.staging).resolve()
    if sensor_root == staging or sensor_root in staging.parents or staging in sensor_root.parents:
        raise ValueError("Staging and live sensor root must be separate, non-nested directories")
    missing = []
    staged = []
    for relative in manifest["paths"]:
        target = checked_path(sensor_root, relative)
        candidate = checked_path(staging, relative)
        if nonempty(target):
            continue
        if os.path.lexists(target):
            raise ValueError("Refusing to replace an existing invalid target: " + str(target))
        if nonempty(candidate):
            verify_image(candidate)
            staged.append(relative)
        elif os.path.lexists(candidate):
            raise ValueError("Invalid existing staging image: " + str(candidate))
        else:
            missing.append(relative)
    archives = [row for row in manifest["archives"]
                if any(relative.startswith(row["log"] + "/") for relative in missing)]
    print(json.dumps({
        "mode": args.mode, "allowlisted": 46, "present": 46 - len(missing) - len(staged),
        "staged": len(staged), "missing": len(missing),
        "camera_archive_shards": [row["shard"] for row in archives],
        "archive_bytes_if_enabled": sum(row["size"] for row in archives),
        "next_action": "Use verified per-file URLs; whole archives require --allow-archive-transfer.",
    }), flush=True)
    if args.mode == "plan":
        return
    if not os.environ.get("JOB_ID"):
        raise ValueError("Run downloads, archive extraction and installation through qsub")
    if args.mode == "install":
        if missing:
            raise ValueError("Stage all missing images before installation")
        for relative in staged:
            target = checked_path(sensor_root, relative)
            with checked_path(staging, relative).open("rb") as source:
                publish(source, target)
        for relative in manifest["paths"]:
            verify_image(checked_path(sensor_root, relative))
        print(json.dumps({"installed": len(staged), "verified_allowlist": 46}))
        return
    if args.file_urls and args.allow_archive_transfer:
        raise ValueError("Choose either per-file URLs or whole-archive transfer")
    if args.file_urls:
        entries = json.loads(Path(args.file_urls).read_text())
        if not isinstance(entries, dict) or set(entries) - set(manifest["paths"]):
            raise ValueError("Per-file URL map must contain only allowlisted image paths")
        if set(missing) - set(entries):
            raise ValueError("Verified per-file URLs are missing for some requested images")
        for relative in missing:
            row = entries[relative]
            cached = fetch(row["url"], checked_path(staging / "downloads", relative),
                           row["size"], row["sha256"])
            with cached.open("rb") as source:
                publish(source, checked_path(staging, relative))
    else:
        cache = staging / "archives"
        for row in archives:
            archive = cache / ("camera_" + str(row["shard"]) + ".tgz")
            if not args.allow_archive_transfer and not archive.exists():
                raise ValueError("No per-file URLs/local archive; refusing redundant archive transfer")
        for row in archives:
            archive = cache / ("camera_" + str(row["shard"]) + ".tgz")
            archive = fetch(row["url"], archive, row["size"], row["sha256"])
            wanted = [relative for relative in missing if relative.startswith(row["log"] + "/")]
            extract_selected(archive, wanted, staging)
    for relative in manifest["paths"]:
        live = checked_path(sensor_root, relative)
        verify_image(live if nonempty(live) else checked_path(staging, relative))
    print(json.dumps({"newly_staged": len(missing), "verified_allowlist": 46,
                      "next_action": "Run install through qsub to add missing files to the live root."}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "stage", "install"), default="plan")
    parser.add_argument("--sensor-root")
    parser.add_argument("--staging", default="outputs/navtrain_camera_gap_staging")
    parser.add_argument("--file-urls", help="JSON: relative path -> {url, size, sha256}")
    parser.add_argument("--allow-archive-transfer", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, ImportError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, json.dumps({"error": str(error), "next_action": "Resolve the reported input/source issue and retry."}) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Find WOMD rasters that RAP would load as empty (all-black or undecodable).

`create_womd_metadata.py audit` only checks that every jpg exists. The loader
(navsim/common/dataclasses.py) opens it with PIL, crops [20:-20], and silently
substitutes zeros if decoding fails, so an existing file can still reach the
model as a black image. Two modes:

    # Inspect given anchors: file size, decode, pixel stats, plus a webp sheet
    python scripts/data_audit/scan_womd_rendered.py inspect \
        --ds-root /gs/bs/tga-RLA/qdeng/womd_dataset/rap_raster \
        --tokens 101c955dff88549e_135 101c955dff88549e_115

    # Scan every jpg: stat all, decode the small ones (black jpgs compress to
    # a fixed tiny size), count black / near-black / undecodable per camera
    python scripts/data_audit/scan_womd_rendered.py scan \
        --ds-root /gs/bs/tga-RLA/qdeng/womd_dataset/rap_raster --workers 16

    # Same statistics on NAVSIM's official rasters (cv2 default quality 95)
    python scripts/data_audit/scan_womd_rendered.py scan \
        --ds-root /gs/bs/tga-RLA/qdeng/navsim_workspace/dataset --split trainval \
        --jpeg-quality 95 --sample-logs 300 --workers 8

Results go to RAP/eval/womd_black/ (webp + json, both synced by mutagen).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
NEAR_BLACK_FRAC = 1e-3  # fewer lit pixels than this (after crop) counts as near-black
LIT_THRESHOLD = 16      # jpeg ringing around black stays below this


def pixel_stats(path: Path) -> Dict[str, object]:
    """Decode exactly like the loader and report what the model would see."""
    out: Dict[str, object] = {"bytes": path.stat().st_size if path.exists() else None}
    if out["bytes"] is None:
        out["status"] = "missing"
        return out
    try:
        img = np.array(Image.open(path))
    except Exception as exc:  # noqa: BLE001 - mirror the loader's bare except
        out.update(status="decode_fail", error=repr(exc))
        return out
    crop = img[20:-20]
    lit_full = float((img.max(axis=-1) > LIT_THRESHOLD).mean())
    lit_crop = float((crop.max(axis=-1) > LIT_THRESHOLD).mean())
    out.update(shape=list(img.shape), max_full=int(img.max()), max_crop=int(crop.max()),
               lit_frac_full=lit_full, lit_frac_crop=lit_crop,
               status="black" if not crop.any() else
               "near_black" if lit_crop < NEAR_BLACK_FRAC else "ok")
    return out


# =============================================================================
# inspect
# =============================================================================
def cmd_inspect(args: argparse.Namespace) -> None:
    split = args.split
    render_root = args.ds_root / "rendered_sensor_blobs" / split
    args.out.mkdir(parents=True, exist_ok=True)
    for token in args.tokens:
        scenario_id = token.rsplit("_", 1)[0]
        log_name = f"womd-{split.removeprefix('womd_')}-{scenario_id}"
        with open(args.ds_root / "navsim_logs" / split / f"{log_name}.pkl", "rb") as fp:
            infos = pickle.load(fp)
        info = next((x for x in infos if x["token"] == token), None)
        if info is None:
            print(f"{token}: 不在 {log_name}.pkl 里")
            continue
        anns = info["anns"]
        boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))))
        dists = np.hypot(boxes[:, 0], boxes[:, 1]) if len(boxes) else np.zeros(0)
        print(f"== {token}  is_valid={info['is_valid']}  cmd={list(info['driving_command'])}  "
              f"v={info['ego_dynamic_state'][:2]}  boxes={len(boxes)} (<30m: {int((dists < 30).sum())})  "
              f"lights={len(info['traffic_lights'])}")
        tiles = []
        for ch in CAMERAS:
            path = render_root / info["cams"][ch]["data_path"]
            stats = pixel_stats(path)
            print(f"  {ch}: {json.dumps(stats, ensure_ascii=False)}")
            img = cv2.imread(str(path)) if stats["status"] not in ("missing", "decode_fail") else None
            tile = np.zeros((280, 480, 3), np.uint8) if img is None else cv2.resize(img, (480, 280))
            cv2.putText(tile, f"{ch} {stats['status']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            tiles.append(tile)
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        target = args.out / f"{token}.webp"
        cv2.imwrite(str(target), sheet, [cv2.IMWRITE_WEBP_QUALITY, 90])
        print(f"  -> {target}")


# =============================================================================
# scan
# =============================================================================
def scan_log(job: Tuple[Path, int]) -> Tuple[Counter, List[Dict[str, object]], List[int]]:
    log_dir, decode_below = job
    counts: Counter = Counter()
    flagged: List[Dict[str, object]] = []
    sizes: List[int] = []
    for cam_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        with os.scandir(cam_dir) as it:
            for entry in it:
                if not entry.name.endswith(".jpg"):
                    continue
                size = entry.stat().st_size
                sizes.append(size)
                counts[f"{cam_dir.name}/files"] += 1
                if size >= decode_below:
                    continue
                stats = pixel_stats(Path(entry.path))
                counts[f"{cam_dir.name}/{stats['status']}"] += 1
                counts[f"{cam_dir.name}/decoded"] += 1
                if stats["status"] != "ok":
                    flagged.append({"path": f"{log_dir.name}/{cam_dir.name}/{entry.name}", **stats})
    return counts, flagged, sizes


def cmd_scan(args: argparse.Namespace) -> None:
    render_root = args.ds_root / "rendered_sensor_blobs" / args.split
    ok, black_jpg = cv2.imencode(".jpg", np.zeros((1120, 1920, 3), np.uint8),
                                 [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    decode_below = args.decode_below or int(1.5 * len(black_jpg))
    logs = sorted(p for p in render_root.iterdir() if p.is_dir())
    if args.sample_logs and args.sample_logs < len(logs):
        rng = np.random.default_rng(args.seed)
        logs = sorted(rng.choice(np.array(logs, dtype=object), args.sample_logs, replace=False))
    if args.limit_logs:
        logs = logs[: args.limit_logs]
    print(f"[scan] {len(logs)} logs, 纯黑 jpg = {len(black_jpg)} B，解码 < {decode_below} B 的文件")

    t0 = time.time()
    counts: Counter = Counter()
    flagged: List[Dict[str, object]] = []
    sizes: List[int] = []
    with Pool(args.workers) as pool:
        for i, (c, f, s) in enumerate(pool.imap_unordered(scan_log, [(p, decode_below) for p in logs],
                                                          chunksize=64), 1):
            counts.update(c)
            flagged.extend(f)
            sizes.extend(s)
            if i % 5000 == 0:
                print(f"  {i}/{len(logs)} logs  {time.time() - t0:.0f}s")

    size_pct = {f"p{q}": int(np.percentile(sizes, q)) for q in (0, 0.1, 1, 5, 50, 95, 100)} if sizes else {}
    summary = {"logs": len(logs), "black_jpg_bytes": len(black_jpg), "decode_below": decode_below,
               "size_bytes": size_pct, "counts": dict(sorted(counts.items())),
               "flagged": flagged[: args.max_examples], "flagged_total": len(flagged)}
    print(f"[scan] {len(sizes)} jpg，用时 {time.time() - t0:.0f}s，大小分位 {size_pct}")
    for ch in CAMERAS:
        n = counts[f"{ch}/files"]
        print(f"  {ch}: files={n} decoded={counts[f'{ch}/decoded']} black={counts[f'{ch}/black']} "
              f"near_black={counts[f'{ch}/near_black']} decode_fail={counts[f'{ch}/decode_fail']}")
    for item in flagged[:10]:
        print(f"  例: {item['path']} {item['status']} {item.get('bytes')}B lit={item.get('lit_frac_crop')}")
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"scan_{args.ds_root.name}_{args.split}.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[scan] -> {target}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "scan"):
        s = sub.add_parser(name)
        s.add_argument("--ds-root", type=Path, required=True)
        s.add_argument("--split", default="womd_training_20s")
        s.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "womd_black")
    sub.choices["inspect"].add_argument("--tokens", nargs="+", required=True)
    scan = sub.choices["scan"]
    scan.add_argument("--workers", type=int, default=16)
    scan.add_argument("--jpeg-quality", type=int, default=90, help="must match create_womd_metadata run")
    scan.add_argument("--decode-below", type=int, default=None, help="bytes; default 1.5x a black jpg")
    scan.add_argument("--limit-logs", type=int, default=None)
    scan.add_argument("--sample-logs", type=int, default=None, help="random subset of logs (e.g. NAVSIM)")
    scan.add_argument("--seed", type=int, default=0)
    scan.add_argument("--max-examples", type=int, default=500)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    {"inspect": cmd_inspect, "scan": cmd_scan}[args.command](args)


if __name__ == "__main__":
    main()

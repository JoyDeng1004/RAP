"""Validate the raw-image export produced by ``build_alignment_small_data.py``.

Image export is intentionally part of cache construction so the readable files
and the tensors cannot silently come from different cache roots.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "outputs/alignment_small/input_data"


def main() -> None:
    audit_path = INPUT_ROOT / "input_audit.json"
    manifest_path = INPUT_ROOT / "token_manifest.json"
    with audit_path.open(encoding="utf-8") as file:
        audit = json.load(file)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if audit.get("status") != "passed":
        raise RuntimeError(f"Input audit did not pass: {audit_path}")

    missing = []
    for split in ("train", "val"):
        for token in manifest[f"{split}_tokens"]:
            sample = INPUT_ROOT / split / token
            required = [sample / "metadata.json", sample / "target.json"]
            required.extend((sample / modality / f"{camera}.jpg")
                            for modality in ("real", "raster")
                            for camera in ("cam_b0", "cam_f0", "cam_l0", "cam_r0"))
            missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing exported input files:\n" + "\n".join(missing))
    print(
        f"Validated {len(manifest['train_tokens'])} train and "
        f"{len(manifest['val_tokens'])} val exports under {INPUT_ROOT}"
    )


if __name__ == "__main__":
    main()

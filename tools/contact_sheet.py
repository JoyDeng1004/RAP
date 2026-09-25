#!/usr/bin/env python3
"""把 create_nuscenes_metadata.py verify 产出的多路图拼成一张联络图。

只读脚本：只往 --out 写一张 jpg。

    python tools/contact_sheet.py exp/smoke/verify_navsim --mode navsim

verify 的文件名格式是 create_nuscenes_metadata.py:532 定的：

    verify_<sample_token>_<channel>_<camera_mode>.jpg

注意 channel 自身含下划线（CAM_F0），所以不能用 split("_")[-2] 取它 ——
那样只会拿到 "F0"。这里的做法是掐头去尾后按【第一个】下划线切，
token 里不含下划线（32 位 hex），所以这个切法是确定的。

布局按车身方位摆，方便一眼看出左右是否互为镜像：

    CAM_L0   CAM_F0   CAM_R0
    CAM_B0     --       --
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

TOP_ROW = ("CAM_L0", "CAM_F0", "CAM_R0")
BOTTOM_ROW = ("CAM_B0",)


def parse_verify_name(stem: str, mode: str) -> Optional[Tuple[str, str]]:
    """'verify_<token>_<channel>_<mode>' -> (token, channel)。不匹配返回 None。"""
    prefix, suffix = "verify_", f"_{mode}"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    core = stem[len(prefix): -len(suffix)]
    if "_" not in core:
        return None
    token, channel = core.split("_", 1)
    return token, channel


def collect(directory: Path, mode: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for path in sorted(directory.glob(f"verify_*_{mode}.jpg")):
        parsed = parse_verify_name(path.stem, mode)
        if parsed is None:
            continue
        found[parsed[1]] = path
    return found


def build(directory: Path, mode: str, panel_width: int, out: Path) -> None:
    import cv2
    import numpy as np

    found = collect(directory, mode)
    if not found:
        raise SystemExit(f"FAIL {directory} 下没有 verify_*_{mode}.jpg")

    wanted = TOP_ROW + BOTTOM_ROW
    missing = [c for c in wanted if c not in found]
    if missing:
        print(f"[sheet] 缺这几路: {missing}（有的: {sorted(found)}）")

    probe = cv2.imread(str(next(iter(found.values()))))
    if probe is None:
        raise SystemExit(f"FAIL 读不出图: {next(iter(found.values()))}")
    panel_height = int(probe.shape[0] * panel_width / probe.shape[1])
    blank = np.zeros((panel_height, panel_width, 3), np.uint8)

    def panel(channel: str) -> np.ndarray:
        if channel not in found:
            return blank.copy()
        image = cv2.imread(str(found[channel]))
        if image is None:
            return blank.copy()
        tile = cv2.resize(image, (panel_width, panel_height))
        nonzero = float(np.count_nonzero(image)) / image.size
        cv2.putText(tile, f"{channel}  nz={nonzero:.3%}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return tile

    def row(channels) -> np.ndarray:
        tiles = [panel(c) for c in channels]
        tiles += [blank.copy()] * (len(TOP_ROW) - len(tiles))
        return np.hstack(tiles)

    sheet = np.vstack([row(TOP_ROW), row(BOTTOM_ROW)])
    cv2.putText(sheet, f"camera-mode={mode}   src={directory}",
                (10, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[sheet] {mode}: {out}  {sheet.shape[1]}x{sheet.shape[0]}  "
          f"({len(found)} 路)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path, help="verify 的 --out-dir")
    parser.add_argument("--mode", required=True, help="navsim | native | hybrid")
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--out", type=Path, default=None,
                        help="默认 <directory>/../sheet_<mode>.jpg")
    args = parser.parse_args()
    out = args.out or args.directory.parent / f"sheet_{args.mode}.jpg"
    build(args.directory, args.mode, args.panel_width, out)


if __name__ == "__main__":
    main()

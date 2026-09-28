#!/usr/bin/env python3
"""Restore nominal NAVSIM-rig translations in legacy nuScenes metadata.

Only translations matching the canonical rig or that rig plus RAP_VIEWPOINT_SHIFT
are accepted. Without --apply, report changes without writing files.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_data.create_nuscenes_metadata import RAP_VIEWPOINT_SHIFT, load_navsim_rig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs_dir", type=Path, help="navsim_logs/<split> with one pkl per scene")
    ap.add_argument("--apply", action="store_true", help="write the patched pkls (default: dry run)")
    args = ap.parse_args()

    rig = load_navsim_rig()
    counts = {"patched": 0, "already_nominal": 0}
    pkls = sorted(args.logs_dir.glob("*.pkl"))
    for path in pkls:
        frames = pickle.loads(path.read_bytes())
        changed = False
        for info in frames:
            for ch, cam in info["cams"].items():
                t = np.asarray(cam["render_sensor2lidar_translation"], dtype=np.float64)
                nominal = np.asarray(rig[ch]["sensor2lidar_translation"], dtype=np.float64)
                if np.allclose(t, nominal + RAP_VIEWPOINT_SHIFT, atol=1e-6):
                    cam["render_sensor2lidar_translation"] = nominal.copy()
                    counts["patched"] += 1
                    changed = True
                elif np.allclose(t, nominal, atol=1e-6):
                    counts["already_nominal"] += 1
                else:
                    sys.exit(f"ABORT {path.name} {info['token']} {ch}: translation {t} is neither "
                             f"nominal {nominal} nor nominal+shift; not NAVSIM-rig metadata?")
        if changed and args.apply:
            tmp = path.with_suffix(".pkl.tmp")
            tmp.write_bytes(pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(tmp, path)
    print(f"pkl files: {len(pkls)}  cameras patched: {counts['patched']}  "
          f"already nominal: {counts['already_nominal']}  ({'applied' if args.apply else 'dry run'})")


if __name__ == "__main__":
    main()

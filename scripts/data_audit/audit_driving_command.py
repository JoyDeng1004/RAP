"""Audit driving_command consistency between NAVSIM (route-based) and nuScenes (track-based).

NAVSIM metadata derives the command from the nuPlan route centerline
(process_data/helpers/driving_command.py); the nuScenes metadata derives it from the
ego's own future track (nuscenes_adapter.driving_command_from_track), which can leak
the future trajectory. Read-only over metadata pkls; prints three tables:

  Q1  command class shares per source
  Q2  confusion of the stored command vs the track rule re-applied to the same frames
      (NAVSIM: route vs track semantics; nuScenes: sanity check, should be ~100%)
  Q3  how much the command reveals the answer: ego-frame lateral offset at +4 s,
      binned per stored command

Usage (TSUBAME):
  python scripts/data_audit/audit_driving_command.py \
      --navsim-logs $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
      --nusc-logs $OPENSCENE_DATA_ROOT/navsim_logs/nuscenes_trainval_native \
      --split navsim/planning/script/config/training/train_val_test_log_split/navsim_kshot_p010_seed0.yaml
"""
import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.nuscenes_adapter import (  # noqa: E402
    driving_command_from_track,
    quaternion_to_yaw,
)

NAMES = ("left", "forward", "right", "unknown")
LAT_BAND = 0.5  # metres; |lateral| below this counts as "straight" in Q3


def audit_log(args):
    """Per-log counts: stored class, stored-vs-rule confusion, lateral bins at +4 s."""
    path, lookahead_s, lateral_horizon_s = args
    with open(path, "rb") as f:
        frames = sorted(pickle.load(f), key=lambda fr: fr["timestamp"])
    out = {"frames": 0, "dt": [], "cls": np.zeros(4, int),
           "conf": np.zeros((4, 4), int), "lat": np.zeros((4, 3), int)}
    if len(frames) < 2:
        return out
    xy = np.array([fr["ego2global_translation"][:2] for fr in frames], dtype=np.float64)
    yaw = np.array([quaternion_to_yaw(fr["ego2global_rotation"]) for fr in frames])
    ts = np.array([fr["timestamp"] for fr in frames], dtype=np.float64) * 1e-6
    dt = float(np.median(np.diff(ts)))
    look = max(int(round(lookahead_s / dt)), 1)
    k4 = int(round(lateral_horizon_s / dt))
    out["dt"].append(dt)
    for i, fr in enumerate(frames):
        stored = int(np.argmax(fr["driving_command"]))
        rule = int(np.argmax(driving_command_from_track(xy, yaw, i, lookahead_frames=look)))
        out["frames"] += 1
        out["cls"][stored] += 1
        out["conf"][stored, rule] += 1
        if i + k4 < len(frames):
            d = xy[i + k4] - xy[i]
            lat = -np.sin(yaw[i]) * d[0] + np.cos(yaw[i]) * d[1]  # +left
            out["lat"][stored, 0 if lat > LAT_BAND else (2 if lat < -LAT_BAND else 1)] += 1
    return out


def run(paths, workers, lookahead_s, lateral_horizon_s):
    total = {"frames": 0, "dt": [], "cls": np.zeros(4, int),
             "conf": np.zeros((4, 4), int), "lat": np.zeros((4, 3), int)}
    jobs = [(p, lookahead_s, lateral_horizon_s) for p in paths]
    with Pool(workers) as pool:
        for r in pool.imap_unordered(audit_log, jobs, chunksize=4):
            total["frames"] += r["frames"]
            total["dt"] += r["dt"]
            for key in ("cls", "conf", "lat"):
                total[key] += r[key]
    return total


def report(name, n_logs, t):
    print(f"\n## {name}: {n_logs} logs, {t['frames']} frames, "
          f"median dt {np.median(t['dt']) if t['dt'] else float('nan'):.3f}s")
    shares = t["cls"] / max(t["cls"].sum(), 1)
    print("Q1 stored class share: " + ", ".join(f"{n}={s:.1%}" for n, s in zip(NAMES, shares)))
    conf = t["conf"]
    known = conf[:3].sum()
    agree_known = np.trace(conf[:3, :3]) / max(known, 1)
    print(f"Q2 stored(rows) vs track rule(cols); agreement on stored-known frames = "
          f"{agree_known:.1%}, all frames = {np.trace(conf) / max(conf.sum(), 1):.1%}")
    print("| stored \\ rule | " + " | ".join(NAMES) + " |")
    print("|---|" + "---|" * 4)
    for i, n in enumerate(NAMES):
        print(f"| {n} | " + " | ".join(str(v) for v in conf[i]) + " |")
    print(f"Q3 lateral offset at +4 s per stored command (band ±{LAT_BAND} m):")
    print("| stored | n | lat>+band (left) | abs(lat)<=band | lat<-band (right) |")
    print("|---|---|---|---|---|")
    for i, n in enumerate(NAMES):
        row = t["lat"][i]
        s = max(row.sum(), 1)
        print(f"| {n} | {row.sum()} | {row[0] / s:.1%} | {row[1] / s:.1%} | {row[2] / s:.1%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--navsim-logs", required=True, help="dir of NAVSIM trainval log pkls")
    ap.add_argument("--nusc-logs", required=True, help="dir of nuScenes metadata pkls")
    ap.add_argument("--split", required=True, help="train_val_test_log_split yaml (train_logs/val_logs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lookahead-s", type=float, default=10.0,
                    help="track-rule look-ahead; nuScenes metadata used 20 frames at 2 Hz = 10 s")
    ap.add_argument("--lateral-horizon-s", type=float, default=4.0)
    a = ap.parse_args()

    split = yaml.safe_load(open(a.split))
    nav = Path(a.navsim_logs)
    for name, logs in (("NAVSIM train (split train_logs)", split["train_logs"]),
                       ("NAVSIM val (split val_logs)", split["val_logs"])):
        paths = [nav / f"{l.strip()}.pkl" for l in logs]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            print(f"[warn] {name}: {len(missing)} log pkls missing, e.g. {missing[0]}")
        paths = [p for p in paths if p.is_file()]
        report(name, len(paths), run(paths, a.workers, a.lookahead_s, a.lateral_horizon_s))

    nusc = sorted(Path(a.nusc_logs).glob("*.pkl"))
    report("nuScenes (all scene pkls)", len(nusc), run(nusc, a.workers, a.lookahead_s, a.lateral_horizon_s))


if __name__ == "__main__":
    main()

"""T2 factor experiment: summarize run_t2_probe_v1.py runs of one tag into eval/t2_results/t2_summary_<tag>.md.

All ΔPDMS are paired on the same tokens, in points (×100), with 95% log-clustered bootstrap CIs
(summarize_t1.boot_ci). Reference: c1 = raster under the identity rig (official convention).

Sections
  1  per preset: Δ new, Δ fix_all (c1 candidates re-ranked under the new rig: selection only),
     Δ oracle@64, Δ mean64 (candidate pool), and the image statistics of the render
  2  pre-registered contrasts: calibration benefit (updated - stale, same images), view-mask
     rescue (mask - full, same geometry), height (pure position) vs focal / depth (scale)
  3  mechanism: round 1 / layer 0 flip statistics, feature cosine at the 4 distinct reference points,
     query cosine per round, proposal ADE / pool coverage
  4  token level: Spearman between r1l0 flip-in mass and ΔPDMS, log-clustered bootstrap CI
  5  validity: c1 vs real, raster f_1.25 / drop_f0 vs the T1 real-image runs of the same presets
  6  pre-registered verdicts (H0, H_A, H_B, H_enc, H_cal), computed from the tables above
  7  H_par confirmation (criteria registered 10/22 before these runs): height dose curve, feature cosine
     per waypoint (H_par: worse far away under height changes only), planar map-only scene
  8  H_scale test (criteria registered 10/23 before these runs): output equivalence of a moved rig and a
     scaled map in the planar scene, and the behavioural signature of a misread metric scale

    python tools/summarize_t2.py --tag full
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, load, token_logs  # noqa: E402

BASE = ("h_p1.0", "h_m0.7", "d_p1.0", "d_m1.0", "l_p0.5", "l_m0.5", "f_0.8", "f_1.25")
MASKED = ("h_m0.7", "h_p1.0", "d_m1.0", "f_0.8")
HEIGHT, SCALE = ("h_p1.0", "h_m0.7"), ("f_0.8", "f_1.25", "d_p1.0", "d_m1.0")
# Pre-registered thresholds (PDMS fraction, i.e. 0.015 = 1.5 points)
H0_MAX_ABS = 0.015
HA_RESCUE = 0.60
HB_SCALE_MIN, HB_HEIGHT_MAX = 0.02, 0.01
HENC_COS = 0.90
# H_par (registered 2026-10-22, before the dose / map-only runs)
DOSE = (("h_m0.7", -0.7), ("h_m0.5", -0.5), ("h_m0.3", -0.3), ("h_p0.3", 0.3), ("h_p0.5", 0.5), ("h_p1.0", 1.0))
MAP_PAIRS = (("h_m0.7", "map_h_m0.7"), ("h_p1.0", "map_h_p1.0"), ("d_m1.0", "map_d_m1.0"))
HPAR_MAP_KEEP = 0.5  # map-only keeps >= half of the full-scene height drop
HPAR_MAP_D_MAX = 0.015  # map-only longitudinal control: |Δ| CI bound < 1.5
EARLY, LATE = (0, 1), (5, 6, 7)  # waypoint indices (0.5 s steps) for the distance trend
H_F0 = 1.49112208 + 0.8  # raster F0 height above the map plane: renderer camera_params z + OFFICIAL_SHIFT
POINT_Z = (-2.0, 2.0 / 3.0, 10.0 / 3.0, 6.0)  # the 4 distinct reference points (encoder.py:252-256)


# H_scale (registered 2026-10-23, before the mapscale runs). A planar scene seen from height H + dh is the
# scene scaled by k = H / (H + dh) seen from H, so a model that reads metric scale off the ground with its
# training camera height cannot tell the two apart.
SCALE_PAIRS = (("map_h_m0.3", "mapscale_h_m0.3"), ("map_h_m0.7", "mapscale_h_m0.7"), ("map_h_p1.0", "mapscale_h_p1.0"))
DECISIVE = ("map_h_m0.7", "map_h_p1.0")
S1_RATIO_MAX = 0.5  # median ADE(moved, scaled) / median ADE(moved, c1)
S1_GAP_REL, S1_GAP_ABS = 0.33, 0.01  # |Δ scaled − Δ moved| <= max(0.33 |Δ moved|, 1 point)
S2_PRESETS = tuple(n for n, _ in DOSE)
S2_MIN_CI = 4  # of the 6 height presets, CI excluding 0 in the predicted direction
LEN_MIN, LAT_MIN = 2.0, 1.0  # m: c1 4 s displacement / |lateral| below which the ratio is not used


def chosen(z, var: str) -> pd.Series:
    """Picked trajectory (first 8 poses, xy) per token from a run's arrays: argmax of the scorer."""
    cand, score = z[f"cand_{var}"], z[f"score_{var}"]
    pick = cand[np.arange(len(cand)), score.argmax(1), :8, :2]  # (N, 8, 2)
    return pd.Series(list(pick), index=pd.Index(z["token"]))


def ade(a: pd.Series, b: pd.Series) -> pd.Series:
    common = a.index.intersection(b.index)
    return pd.Series([float(np.linalg.norm(a[t] - b[t], axis=-1).mean()) for t in common], index=common)


def ground_hit(z_pt: float, h: float, dist: float) -> float:
    """Distance at which the camera ray through a point at height z_pt, dist m ahead, meets the ground."""
    return dist * h / (h - z_pt) if h > z_pt else float("inf")


def ci(d: pd.Series, logs, rng, n_boot):
    d = d.dropna()
    if d.empty:
        return None
    return boot_ci(d, logs, n_boot, rng)


def cell(t) -> str:
    return "–" if t is None else f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


def mean_cell(s: pd.Series, fmt: str = "{:.3f}") -> str:
    s = s.dropna() if s is not None else None
    return "–" if s is None or s.empty else fmt.format(s.mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def spearman_ci(x: pd.Series, y: pd.Series, logs: pd.Series, n_boot: int, rng):
    df = pd.DataFrame({"x": x, "y": y, "log": logs.reindex(x.index)}).dropna()
    if len(df) < 10:
        return None
    groups = [g.index.to_numpy() for _, g in df.groupby("log")]
    pos = {t: i for i, t in enumerate(df.index)}
    gidx = [np.array([pos[t] for t in g]) for g in groups]
    xv, yv = df["x"].to_numpy(), df["y"].to_numpy()
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(gidx), len(gidx))
        sel = np.concatenate([gidx[i] for i in pick])
        boots.append(spearman(xv[sel], yv[sel]))
    return spearman(xv, yv), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--t1-tag", default="official", help="T1 sweep (real images) for the validity rows")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "t2_results")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    logs = token_logs(args.subset_csv)

    runs, npzs = {}, {}
    pat = re.compile(rf"^t2probe_{re.escape(args.tag)}_(.+)$")
    for d in sorted(args.exp_root.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir() and (csv := latest_csv(d)) is not None:
            df = load(csv)
            runs[m.group(1)] = (csv, df[df["valid"].astype(bool)], int((~df["valid"].astype(bool)).sum()))
            npz = csv.with_name(csv.stem + "_arrays.npz")
            if npz.exists():
                npzs[m.group(1)] = np.load(npz)
    if not runs:
        raise SystemExit(f"no t2probe_{args.tag}_* runs under {args.exp_root}")

    def d_new(name):
        ok = runs[name][1]
        return ok["pdms_new"] - ok["pdms_c1"]

    # 1 per preset
    rows1, stats = [], {}
    for name, (csv, ok, n_bad) in sorted(runs.items()):
        stats[name] = ci(d_new(name), logs, rng, args.n_boot)
        rows1.append(
            f"| {name} | {len(ok)} | {n_bad} | {cell(stats[name])} | "
            f"{cell(ci(ok['pdms_fix_all'] - ok['pdms_c1'], logs, rng, args.n_boot))} | "
            f"{cell(ci(ok['oracle_new'] - ok['oracle_c1'], logs, rng, args.n_boot))} | "
            f"{cell(ci(ok['mean64_new'] - ok['mean64_c1'], logs, rng, args.n_boot))} | "
            f"{mean_cell(ok.get('img_new_cam_f0_nonblack_frac'))} | {mean_cell(ok.get('img_new_cam_f0_masked_frac'), '{:.4f}')} |")

    # 2 contrasts (paired on common tokens)
    def paired(a, b, col="pdms_new"):
        x, y = runs[a][1][col], runs[b][1][col]
        common = x.index.intersection(y.index)
        return x[common] - y[common]

    rows_cal, cal = [], {}
    for b in BASE:
        if b in runs and f"stale_{b}" in runs:
            cal[b] = ci(paired(b, f"stale_{b}"), logs, rng, args.n_boot)
            rows_cal.append(f"| {b} | {cell(stats.get(b))} | {cell(stats.get(f'stale_{b}'))} | {cell(cal[b])} |")
    rows_mask, rescue = [], {}
    for b in MASKED:
        if b in runs and f"mask_{b}" in runs:
            r = ci(paired(f"mask_{b}", b), logs, rng, args.n_boot)
            full = stats[b][0]
            frac = r[0] / -full if full < -1e-9 else float("nan")
            rescue[b] = (r, frac)
            rows_mask.append(f"| {b} | {cell(stats[b])} | {cell(stats.get(f'mask_{b}'))} | {cell(r)} | "
                             f"{'–' if np.isnan(frac) else f'{100 * frac:.0f}%'} |")

    # 3 mechanism
    rows3 = []
    for name, (csv, ok, _) in sorted(runs.items()):
        g = lambda c, f="{:.3f}": mean_cell(ok.get(c), f)  # noqa: E731
        rows3.append(
            f"| {name} | {g('r1l0_sink_ref')} | {g('r1l0_sink_new')} | {g('r1l0_flip_in', '{:.4f}')} | "
            f"{g('r1l0_flip_in_obj', '{:.4f}')} | {g('r1l0_flip_out', '{:.4f}')} | "
            f"{g('r1l0_lost_query_frac')} / {g('r1l0_gained_query_frac')} | {g('anchor_cos_new_c1')} | "
            + " / ".join(g(f"anchor_cos_p{i}_new_c1") for i in range(4)) + " | "
            + " / ".join(g(f"qcos_r{r}_new_c1") for r in range(1, 5)) + f" | {g('pade_r4_new_c1', '{:.2f}')} | "
            f"{g('cover_new_c1', '{:.2f}')} |")

    # 4 token level
    rows4 = []
    for name, (csv, ok, _) in sorted(runs.items()):
        if "r1l0_flip_in" not in ok or ok["r1l0_flip_in"].fillna(0).abs().sum() == 0:
            continue
        s = spearman_ci(ok["r1l0_flip_in"], ok["pdms_new"] - ok["pdms_c1"], logs, args.n_boot, rng)
        s2 = spearman_ci(ok["r1l0_flip_in_obj"], ok["pdms_new"] - ok["pdms_c1"], logs, args.n_boot, rng)
        fmt = lambda t: "–" if t is None else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"  # noqa: E731
        rows4.append(f"| {name} | {fmt(s)} | {fmt(s2)} |")

    # 5 validity
    rows5 = []
    any_name = next((k for k in sorted(runs) if not k.startswith("map_")), next(iter(runs)))
    any_run = runs[any_name][1]
    rows5.append(f"| c1 − real (raster vs real, identity rig, all tokens of {any_name}) | "
                 f"{cell(ci(any_run['pdms_c1'] - any_run['pdms_real'], logs, rng, args.n_boot))} | – |")
    for t2name, t1name in (("f_1.25", "focal_1.25"), ("f_0.8", "focal_0.8"), ("drop_f0", "drop_black_f0")):
        t1d, rawd = args.exp_root / f"t1_sub_{args.t1_tag}_{t1name}", args.exp_root / f"t1_sub_{args.t1_tag}_raw"
        real = "–"
        if t1d.is_dir() and rawd.is_dir() and latest_csv(t1d) and latest_csv(rawd):
            a, b = load(latest_csv(t1d))["score"], load(latest_csv(rawd))["score"]
            common = a.index.intersection(b.index)
            real = cell(ci(a[common] - b[common], logs, rng, args.n_boot))
        rows5.append(f"| {t2name} raster vs T1 real {t1name} | {cell(stats.get(t2name))} | {real} |")

    # 6 verdicts
    v = []
    upd = [b for b in BASE if b in stats and stats[b] is not None]
    if upd:
        worst = max(max(abs(stats[b][1]), abs(stats[b][2])) for b in upd)
        v.append(f"- **H0** (RAP robust, all updated-calibration presets |Δ| CI bound < {100 * H0_MAX_ABS:.1f}): "
                 f"{'SUPPORTED' if worst < H0_MAX_ABS else 'rejected'} (largest CI bound {100 * worst:.1f})")
    if rescue:
        fr = {b: f for b, (_, f) in rescue.items()}
        ok_a = [b for b, f in fr.items() if not np.isnan(f) and f >= HA_RESCUE]
        v.append(f"- **H_A** (view mask rescues ≥ {100 * HA_RESCUE:.0f}%): rescued in {ok_a or 'none'} "
                 f"(rescue fractions {', '.join(f'{b} {100 * f:.0f}%' for b, f in fr.items() if not np.isnan(f))}); "
                 "also needs a negative flip-in / ΔPDMS Spearman in section 4")
    hs = [b for b in HEIGHT if b in stats and stats[b]]
    sc = [b for b in SCALE if b in stats and stats[b]]
    if hs and sc:
        h_max = max(-stats[b][0] for b in hs)
        s_after = {b: (stats.get(f"mask_{b}") or stats[b])[0] for b in sc}
        s_max = max(-x for x in s_after.values())
        v.append(f"- **H_B** (scale presets drop ≥ {100 * HB_SCALE_MIN:.0f} after masking, height < {100 * HB_HEIGHT_MAX:.0f}): "
                 f"{'SUPPORTED' if s_max >= HB_SCALE_MIN and h_max < HB_HEIGHT_MAX else 'not supported'} "
                 f"(largest scale drop {100 * s_max:.1f}, largest height drop {100 * h_max:.1f})")
    enc = {b: runs[b][1]["anchor_cos_new_c1"].mean() for b in BASE if b in runs and "anchor_cos_new_c1" in runs[b][1]}
    if enc:
        low = [b for b, c in enc.items() if c < HENC_COS]
        v.append(f"- **H_enc** (feature cosine at the same 3D reference points < {HENC_COS}): below in {low or 'none'} "
                 f"({', '.join(f'{b} {c:.3f}' for b, c in enc.items())}); compare with c1 vs real "
                 f"{any_run['anchor_cos_c1_real'].mean():.3f} (modality gap at the identity rig)")
    if cal:
        bad = [b for b, t in cal.items() if t and t[0] <= 0]
        v.append(f"- **H_cal** (correct calibration helps ≤ 0): updated ≤ stale in {bad or 'none'} of {list(cal)}")

    # 7 H_par confirmation
    rows7a, rows7b, rows7c, v7 = [], [], [], []
    for name, dh in DOSE:
        if name not in runs:
            continue
        ok = runs[name][1]
        shift = " / ".join(f"{ground_hit(z, H_F0 + dh, 20.0) - ground_hit(z, H_F0, 20.0):+.1f}" for z in POINT_Z[:2])
        rows7a.append(f"| {name} | {dh:+.1f} | {cell(stats.get(name))} | "
                      f"{cell(ci(ok['mean64_new'] - ok['mean64_c1'], logs, rng, args.n_boot))} | "
                      f"{mean_cell(ok.get('anchor_cos_new_c1'))} | "
                      + " / ".join(mean_cell(ok.get(f"anchor_cos_p{i}_new_c1")) for i in range(4))
                      + f" | {shift} |")
    sides = ([n for n, d in DOSE if d < 0][::-1], [n for n, d in DOSE if d > 0])  # increasing |Δh|
    mono = []
    for side in sides:
        side = [n for n in side if n in stats and stats[n]]
        if len(side) >= 2:
            est = [stats[n][0] for n in side]
            mono.append(all(a >= b for a, b in zip(est, est[1:])))
    if mono:
        v7.append(f"- **P1 dose** (drop grows with |Δh| on each side, point estimates): "
                  f"{'SUPPORTED' if all(mono) else 'not supported'} (per side {mono})")
    trend = {}
    for name in [n for n, _ in DOSE] + ["d_p1.0", "d_m1.0", "l_p0.5", "l_m0.5", "f_0.8", "f_1.25"]:
        z = npzs.get(name)
        if z is None or "pose_pt_new_c1" not in z:
            continue
        a = z["pose_pt_new_c1"]  # (N, 4 points, T waypoints)
        with np.errstate(invalid="ignore"):
            per_wp = np.nanmean(a, axis=(0, 1))
        trend[name] = float(np.nanmean(per_wp[list(LATE)]) - np.nanmean(per_wp[list(EARLY)]))
        rows7b.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in per_wp[:8]) + f" | {trend[name]:+.3f} |")
    th = [trend[n] for n, _ in DOSE if n in trend]
    td = [trend[n] for n in ("d_p1.0", "d_m1.0") if n in trend]
    if th and td:
        v7.append(f"- **P2 distance** (late − early anchor cos more negative for every height preset than for every "
                  f"longitudinal one): {'SUPPORTED' if max(th) < min(td) else 'not supported'} "
                  f"(height {min(th):+.3f}..{max(th):+.3f}, longitudinal {min(td):+.3f}..{max(td):+.3f})")
    keep, d_ctrl = [], None
    for full, mp in MAP_PAIRS:
        if full not in stats or mp not in stats or not stats[full] or not stats[mp]:
            continue
        frac = stats[mp][0] / stats[full][0] if stats[full][0] < -1e-9 else float("nan")
        c1map = runs[mp][1]["pdms_c1"].mean()
        rows7c.append(f"| {full} | {cell(stats[full])} | {cell(stats[mp])} | "
                      f"{'–' if np.isnan(frac) else f'{100 * frac:.0f}%'} | {100 * c1map:.1f} |")
        if full.startswith("h_"):
            keep.append(frac)
        else:
            d_ctrl = max(abs(stats[mp][1]), abs(stats[mp][2]))
    if keep:
        ok_keep = all(not np.isnan(f) and f >= HPAR_MAP_KEEP for f in keep)
        ok_ctrl = d_ctrl is None or d_ctrl < HPAR_MAP_D_MAX
        v7.append(f"- **P3 map-only** (planar scene keeps ≥ {100 * HPAR_MAP_KEEP:.0f}% of each height drop, longitudinal "
                  f"control |Δ| CI bound < {100 * HPAR_MAP_D_MAX:.1f}): {'SUPPORTED' if ok_keep and ok_ctrl else 'not supported'} "
                  f"(kept {', '.join(f'{100 * f:.0f}%' for f in keep)}; control bound "
                  f"{'–' if d_ctrl is None else f'{100 * d_ctrl:.1f}'})")

    # 8 H_scale
    f0 = H_F0
    rows8a, rows8b, v8 = [], [], []
    s1_ok = {}
    for moved, scaled in SCALE_PAIRS:
        if moved not in npzs or scaled not in npzs or moved not in stats or scaled not in stats:
            continue
        tm, ts, tc = chosen(npzs[moved], "new"), chosen(npzs[scaled], "new"), chosen(npzs[moved], "c1")
        eq, ch, sc_c1 = ade(tm, ts), ade(tm, tc), ade(ts, tc)
        ratio = float(eq.median() / ch.median()) if ch.median() > 0 else float("nan")
        gap = ci(paired(scaled, moved), logs, rng, args.n_boot)
        d_m = stats[moved][0]
        ok = (not np.isnan(ratio) and ratio <= S1_RATIO_MAX and gap is not None
              and abs(gap[0]) <= max(S1_GAP_REL * abs(d_m), S1_GAP_ABS))
        s1_ok[moved] = ok
        eqv = runs[scaled][1]
        rows8a.append(f"| {moved} vs {scaled} | {cell(stats[moved])} | {cell(stats[scaled])} | {cell(gap)} | "
                      f"{eq.median():.2f} | {ch.median():.2f} | {sc_c1.median():.2f} | {ratio:.2f} | "
                      f"{mean_cell(eqv.get('equiv_f0_iou'))} / {mean_cell(eqv.get('equiv_f0_iou_tol1'))} / "
                      f"{mean_cell(eqv.get('equiv_f0_mad'), '{:.1f}')} | {'yes' if ok else 'no'} |")
    if all(m in s1_ok for m in DECISIVE):
        v8.append(f"- **S1 output equivalence** (both {', '.join(DECISIVE)}: ADE ratio ≤ {S1_RATIO_MAX} and "
                  f"|Δ scaled − Δ moved| ≤ max({S1_GAP_REL:.2f}·|Δ moved|, {100 * S1_GAP_ABS:.0f} pt)): "
                  f"{'SUPPORTED' if all(s1_ok[m] for m in DECISIVE) else 'not supported'} ({s1_ok})")
    sig = {"len": [], "lat": []}
    for name in list(S2_PRESETS) + ["d_p1.0", "d_m1.0"]:
        if name not in npzs:
            continue
        dh = dict(DOSE).get(name)
        tn, tc = chosen(npzs[name], "new"), chosen(npzs[name], "c1")
        common = tn.index.intersection(tc.index)
        p_new = np.stack([tn[t][-1] for t in common])
        p_c1 = np.stack([tc[t][-1] for t in common])
        l_new, l_c1 = np.linalg.norm(p_new, axis=1), np.linalg.norm(p_c1, axis=1)
        y_new, y_c1 = np.abs(p_new[:, 1]), np.abs(p_c1[:, 1])
        L = pd.Series(np.log(np.maximum(l_new, 1e-3) / l_c1), index=common)[l_c1 > LEN_MIN]
        Y = pd.Series(np.log(np.maximum(y_new, 1e-3) / y_c1), index=common)[y_c1 > LAT_MIN]
        cl, cy = ci(L, logs, rng, args.n_boot), ci(Y, logs, rng, args.n_boot)
        pred = "–" if dh is None else f"{np.log(f0 / (f0 + dh)):+.3f}"
        rows8b.append(f"| {name} | {'–' if dh is None else f'{dh:+.1f}'} | {pred} | "
                      f"{'–' if cl is None else f'{cl[0]:+.3f} [{cl[1]:+.3f}, {cl[2]:+.3f}]'} ({len(L)}) | "
                      f"{'–' if cy is None else f'{cy[0]:+.3f} [{cy[1]:+.3f}, {cy[2]:+.3f}]'} ({len(Y)}) |")
        if dh is not None:
            for key, t in (("len", cl), ("lat", cy)):
                if t is not None:
                    want = -np.sign(dh)  # lowered (dh < 0): the world looks bigger -> longer, wider
                    sig[key].append((np.sign(t[0]) == want, (t[1] > 0) if want > 0 else (t[2] < 0)))
    for key, label in (("len", "4 s displacement"), ("lat", "lateral offset")):
        if len(sig[key]) == len(S2_PRESETS):
            signs = all(a for a, _ in sig[key])
            n_ci = sum(b for _, b in sig[key])
            v8.append(f"- **S2 {label}** (predicted sign in all {len(S2_PRESETS)} height presets, CI excludes 0 in ≥ "
                      f"{S2_MIN_CI}): {'SUPPORTED' if signs and n_ci >= S2_MIN_CI else 'not supported'} "
                      f"(signs right {sum(a for a, _ in sig[key])}/{len(sig[key])}, CIs {n_ci})")

    lines = [f"# T2 factor experiment — tag {args.tag}", "",
             "Official RAP v1 weights, NAVSIM navtest t1sub tokens, forward only. ΔPDMS in points (×100) vs c1 "
             "(raster, identity rig, official convention) on the same tokens, 95% log-clustered bootstrap CI.", "",
             "## 1 per preset", "",
             "fix_all: c1 candidates teacher-forced, all rounds read the new images (selection only). "
             "F0 nonblack: fraction of drawn raster pixels; masked: fraction blacked out by the view mask.", "",
             "| preset | n | invalid | Δ new | Δ fix_all | Δ oracle@64 | Δ mean64 | F0 nonblack | F0 masked |",
             "|---|---|---|---|---|---|---|---|---|", *rows1, "",
             "## 2 pre-registered contrasts", "",
             "Calibration benefit = PDMS(updated) − PDMS(stale), same images, per token.", "",
             "| rig change | Δ updated | Δ stale | updated − stale |", "|---|---|---|---|", *rows_cal, "",
             "View-mask rescue = PDMS(mask) − PDMS(full), same geometry and calibration; rescued = rescue / −Δ full.", "",
             "| rig change | Δ full | Δ masked | mask − full | rescued |", "|---|---|---|---|---|", *rows_mask, "",
             "## 3 mechanism (F0, round 1 / layer 0 unless noted)", "",
             "sink: attention mass outside the image or on the padding rows; flip in: mass whose sample moved from "
             "sink (c1) to content (new); obj: of which on a drawn raster primitive; lost / gained: F0 queries that stop / "
             "start seeing the camera. anchor cos: feature cosine at the projections of the same round-1 reference "
             "points, all / per distinct point (p0 FL z=−2, p1 FR z=0.67, p2 RR z=3.33, p3 RL z=6 m).", "",
             "| preset | sink c1 | sink new | flip in | flip in obj | flip out | lost / gained | anchor cos | p0 / p1 / p2 / p3 | query cos r1 / r2 / r3 / r4 | prop ADE r4 | cover |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows3, "",
             "## 4 token level", "",
             "Spearman between the flip-in mass and ΔPDMS (new − c1) over tokens; CI by resampling logs. "
             "H_A predicts a negative correlation.", "",
             "| preset | flip in vs ΔPDMS | flip in obj vs ΔPDMS |", "|---|---|---|", *rows4, "",
             "## 5 validity", "",
             "| check | Δ raster (T2) | Δ real (T1) |", "|---|---|---|", *rows5, "",
             "## 6 pre-registered verdicts", "", *v, "",
             "Raster has no texture: H_A can be underestimated (newly visible content is mostly black background) "
             "and H_enc is a raster-only measure. Claims about real deployment need the real-image experiments.", ""]
    if rows7a or rows7b or rows7c:
        lines += ["## 7 H_par confirmation (criteria registered 2026-10-22)", "",
                  "H_par: RAP's 4 distinct reference points per waypoint sit off the ground (z = −2 / 0.67 / 3.33 / 6 m); "
                  "the image at their projection shows the surface behind them, and where the ray meets the ground "
                  "depends on camera height. Shift = change of that ground distance for a point 20 m ahead "
                  f"(p0 / p1), raster F0 height {H_F0:.2f} m.", "",
                  "### 7a height dose curve", "",
                  "| preset | Δh (m) | Δ new | Δ mean64 | anchor cos | p0 / p1 / p2 / p3 | ground shift @20 m p0 / p1 (m) |",
                  "|---|---|---|---|---|---|---|", *rows7a, "",
                  "### 7b anchor cos per waypoint (mean over the 4 points; waypoints 0.5–4 s)", "",
                  "| preset | 0.5 s | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 s | late − early |",
                  "|---|---|---|---|---|---|---|---|---|---|", *rows7b, "",
                  "### 7c planar map-only scene (boxes and lights removed in c1 and new)", "",
                  "Δ map-only is against the map-only c1 of the same tokens; kept = Δ map-only / Δ full.", "",
                  "| rig change | Δ full | Δ map-only | kept | PDMS of map-only c1 |", "|---|---|---|---|---|", *rows7c, "",
                  "### verdicts", "", *v7, ""]
    if rows8a or rows8b:
        lines += ["## 8 H_scale test (criteria registered 2026-10-23)", "",
                  "H_scale: a planar scene seen from height H + dh equals the scene scaled by k = H / (H + dh) seen from "
                  f"H (exact for CAM_F0, H = {f0:.2f} m); if RAP reads metric scale off the ground with its training "
                  "height, a moved rig and a scaled map give the same plan. All map-only (no boxes / lights).", "",
                  "### 8a output equivalence", "",
                  "ADE (m) between picked trajectories (8 poses, 4 s): moved vs scaled, moved vs c1, scaled vs c1; "
                  "ratio = moved-vs-scaled / moved-vs-c1 (medians). Render equivalence of the F0 inputs: drawn-pixel IoU "
                  "/ with 1 px tolerance / mean |Δ| on drawn pixels (depth shading is not scale invariant).", "",
                  "| pair | Δ moved | Δ scaled | Δ scaled − Δ moved | ADE moved–scaled | ADE moved–c1 | ADE scaled–c1 | ratio | render IoU / tol1 / MAD | S1 |",
                  "|---|---|---|---|---|---|---|---|---|---|", *rows8a, "",
                  "### 8b behavioural signature (full scene)", "",
                  "Mean log ratio new / c1 of the picked trajectory's 4 s displacement (tokens with c1 > "
                  f"{LEN_MIN:.0f} m) and |lateral| at 4 s (tokens with c1 > {LAT_MIN:.0f} m), (n tokens). "
                  "Prediction at full strength: log k; lowered rigs positive.", "",
                  "| preset | Δh | predicted log k | log length ratio | log lateral ratio |", "|---|---|---|---|---|", *rows8b, "",
                  "### verdicts", "", *v8, ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"t2_summary_{args.tag}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(runs)} presets)")


if __name__ == "__main__":
    main()

"""T2 feature-swap intervention: summarize the run_t2_swap_v1.py runs of one tag into
eval/t2_results/t2_swap_<tag>.md.

ΔPDMS in points (×100) vs c1, paired on the same tokens, 95% log-clustered bootstrap CIs
(summarize_t1.boot_ci). rescue = PDMS(swap) − PDMS(new); share = mean rescue / mean loss (c1 − new).

Pre-registered reading (height presets h_p1.0 and h_m0.7, variant swap_all = same-surface-invariant
encoder on every rendered camera; both presets must agree):
  ENC-DEAD    rescue CI high <= 25% of the loss  ->  a same-surface feature alignment (entry point 1)
              cannot recover the height loss; it lives in the decoder geometry
  ENC-MATTERS rescue CI low >= 50% of the loss
  otherwise partial. swap_all_bg (unmatched cells from c1 at the same position, unwarped) is a
  control, not a bound: it reads c1 content with the moved calibration, like T2's stale presets.
  Only drawn surfaces can be swapped (~10% of F0 cells on raster); background cells stay the new rig's.
  The focal presets are the contrast where the encoder itself changes (T2 surface diagnostic).

    python tools/summarize_t2_swap.py --tag full
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, load, token_logs  # noqa: E402

ORDER = ("ident", "h_p1.0", "h_m0.7", "f_0.8", "f_1.25", "d_p1.0", "d_m1.0", "l_p0.5", "l_m0.5")
HEIGHT = ("h_p1.0", "h_m0.7")
SWAPS = ("swap_f0", "swap_all", "swap_all_bg")
DEAD_MAX, MATTERS_MIN = 0.25, 0.50


def ci(d: pd.Series, logs, rng, n_boot):
    d = d.dropna()
    return None if d.empty else boot_ci(d, logs, n_boot, rng)


def pts(t) -> str:
    return "–" if t is None else f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


def share(r, loss) -> str:
    if r is None or loss is None or loss <= 1e-9:
        return "–"
    return f"{100 * r[0] / loss:.0f}% [{100 * r[1] / loss:.0f}, {100 * r[2] / loss:.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "t2_results")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    logs = token_logs(args.subset_csv)

    runs = {}
    pat = re.compile(rf"^t2swap_{re.escape(args.tag)}_(.+)$")
    for d in sorted(args.exp_root.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir() and (csv := latest_csv(d)) is not None:
            df = load(csv)
            runs[m.group(1)] = (csv, df[df["valid"].astype(bool)], int((~df["valid"].astype(bool)).sum()))
    if not runs:
        raise SystemExit(f"no t2swap_{args.tag}_* runs under {args.exp_root}")
    names = [p for p in ORDER if p in runs] + sorted(set(runs) - set(ORDER))

    rows1, rows2, rows3, verdict_in = [], [], [], {}
    for p in names:
        csv, ok, n_bad = runs[p]
        loss = float((ok["pdms_c1"] - ok["pdms_new"]).mean())
        d = {k: ci(ok[f"pdms_{k}"] - ok["pdms_c1"], logs, rng, args.n_boot) for k in ("new",) + SWAPS}
        r = {k: ci(ok[f"pdms_{k}"] - ok["pdms_new"], logs, rng, args.n_boot) for k in SWAPS}
        verdict_in[p] = (r["swap_all"], loss)
        rows1.append(f"| {p} | {len(ok)} | {n_bad} | {pts(d['new'])} | " + " | ".join(pts(d[k]) for k in SWAPS) + " |")
        rows2.append(f"| {p} | " + " | ".join(f"{pts(r[k])} / {share(r[k], loss)}" for k in SWAPS) + " |")
        dc = lambda a, b: pts(ci(ok[a] - ok[b], logs, rng, args.n_boot))  # noqa: E731
        rows3.append(
            f"| {p} | {dc('oracle_new', 'oracle_c1')} | {dc('oracle_swap_all_bg', 'oracle_c1')} | "
            f"{dc('mean64_new', 'mean64_c1')} | {dc('mean64_swap_all', 'mean64_c1')} | "
            f"{dc('mean64_swap_all_bg', 'mean64_c1')} | "
            f"{ok['swap_frac_s1'].mean():.3f} / {ok['swap_frac_s2'].mean():.3f} / {ok['swap_frac_s3'].mean():.3f} | "
            f"{ok['pade_new_c1'].mean():.2f} | {ok['pade_swap_all_c1'].mean():.2f} | {ok['pade_swap_all_bg_c1'].mean():.2f} |")

    have = [p for p in HEIGHT if p in verdict_in and verdict_in[p][0] is not None and verdict_in[p][1] > 1e-9]
    dead = len(have) == len(HEIGHT) and all(verdict_in[p][0][2] <= DEAD_MAX * verdict_in[p][1] for p in have)
    matters = len(have) == len(HEIGHT) and all(verdict_in[p][0][1] >= MATTERS_MIN * verdict_in[p][1] for p in have)
    verdict = "ENC-DEAD" if dead else "ENC-MATTERS" if matters else "partial"
    detail = ", ".join(f"{p}: swap_all rescue {pts(verdict_in[p][0])} of loss {100 * verdict_in[p][1]:.1f} "
                       f"({share(verdict_in[p][0], verdict_in[p][1])})" for p in have)

    lines = [
        f"# T2 feature-swap intervention — tag {args.tag}",
        "",
        "Official RAP v1 weights, raster, c1 = identity rig, updated calibration. Cells of the changed rig whose "
        "centre shows a surface also seen under c1 take the c1 feature of that surface (F0 only / F0+L0+R0); "
        "swap_all_bg (control, not a bound) also gives every other content cell the unwarped c1 cell at the same position. Decoder and "
        "calibration unchanged. ΔPDMS in points vs c1, 95% log-clustered bootstrap CI. Sources: "
        + ", ".join(f"`{runs[p][0]}`" for p in names),
        "",
        "## 1 ΔPDMS vs c1",
        "",
        "| preset | n | invalid | new | swap_f0 | swap_all | swap_all_bg |",
        "|---|---|---|---|---|---|---|",
        *rows1,
        "",
        "## 2 rescue = swap − new (points) / share of the loss",
        "",
        "| preset | swap_f0 | swap_all | swap_all_bg |",
        "|---|---|---|---|",
        *rows2,
        "",
        "## 3 candidate pool and swap coverage",
        "",
        "Δ oracle@64 / Δ mean64 vs c1; swapped cell fraction F0 / L0 / R0 (of 27 × 48 content cells); "
        "proposal ADE (m, slot-matched) new / swap_all / swap_all_bg vs c1.",
        "",
        "| preset | Δ oracle new | Δ oracle swap_all_bg | Δ mean64 new | Δ mean64 swap_all | Δ mean64 swap_all_bg | swap frac | ADE new | ADE swap_all | ADE swap_all_bg |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows3,
        "",
        "## 4 pre-registered reading",
        "",
        f"- ENC-DEAD: for both height presets the swap_all rescue CI high ≤ {int(100 * DEAD_MAX)}% of the loss",
        f"- ENC-MATTERS: for both height presets the swap_all rescue CI low ≥ {int(100 * MATTERS_MIN)}% of the loss",
        f"- **verdict: {verdict}** ({detail or 'height presets missing'})",
        "",
        "Raster only; the swap is at feature-cell resolution (one surface point per 16 px cell).",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"t2_swap_{args.tag}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}; verdict {verdict}")


if __name__ == "__main__":
    main()

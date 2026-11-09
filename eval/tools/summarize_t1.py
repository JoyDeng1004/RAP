"""Exp-0 / T1: paired ΔPDMS of every rig preset against ``raw`` on the same tokens.

CIs come from a bootstrap over logs (tokens of one log move together), as in the
E-series summaries. For each transform with a ``mask_`` control it also reports
the geometry-only part, Δ(transform) − Δ(mask control), with its own paired CI.

    python tools/summarize_t1.py --tag official
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/exp")
SUBSET = EXP / "metric_cache_v1_navtest_t1sub/metadata/metric_cache_v1_navtest_t1sub_metadata_node_0.csv"


def token_logs(csv: Path) -> pd.Series:
    rows = csv.read_text().splitlines()[1:]
    parts = [Path(r.split(",")[0]).parts for r in rows if r.strip()]
    return pd.Series({p[-2]: p[-4] for p in parts}, name="log")  # <cache>/<log>/<type>/<token>/metric_cache.pkl


def latest_csv(run_dir: Path):
    jobs = sorted((d for d in run_dir.iterdir() if d.is_dir() and list(d.glob("*.csv"))),
                  key=lambda d: int(d.name) if d.name.isdigit() else -1)
    return sorted(jobs[-1].glob("*.csv"))[-1] if jobs else None


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, index_col=0)
    return df[df["token"] != "average"].set_index("token")


def boot_ci(delta: pd.Series, logs: pd.Series, n_boot: int, rng) -> tuple:
    """Token-weighted mean of ``delta`` and a log-clustered bootstrap 95% CI."""
    g = pd.DataFrame({"d": delta, "log": logs.reindex(delta.index)}).dropna().groupby("log")["d"]
    sums, cnts = g.sum().to_numpy(), g.count().to_numpy()
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="official")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--metric", default="score")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "t1_results",
                    help="inside RAP/, so mutagen syncs the summary back")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    logs = token_logs(args.subset_csv)

    runs = {}
    pat = re.compile(rf"^t1_sub_{re.escape(args.tag)}_(.+)$")
    for d in sorted(args.exp_root.iterdir()):
        m = pat.match(d.name)
        if m and d.is_dir() and (csv := latest_csv(d)) is not None:
            runs[m.group(1)] = (csv, load(csv))
    if "raw" not in runs:
        raise SystemExit(f"no raw run for tag {args.tag} under {args.exp_root}")
    raw = runs["raw"][1]
    sub_cols = [c for c in raw.columns if c not in ("valid", args.metric) and pd.api.types.is_numeric_dtype(raw[c])]

    lines = [f"# T1 summary — tag {args.tag}, metric {args.metric}, raw mean {raw[args.metric].mean():.4f} "
             f"({len(raw)} tokens)", "",
             "| preset | n | invalid tok | Δ score | 95% CI | " + " | ".join(f"Δ {c}" for c in sub_cols) + " |",
             "|---|---|---|---|---|" + "---|" * len(sub_cols)]
    deltas = {}
    for name, (_, df) in sorted(runs.items()):
        if name == "raw":
            continue
        common = raw.index.intersection(df.index)
        d = df.loc[common, args.metric] - raw.loc[common, args.metric]
        deltas[name] = d
        m, lo, hi = boot_ci(d, logs, args.n_boot, rng)
        subs = " | ".join(f"{(df.loc[common, c] - raw.loc[common, c]).mean():+.4f}" for c in sub_cols)
        n_bad = int((~df.loc[common, "valid"].astype(bool)).sum())
        lines.append(f"| {name} | {len(common)} | {n_bad} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {subs} |")

    lines += ["", "## geometry-only: Δ(transform) − Δ(mask_transform)", "",
              "| preset | Δ transform | Δ mask | geometry-only | 95% CI |", "|---|---|---|---|---|"]
    for name, d in sorted(deltas.items()):
        if f"mask_{name}" not in deltas:
            continue
        dm = deltas[f"mask_{name}"]
        common = d.index.intersection(dm.index)
        m, lo, hi = boot_ci(d[common] - dm[common], logs, args.n_boot, rng)
        lines.append(f"| {name} | {d[common].mean():+.4f} | {dm[common].mean():+.4f} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")

    lines += ["", "## position-only: Δ(transform) − Δ(crop_transform)",
              "(crop_ keeps exactly the content that stays in view, at its original position)", "",
              "| preset | Δ transform | Δ crop | position-only | 95% CI |", "|---|---|---|---|---|"]
    for name, d in sorted(deltas.items()):
        if f"crop_{name}" not in deltas:
            continue
        dc = deltas[f"crop_{name}"]
        common = d.index.intersection(dc.index)
        m, lo, hi = boot_ci(d[common] - dc[common], logs, args.n_boot, rng)
        lines.append(f"| {name} | {d[common].mean():+.4f} | {dc[common].mean():+.4f} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")

    text = "\n".join(lines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"t1_summary_{args.tag}.md"
    out.write_text(text + "\n")
    print(text)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

"""Read-only cross-model summary of RAP NAVSIM results: v1 PDMS, v2 navtest EPDMS,
navhard two-stage EPDMS and v1 oracle@64, plus paired token-level comparisons.

Reads only result CSVs, oracle *_arrays.npz and (optionally) the v1 metric-cache
metadata csv (token -> log, for a log-clustered bootstrap). Imports nothing from
navsim. Writes nothing unless --out-md is given.

    python summarize_rap_eval.py --exp-root $NAVSIM_EXP_ROOT \
        --run E1:v1_scorer=eval_v1_navtest_E1_scorer/8812833 ... \
        --ref official --pair E2b:E2a --pair E3:E2a \
        --v1-cache $RAP/metric_cache_v1_navtest

--run MODEL:KIND=PATH, PATH relative to --exp-root (or absolute), a result dir
(newest *.csv inside) or a csv. KIND is one of KINDS below. Scores print x100.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KINDS = {
    "v1_scorer": "average",
    "v1_medoid": "average",
    "v2_scorer": "average_all_frames",
    "v2_medoid": "average_all_frames",
    "navhard_scorer": "extended_pdm_score_combined",
    "navhard_medoid": "extended_pdm_score_combined",
    "oracle64": "average",
}
AGG_ROWS = {"average", "average_all_frames", "extended_pdm_score_stage_one",
            "extended_pdm_score_stage_two", "extended_pdm_score_combined"}
V1_SUB = [("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"), ("EP", "ego_progress"),
          ("TTC", "time_to_collision_within_bound"), ("C", "comfort"), ("DDC", "driving_direction_compliance"),
          ("PDMS", "score")]
V2_SUB = [("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"),
          ("DDC", "driving_direction_compliance"), ("TLC", "traffic_light_compliance"), ("EP", "ego_progress"),
          ("TTC", "time_to_collision_within_bound"), ("LK", "lane_keeping"), ("HC", "history_comfort"),
          ("EC", "two_frame_extended_comfort")]
TOPK = (1, 4, 8, 16)
TIE = 1e-9


def resolve_csv(root: Path, spec: str) -> Path:
    p = Path(spec) if Path(spec).is_absolute() else root / spec
    if p.is_dir():
        csvs = [c for c in p.glob("*.csv")]
        if not csvs:
            raise SystemExit(f"no *.csv in {p}")
        p = max(csvs, key=lambda c: c.stat().st_mtime)
    if not p.is_file():
        raise SystemExit(f"missing result: {p}")
    return p


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["token"] = df["token"].astype(str)
    return df


def agg(df: pd.DataFrame, name: str) -> pd.Series:
    rows = df[df["token"] == name]
    if len(rows) != 1:
        raise SystemExit(f"expected one '{name}' row, found {len(rows)}")
    return pd.to_numeric(rows.iloc[0].drop(["token", "valid"], errors="ignore"), errors="coerce")


def token_rows(df: pd.DataFrame) -> pd.DataFrame:
    t = df[~df["token"].isin(AGG_ROWS)]
    if "valid" in t.columns:
        t = t[t["valid"].astype(str).str.lower().eq("true")]
    return t.set_index("token")


def pct(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def token_logs(cache: Optional[Path]) -> Optional[Dict[str, str]]:
    if cache is None:
        return None
    meta = next(p for p in sorted((cache / "metadata").iterdir()) if ".csv" in p.name)
    with open(meta) as f:
        rows = list(csv.reader(f))[1:]
    return {r[0].split("/")[-2]: r[0].split("/")[-4] for r in rows}


def bootstrap_ci(d: np.ndarray, groups: Optional[np.ndarray], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95% CI of mean(d); resamples whole logs when groups is given, else tokens."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        _, inv = np.unique(groups, return_inverse=True)
        sums, counts = np.bincount(inv, weights=d), np.bincount(inv).astype(np.float64)
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Spearman rho (average ranks for ties); NaN where a row of b is constant."""
    ra = pd.DataFrame(a).rank(axis=1).to_numpy()
    rb = pd.DataFrame(b).rank(axis=1).to_numpy()
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)


def oracle_stats(csv_path: Path) -> Dict[str, float]:
    df = load(csv_path)
    avg = agg(df, "average")
    npz_path = csv_path.with_name(csv_path.stem + "_arrays.npz")
    st = {k: float(avg[k]) for k in ("oracle64", "mean64", "score_scorer", "score_medoid",
                                    "minADE64", "ade_scorer", "ade_oracle") if k in avg}
    if not npz_path.is_file():
        print(f"warning: no {npz_path.name}; ranking stats skipped", file=sys.stderr)
        return st
    z = np.load(npz_path)
    s, logit = z["score64"], z["logit64"]
    pick = logit.argmax(axis=1)
    s_pick = s[np.arange(len(s)), pick]
    rows = token_rows(df)
    idx_csv = rows.loc[z["token"].astype(str), "idx_scorer"].to_numpy().astype(int)
    st["argmax_matches_csv"] = float((idx_csv == pick).mean())
    s_csv = rows.loc[z["token"].astype(str), "score_scorer"].to_numpy()
    st["npz_scorer_mean"] = float(s_pick.mean())
    st["npz_vs_csv_max_abs"] = float(np.abs(s[np.arange(len(s)), idx_csv] - s_csv).max())
    best = s.max(axis=1)
    st["pick_zero"] = float((s_pick == 0.0).mean())
    st["oracle_hit"] = float((s_pick >= best - TIE).mean())
    st["pick_pct"] = float(((s < s_pick[:, None] - TIE).mean(axis=1)
                            + 0.5 * (np.abs(s - s_pick[:, None]) <= TIE).mean(axis=1)).mean())
    order = np.argsort(-logit, axis=1)
    for k in TOPK:
        st[f"top{k}"] = float(np.take_along_axis(s, order[:, :k], axis=1).max(axis=1).mean())
    rho = row_spearman(logit, s)
    st["spearman"] = float(np.nanmean(rho))
    st["const_rows"] = float(np.isnan(rho).mean())
    st["safe_frac"] = float((s > 0.0).mean())
    st["n"] = float(len(s))
    return st


def capture(st: Dict[str, float]) -> float:
    o, m, sc = st["oracle64"], st["mean64"], st["score_scorer"]
    return (sc - m) / (o - m) if o > m else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], help="MODEL:KIND=PATH")
    ap.add_argument("--ref", default="official", help="model used as reference in the gap decomposition")
    ap.add_argument("--pair", action="append", default=[], help="A:B, paired token-level A minus B")
    ap.add_argument("--v1-cache", type=Path, help="v1 metric cache, for token -> log (clustered bootstrap)")
    ap.add_argument("--out-md", type=Path)
    args = ap.parse_args()

    runs: Dict[str, Dict[str, Path]] = {}
    for spec in args.run:
        model, rest = spec.split(":", 1)
        kind, path = rest.split("=", 1)
        if kind not in KINDS:
            raise SystemExit(f"unknown kind {kind}; one of {sorted(KINDS)}")
        runs.setdefault(model, {})[kind] = resolve_csv(args.exp_root, path)
    models = list(runs)
    dfs = {(m, k): load(p) for m in models for k, p in runs[m].items()}
    out: List[str] = []

    out.append("## Sources")
    out.append(md_table(["model", "kind", "csv"], [[m, k, str(p)] for m in models for k, p in runs[m].items()]))

    # 1. headline
    def score(m, k, name=None):
        return float(agg(dfs[m, k], name or KINDS[k])["score"]) if (m, k) in dfs else None

    rows = []
    for m in models:
        v1s, v1m, v2s, v2m = (score(m, k) for k in ("v1_scorer", "v1_medoid", "v2_scorer", "v2_medoid"))
        nh = ("—" if (m, "navhard_scorer") not in dfs else
              f"{pct(score(m, 'navhard_scorer'))} ({pct(score(m, 'navhard_scorer', 'extended_pdm_score_stage_one'))}"
              f" / {pct(score(m, 'navhard_scorer', 'extended_pdm_score_stage_two'))})")
        d1 = v1s - v1m if v1s is not None and v1m is not None else None
        d2 = v2s - v2m if v2s is not None and v2m is not None else None
        rows.append([m, pct(v1s), pct(v1m), pct(d1), pct(v2s), pct(v2m), pct(d2), nh, pct(score(m, "navhard_medoid"))])
    out.append("## 1. Headline (x100)")
    out.append(md_table(["model", "v1 scorer", "v1 medoid", "v1 Δ(s−m)", "v2 navtest scorer", "v2 navtest medoid",
                         "v2 Δ(s−m)", "navhard scorer combined (s1 / s2)", "navhard medoid"], rows))

    # 2-3. sub-metrics
    for title, kind, subs in (("## 2. v1 navtest sub-metrics, scorer (x100)", "v1_scorer", V1_SUB),
                              ("## 3. v2 navtest sub-metrics, scorer (x100)", "v2_scorer", V2_SUB + [("EPDMS", "score")])):
        rows = []
        for m in models:
            if (m, kind) in dfs:
                a = agg(dfs[m, kind], KINDS[kind])
                rows.append([m] + [pct(a.get(col, np.nan)) for _, col in subs])
        out.append(title)
        out.append(md_table(["model"] + [s for s, _ in subs], rows))

    # 4. navhard per stage
    rows = []
    for m in models:
        if (m, "navhard_scorer") not in dfs:
            continue
        for stage in ("one", "two"):
            a = agg(dfs[m, "navhard_scorer"], f"extended_pdm_score_stage_{stage}")
            rows.append([m, f"s{1 if stage == 'one' else 2}"]
                        + [pct(a.get(f"{col}_stage_{stage}", np.nan)) for _, col in V2_SUB] + [pct(a["score"])])
    out.append("## 4. navhard two-stage sub-metrics, scorer (x100)")
    out.append(md_table(["model", "stage"] + [s for s, _ in V2_SUB] + ["score"], rows))

    # 5. oracle@64
    ost = {m: oracle_stats(runs[m]["oracle64"]) for m in models if "oracle64" in runs[m]}
    rows = []
    for m, st in ost.items():
        rows.append([m, pct(st["oracle64"]), pct(st["mean64"]), pct(st["score_scorer"]), pct(st["score_medoid"]),
                     f"{capture(st):.2f}", pct(st["oracle64"] - st["score_scorer"]), pct(st.get("pick_zero")),
                     pct(st.get("oracle_hit")), pct(st.get("pick_pct")),
                     " / ".join(pct(st.get(f"top{k}")) for k in TOPK),
                     f"{st.get('spearman', np.nan):.3f}", pct(st.get("safe_frac")),
                     f"{st['minADE64']:.3f} / {st['ade_scorer']:.3f} / {st['ade_oracle']:.3f}"])
    out.append("## 5. v1 oracle@64 (x100; capture = (scorer−mean64)/(oracle64−mean64))")
    out.append(md_table(["model", "oracle64", "mean64", "scorer", "medoid", "capture", "regret", "pick=0",
                         "pick=oracle", "pick pctile", "best of scorer top-1/4/8/16", "spearman(logit,score)",
                         "safe cand.", "minADE64 / ADE scorer / ADE oracle (m)"], rows))
    checks = [f"{m}: argmax(logit64)==idx_scorer {st.get('argmax_matches_csv', float('nan')):.4f}, "
              f"npz vs csv max|Δ| {st.get('npz_vs_csv_max_abs', float('nan')):.2e}, "
              f"constant-score tokens {st.get('const_rows', float('nan')):.3f}, n={int(st.get('n', 0))}"
              for m, st in ost.items()]
    out.append("consistency: " + "; ".join(checks))

    # 6. gap decomposition vs ref
    if args.ref in ost:
        r = ost[args.ref]
        cr = capture(r)
        rows = []
        for m, st in ost.items():
            if m == args.ref:
                continue
            gap = r["score_scorer"] - st["score_scorer"]
            dc = cr - capture(st)
            sa, sb = dc * (st["oracle64"] - st["mean64"]), dc * (r["oracle64"] - r["mean64"])
            lo, hi = sorted((sa, sb))
            rows.append([m, pct(gap), f"{pct(lo)} … {pct(hi)}", f"{pct(gap - hi)} … {pct(gap - lo)}",
                         pct(r["oracle64"] - st["oracle64"]), pct(r["mean64"] - st["mean64"])])
        out.append(f"## 6. v1 gap to {args.ref}: scorer part = Δcapture × (oracle64−mean64) of either model; "
                   "candidate part = rest (x100)")
        out.append(md_table(["model", "PDMS gap", "scorer part", "candidate part", "oracle64 gap", "mean64 gap"], rows))

    # 7. paired token-level comparisons
    logs = token_logs(args.v1_cache)
    if logs is not None:
        out.append(f"token→log map: {len(logs)} tokens, {len(set(logs.values()))} logs (from {args.v1_cache})")
    for pair in args.pair:
        a, b = pair.split(":")
        rows = []
        for kind in ("v1_scorer", "v1_medoid", "v2_scorer", "v2_medoid", "oracle64"):
            if (a, kind) not in dfs or (b, kind) not in dfs:
                continue
            ta, tb = token_rows(dfs[a, kind]), token_rows(dfs[b, kind])
            cols = ["score"] if kind != "oracle64" else ["oracle64", "mean64", "score_scorer", "minADE64"]
            for col in cols:
                j = pd.concat([ta[col].rename("a"), tb[col].rename("b")], axis=1, join="inner").dropna()
                d = (j["a"] - j["b"]).to_numpy(dtype=np.float64)
                groups = np.array([logs.get(t, "?") for t in j.index]) if logs else None
                lo, hi = bootstrap_ci(d, groups)
                scale = 1.0 if col == "minADE64" else 100.0
                fmt = (lambda x: f"{scale * x:.3f}") if col == "minADE64" else (lambda x: f"{scale * x:.2f}")
                rows.append([f"{kind}.{col}", str(len(j)), fmt(j["a"].mean()), fmt(j["b"].mean()), fmt(d.mean()),
                             f"[{fmt(lo)}, {fmt(hi)}]", f"{100 * (d > TIE).mean():.1f}",
                             f"{100 * (np.abs(d) <= TIE).mean():.1f}", f"{100 * (d < -TIE).mean():.1f}"])
        out.append(f"## 7. paired {a} − {b} (per-token mean; 95% CI by "
                   f"{'log-clustered' if logs else 'token'} bootstrap; minADE in m, others x100)")
        out.append(md_table(["metric", "n", a, b, "Δ", "95% CI", "Δ>0 %", "tie %", "Δ<0 %"], rows))

    text = "\n\n".join(out) + "\n"
    print(text)
    if args.out_md:
        if args.out_md.exists():
            raise SystemExit(f"refusing to overwrite {args.out_md}")
        args.out_md.write_text(text)


if __name__ == "__main__":
    main()

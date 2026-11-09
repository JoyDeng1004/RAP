"""K0 real-vs-raster diagnosis, step 1: summarize the ident-preset probe runs (submit_k0_probe.sh) into
eval/k0_results/k0_probe_<tag>.md (+ .csv).

Per model, on the tokens valid in every listed run (paired), real = NAVSIM real images, c1 = the
official-convention raster of the same token under the same calibration:
  1  PDMS of the scorer pick for real and c1, Δ c1 − real (log-clustered bootstrap 95% CI,
     summarize_t1.boot_ci); oracle@64, mean64 and capture = (pick − mean64) / (oracle − mean64)
     on the means, for both inputs
  2  selection on each input's own 64 candidates: Spearman ρ between scorer score and candidate PDMS
     (token mean, tokens with a constant PDMS pool skipped), hit@1 = share of tokens whose pick is
     an oracle candidate
  3  how far the decoder state moves between the two inputs: query cosine after each refiner round,
     proposal ADE of the last round, pool coverage, F0 feature cosine at the round-1 3D anchors
  4  tail of the pick, to tell "a few catastrophic picks" from "every pick a bit worse":
     a  share of tokens whose pick scores 0 (a multiplicative PDMS gate failed) or < 0.5, the paired
        real − c1 zero-rate difference with a log-clustered CI, and the mean pick on non-oracle tokens
     b  regret = oracle − pick: median / p90 / p99 / mean, and the share of the total regret carried by
        the worst 10% of tokens
     c  real-only failures (real pick 0, c1 pick ≥ 0.5) and the reverse; the share of the summed
        c1 − real gap they carry; among real zero picks, how often the pool held a perfect candidate
        (pure selection error) and how much of the pool scores 0 there vs on all tokens; the scorer's
        top-1 − top-2 score margin on oracle hits vs on zero picks (is it confidently wrong?)
     d  (runs with sub-metrics, run_t2_probe_v1 from 10/29 on) which gate a zero pick failed: at-fault
        collision (no_collision = 0), leaving the drivable area (drivable_area = 0) or both, for all
        zero picks and for the real-only failures; how often the pool of a real zero pick fails each gate;
        mean TTC / EP / comfort of the pick over all tokens
     The real-only failures are also written per token to k0_probe_<tag>_tail_tokens.csv (worst
     regret first), as the token list for the feature visualizations.
PDMS-like numbers are in points (×100). The subset is T2's 1,500-token navtest stratified sample, so
absolute PDMS differs from the full navtest evals.

    python tools/summarize_k0_probe.py --tag k0a
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, load, token_logs  # noqa: E402

MODELS = ("D2-K0-scene-mix_e6", "D2-K0-scene-mix_last", "D2-K1-mix_e13", "D2-K10-mix",
          "D1-K1_e16", "D1-K10", "D1-K100_e16", "official")
INPUTS = ("real", "c1")
N_ROUNDS = 4
ZERO = 1e-6  # PDMS at or below this counts as a zero pick
LOW = 0.5
PERFECT = 0.99
GATES = ("no_collision", "drivable_area")  # the multiplicative v1 sub-metrics: 0 on either zeroes PDMS
SOFT = ("ttc", "progress", "comfortable")


def run_dir(exp_root: Path, tag: str, model: str, official_dir: Path) -> Path:
    d = exp_root / f"k0probe_{tag}_{model}"
    return official_dir if model == "official" and not d.is_dir() else d


def spearman_rows(score: np.ndarray, pdms: np.ndarray) -> np.ndarray:
    """Per-row Spearman ρ of (n, 64) arrays with average ranks; NaN where either row is constant."""
    rs = pd.DataFrame(score).rank(axis=1).to_numpy()
    rp = pd.DataFrame(pdms).rank(axis=1).to_numpy()
    rs = rs - rs.mean(1, keepdims=True)
    rp = rp - rp.mean(1, keepdims=True)
    den = np.sqrt((rs ** 2).sum(1) * (rp ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rs * rp).sum(1) / den, np.nan)


def margin_rows(score: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 scorer score per row."""
    top2 = np.sort(score, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def load_run(d: Path):
    """Valid rows of the latest run in ``d`` and per-token stats from its candidate arrays
    (rho_/margin_/pool0_<input>: Spearman ρ, scorer top-1 − top-2 margin, share of the pool scoring 0)."""
    csv = latest_csv(d) if d.is_dir() else None
    if csv is None:
        return None, None, None
    df = load(csv)
    df = df[df["valid"].astype(bool)]
    npz = csv.with_name(f"{csv.stem}_arrays.npz")
    per_tok = {}
    if npz.exists():
        a = np.load(npz)
        tok = pd.Index(a["token"].astype(str))
        for k in INPUTS:
            if f"score_{k}" in a and f"s64_{k}" in a:
                sc, s64 = a[f"score_{k}"], a[f"s64_{k}"]
                pick = pd.Series(s64[np.arange(len(sc)), sc.argmax(1)], index=tok).reindex(df.index)
                bad = int((~np.isclose(pick, df[f"pdms_{k}"], atol=1e-6)).sum())
                if bad:
                    print(f"WARNING {csv}: {bad} tokens whose npz pick ≠ CSV pdms_{k}")
                per_tok[f"rho_{k}"] = pd.Series(spearman_rows(sc, s64), index=tok)
                per_tok[f"margin_{k}"] = pd.Series(margin_rows(sc), index=tok)
                per_tok[f"pool0_{k}"] = pd.Series((s64 <= ZERO).mean(1), index=tok)
                for g in GATES:
                    if f"sub_{g}_{k}" in a:
                        per_tok[f"poolfail_{g}_{k}"] = pd.Series((a[f"sub_{g}_{k}"] <= ZERO).mean(1), index=tok)
    return df, pd.DataFrame(per_tok), csv


def tail_stats(df: pd.DataFrame, tok: pd.DataFrame, logs: pd.Series, n_boot: int, rng) -> dict:
    """Section 4 numbers of one model on the common tokens (PDMS-like values ×100, rates in %)."""
    r = {}
    pick = {k: df[f"pdms_{k}"] for k in INPUTS}
    zero = {k: pick[k] <= ZERO for k in INPUTS}
    for k in INPUTS:
        regret = 100 * (df[f"oracle_{k}"] - pick[k])
        hit = pick[k] >= df[f"oracle_{k}"] - 1e-9
        r[f"zero_{k}"] = 100 * zero[k].mean()
        r[f"low_{k}"] = 100 * (pick[k] < LOW).mean()
        r[f"miss_mean_{k}"] = 100 * pick[k][~hit].mean() if (~hit).any() else float("nan")
        for q, name in ((50, "p50"), (90, "p90"), (99, "p99")):
            r[f"regret_{name}_{k}"] = float(np.percentile(regret, q))
        r[f"regret_mean_{k}"] = regret.mean()
        worst = np.sort(regret.to_numpy())[::-1][:max(1, int(np.ceil(0.1 * len(regret))))]
        r[f"worst10_share_{k}"] = 100 * worst.sum() / regret.sum() if regret.sum() > 0 else float("nan")
        m = tok.get(f"margin_{k}")
        if m is not None:
            m = m.reindex(df.index)
            r[f"margin_hit_{k}"] = float(m[hit].median()) if hit.any() else float("nan")
            r[f"margin_zero_{k}"] = float(m[zero[k]].median()) if zero[k].any() else float("nan")
    d, lo, hi = boot_ci(100 * (zero["real"].astype(float) - zero["c1"].astype(float)), logs, n_boot, rng)
    r.update(d_zero=d, d_zero_lo=lo, d_zero_hi=hi)

    real_only = zero["real"] & (pick["c1"] >= LOW)
    c1_only = zero["c1"] & (pick["real"] >= LOW)
    gap = pick["c1"] - pick["real"]
    r["n_real_only"], r["n_c1_only"] = int(real_only.sum()), int(c1_only.sum())
    r["gap_share_real_only"] = 100 * gap[real_only].sum() / gap.sum() if gap.sum() > 0 else float("nan")
    zr = zero["real"]
    r["perfect_in_pool"] = 100 * (df.loc[zr, "oracle_real"] >= PERFECT).mean() if zr.any() else float("nan")
    if "pool0_real" in tok:
        p0 = tok["pool0_real"].reindex(df.index)
        r["pool0_on_zero"] = 100 * p0[zr].mean() if zr.any() else float("nan")
        r["pool0_all"] = 100 * p0.mean()
    else:
        r["pool0_on_zero"] = r["pool0_all"] = float("nan")
    r["_real_only_tokens"] = real_only[real_only].index

    # 4d: which gate failed (needs pick_<gate>_<input> columns)
    for k in INPUTS:
        have = all(f"pick_{g}_{k}" in df for g in GATES)
        for sel, name in ((zero[k], "zero"), (real_only if k == "real" else c1_only, "only")):
            for g in ("coll", "offroad", "both"):
                r[f"gate_{name}_{g}_{k}"] = float("nan")
            if have and sel.any():
                coll = df.loc[sel, "pick_no_collision_" + k] <= ZERO
                off = df.loc[sel, "pick_drivable_area_" + k] <= ZERO
                r[f"gate_{name}_coll_{k}"] = 100 * (coll & ~off).mean()
                r[f"gate_{name}_offroad_{k}"] = 100 * (off & ~coll).mean()
                r[f"gate_{name}_both_{k}"] = 100 * (coll & off).mean()
        for g in SOFT:
            r[f"pick_{g}_{k}"] = 100 * df[f"pick_{g}_{k}"].mean() if f"pick_{g}_{k}" in df else float("nan")
        for g in GATES:
            c = f"poolfail_{g}_{k}"
            r[f"poolfail_{g}_{k}"] = 100 * tok[c].reindex(df.index)[zero[k]].mean() if c in tok and zero[k].any() \
                else float("nan")
    return r


def capture(df: pd.DataFrame, k: str) -> float:
    p, o, m = df[f"pdms_{k}"].mean(), df[f"oracle_{k}"].mean(), df[f"mean64_{k}"].mean()
    return (p - m) / (o - m) if o > m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--official-dir", type=Path, default=EXP / "t2probe_full_ident",
                    help="ident run of the official weights (T2 full)")
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "k0_results")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    runs, missing = {}, []
    for m in args.models:
        df, rho, csv = load_run(run_dir(args.exp_root, args.tag, m, args.official_dir))
        if df is None or not len(df):
            missing.append(m)
            continue
        runs[m] = (df, rho, csv)
    if not runs:
        raise SystemExit(f"no runs found for tag={args.tag} under {args.exp_root}")
    common = sorted(set.intersection(*(set(df.index) for df, _, _ in runs.values())))
    logs = token_logs(args.subset_csv)

    rows, tail_rows = [], []
    for m, (df, rho, csv) in runs.items():
        df = df.loc[common]
        r = {"model": m, "n": len(df), "csv": str(csv)}
        for k in INPUTS:
            r[f"pdms_{k}"] = 100 * df[f"pdms_{k}"].mean()
            r[f"oracle_{k}"] = 100 * df[f"oracle_{k}"].mean()
            r[f"mean64_{k}"] = 100 * df[f"mean64_{k}"].mean()
            r[f"capture_{k}"] = capture(df, k)
            r[f"hit1_{k}"] = float((df[f"pdms_{k}"] >= df[f"oracle_{k}"] - 1e-9).mean())
            r[f"rho_{k}"] = float(rho[f"rho_{k}"].reindex(common).mean()) if f"rho_{k}" in rho else float("nan")
        d, lo, hi = boot_ci(100 * (df["pdms_c1"] - df["pdms_real"]), logs, args.n_boot, rng)
        r.update(d_pdms=d, d_lo=lo, d_hi=hi)
        for i in range(1, N_ROUNDS + 1):
            r[f"qcos_r{i}"] = df[f"qcos_r{i}_c1_real"].mean() if f"qcos_r{i}_c1_real" in df else float("nan")
        for c in (f"pade_r{N_ROUNDS}_c1_real", "cover_c1_real", "anchor_cos_c1_real"):
            r[c] = df[c].mean() if c in df else float("nan")
        r.update(tail_stats(df, rho, logs, args.n_boot, rng))
        for t in r.pop("_real_only_tokens"):
            tail_rows.append({"model": m, "token": t, "log": logs.get(t),
                              "pdms_real": df.at[t, "pdms_real"], "pdms_c1": df.at[t, "pdms_c1"],
                              "oracle_real": df.at[t, "oracle_real"],
                              "regret_real": df.at[t, "oracle_real"] - df.at[t, "pdms_real"],
                              "margin_real": rho["margin_real"].get(t, np.nan) if "margin_real" in rho else np.nan,
                              "pool0_real": rho["pool0_real"].get(t, np.nan) if "pool0_real" in rho else np.nan,
                              **{f"pick_{g}_{k}": df.at[t, f"pick_{g}_{k}"] if f"pick_{g}_{k}" in df else np.nan
                                 for g in GATES for k in INPUTS}})
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")

    f1, f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:+.2f}")
    out = [f"# K0 real vs raster probe, tag {args.tag}", "",
           f"{len(common)} tokens valid in every run (T2 navtest subset). real = NAVSIM real images, "
           "c1 = official-convention raster, same calibration. PDMS-like columns ×100.", ""]
    if missing:
        out += [f"Missing runs (skipped): {', '.join(missing)}", ""]
    out += ["## 1 Scores", "",
            "| model | PDMS real | PDMS c1 | Δ c1−real [95% CI] | oracle real / c1 | mean64 real / c1 | capture real / c1 |",
            "|---|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        out.append(f"| {m} | {f1(r.pdms_real)} | {f1(r.pdms_c1)} | {f3(r.d_pdms)} [{f3(r.d_lo)}, {f3(r.d_hi)}] | "
                   f"{f1(r.oracle_real)} / {f1(r.oracle_c1)} | {f1(r.mean64_real)} / {f1(r.mean64_c1)} | "
                   f"{f2(r.capture_real)} / {f2(r.capture_c1)} |")
    out += ["", "## 2 Selection on each input's own candidates", "",
            "| model | Spearman ρ real | Spearman ρ c1 | hit@1 real | hit@1 c1 |", "|---|---|---|---|---|"]
    for m, r in t.iterrows():
        out.append(f"| {m} | {f2(r.rho_real)} | {f2(r.rho_c1)} | {f2(r.hit1_real)} | {f2(r.hit1_c1)} |")
    out += ["", "## 3 Decoder state, c1 vs real", "",
            "| model | qcos r1 | r2 | r3 | r4 | proposal ADE r4 (m) | pool cover (m) | F0 anchor feature cos |",
            "|---|---|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        q = " | ".join(f2(r[f"qcos_r{i}"]) for i in range(1, N_ROUNDS + 1))
        out.append(f"| {m} | {q} | {f2(r[f'pade_r{N_ROUNDS}_c1_real'])} | {f2(r.cover_c1_real)} | "
                   f"{f2(r.anchor_cos_c1_real)} |")
    pct = lambda v: f"{v:.1f}"
    out += ["", "## 4 Tail of the pick", "",
            f"zero pick = pick PDMS ≤ {ZERO:g} (a collision / drivable-area / … gate failed); regret = oracle − pick.",
            "", "### 4a How often the pick fails", "",
            "| model | pick = 0 (%) real / c1 | Δ zero real − c1 (pp) [95% CI] | pick < 0.5 (%) real / c1 | "
            "mean pick on non-oracle tokens real / c1 |", "|---|---|---|---|---|"]
    for m, r in t.iterrows():
        out.append(f"| {m} | {pct(r.zero_real)} / {pct(r.zero_c1)} | {f3(r.d_zero)} [{f3(r.d_zero_lo)}, "
                   f"{f3(r.d_zero_hi)}] | {pct(r.low_real)} / {pct(r.low_c1)} | "
                   f"{f1(r.miss_mean_real)} / {f1(r.miss_mean_c1)} |")
    out += ["", "### 4b Regret distribution (points)", "",
            "| model | median real / c1 | p90 real / c1 | p99 real / c1 | mean real / c1 | "
            "worst 10% share of regret (%) real / c1 |", "|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        cells = [f"{f1(r[f'regret_{q}_real'])} / {f1(r[f'regret_{q}_c1'])}" for q in ("p50", "p90", "p99", "mean")]
        out.append(f"| {m} | {' | '.join(cells)} | {pct(r.worst10_share_real)} / {pct(r.worst10_share_c1)} |")
    out += ["", "### 4c Failures that only the real input makes", "",
            f"real-only = real pick 0 and c1 pick ≥ {LOW}; c1-only the reverse. Gap share = part of Σ(c1 − real) "
            f"on those tokens (— when the gap is not positive). Perfect in pool = real zero picks whose pool held "
            f"a candidate ≥ {PERFECT}. Margin = scorer top-1 − top-2 score (median).", "",
            "| model | real-only / c1-only (tokens) | gap share of real-only (%) | perfect in pool (%) | "
            "pool share scoring 0 (%) on real zero picks / all | margin real: oracle hit / zero pick | "
            "margin c1: oracle hit / zero pick |", "|---|---|---|---|---|---|---|"]
    for m, r in t.iterrows():
        gs = "—" if np.isnan(r.gap_share_real_only) else pct(r.gap_share_real_only)
        out.append(f"| {m} | {int(r.n_real_only)} / {int(r.n_c1_only)} | {gs} | {pct(r.perfect_in_pool)} | "
                   f"{pct(r.pool0_on_zero)} / {pct(r.pool0_all)} | {f2(r.margin_hit_real)} / {f2(r.margin_zero_real)} | "
                   f"{f2(r.margin_hit_c1)} / {f2(r.margin_zero_c1)} |")
    out += ["", "### 4d Which gate the zero picks fail", "",
            "Collision = at-fault collision only (no_collision = 0), off-road = drivable_area = 0 only, both = "
            "both; shares of the zero picks (all of them, or the input-only failures of 4c), in %. Pool fails = "
            "share of the 64 candidates failing each gate on that input's zero-pick tokens. Pick TTC / EP / comfort "
            "= mean over all tokens ×100. — = no such picks, or a run without sub-metrics.", "",
            "| model | input | zero picks: collision / off-road / both | input-only failures: collision / off-road / "
            "both | pool fails collision / off-road | pick TTC / EP / comfort |", "|---|---|---|---|---|---|"]
    dash = lambda v: "—" if np.isnan(v) else pct(v)
    for m, r in t.iterrows():
        for k in INPUTS:
            z = " / ".join(dash(r[f"gate_zero_{g}_{k}"]) for g in ("coll", "offroad", "both"))
            o = " / ".join(dash(r[f"gate_only_{g}_{k}"]) for g in ("coll", "offroad", "both"))
            pf = " / ".join(dash(r[f"poolfail_{g}_{k}"]) for g in GATES)
            sp = " / ".join(dash(r[f"pick_{g}_{k}"]) for g in SOFT)
            out.append(f"| {m} | {k} | {z} | {o} | {pf} | {sp} |")
    out += ["", f"Real-only failure tokens: `k0_probe_{args.tag}_tail_tokens.csv`"]
    out += ["", "Sources:", ""] + [f"- {m}: `{r.csv}`" for m, r in t.iterrows()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = args.out_dir / f"k0_probe_{args.tag}.md"
    md.write_text("\n".join(out) + "\n")
    t.to_csv(args.out_dir / f"k0_probe_{args.tag}.csv")
    cols = ["model", "token", "log", "pdms_real", "pdms_c1", "oracle_real", "regret_real", "margin_real", "pool0_real",
            *(f"pick_{g}_{k}" for g in GATES for k in INPUTS)]
    (pd.DataFrame(tail_rows, columns=cols).sort_values(["model", "regret_real"], ascending=[True, False])
     .to_csv(args.out_dir / f"k0_probe_{args.tag}_tail_tokens.csv", index=False))
    print("\n".join(out))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()

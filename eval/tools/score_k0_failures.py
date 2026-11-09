"""K0 real-vs-raster diagnosis, A: does the scorer tell dangerous candidates from safe ones under the real input?
CPU only: everything comes from the probe run's npz (score_*, s64_*, sub_*, cand_* of the real and c1 inputs).

Tokens: viz_k0_failures.pick_tokens (same seed, so the V1 / V3 groups): collision, offroad and their
difficulty-matched controls hard_coll, hard_off.

Per token and input (each on its own 64 candidates; collision = no_collision 0, off-road = drivable_area 0 without
collision, safe = neither gate failed):
  d_coll / d_off      mean scorer score of collision / off-road candidates − mean score of safe candidates
  d_coll_z / d_off_z  the same divided by the std of the 64 scores (scores differ in scale between tokens)
  safe_pct            mean rank percentile of the safe candidates (1 = top score, 0.5 = chance)
  auc_coll / auc_off  P(score of a safe candidate > score of a collision / off-road one), ties 1/2
Summary: group means per input; real vs c1 within a group (paired sign-flip permutation on tokens with both),
failure vs matched control per input (two-sample permutation).

Figures (webp): per group the first --n-plot tokens, BEV of the real (left) and c1 (right) candidates coloured by
score rank, collision dashed, off-road dotted, safe solid, each input's pick bold with a star.

Writes eval/k0_results/scores/<model>/summary.md, tokens.csv and the figures.

    python tools/score_k0_failures.py --scenario-dir <stage A dir> --tag k0b --model D2-K0-scene-mix_last
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_t1 import EXP, SUBSET, latest_csv, load, token_logs  # noqa: E402
from viz_k0_failures import ZERO, draw_scene, perm_test, pick_tokens, pool_shares, save  # noqa: E402

GROUPS = ("collision", "offroad", "hard_coll", "hard_off")
INPUTS = ("real", "c1")
METRICS = ("d_coll", "d_off", "d_coll_z", "d_off_z", "safe_pct", "auc_coll", "auc_off")
MATCHED = (("collision", "hard_coll"), ("offroad", "hard_off"))
N_PERM = 20000


def rank_pct(score: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1]; 1 = highest score."""
    return pd.Series(score).rank(method="average").to_numpy() / len(score)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(pos == neg); NaN if either is empty."""
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def token_stats(score: np.ndarray, nc: np.ndarray, da: np.ndarray) -> dict:
    coll = nc <= ZERO
    off = (da <= ZERO) & ~coll
    safe = ~coll & ~(da <= ZERO)
    sd = score.std()
    pct = rank_pct(score)

    def diff(m):
        return float(score[m].mean() - score[safe].mean()) if m.any() and safe.any() else float("nan")

    out = {"d_coll": diff(coll), "d_off": diff(off), "safe_pct": float(pct[safe].mean()) if safe.any() else float("nan"),
           "auc_coll": auc(score[safe], score[coll]), "auc_off": auc(score[safe], score[off]),
           "n_coll": int(coll.sum()), "n_off": int(off.sum()), "n_safe": int(safe.sum())}
    out["d_coll_z"] = out["d_coll"] / sd if sd > 0 else float("nan")
    out["d_off_z"] = out["d_off"] / sd if sd > 0 else float("nan")
    return out


def sign_flip(d: np.ndarray, rng, n: int = N_PERM) -> tuple:
    """Mean of paired differences and its two-sided sign-flip permutation p."""
    d = d[~np.isnan(d)]
    if not len(d):
        return float("nan"), float("nan"), 0
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (flips * d).mean(1)
    return float(obs), float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)), len(d)


def plot_pair(scenario: dict, arrs: dict, token: str, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    cmap = plt.get_cmap("viridis")
    for ax, k in zip(axes, INPUTS):
        plot = draw_scene(ax, scenario)
        sc, cand = arrs[f"score_{k}"], arrs[f"cand_{k}"]
        nc, da = arrs[f"sub_no_collision_{k}"], arrs[f"sub_drivable_area_{k}"]
        pct = rank_pct(sc)
        for j in np.argsort(pct):  # best on top
            ls = "--" if nc[j] <= ZERO else (":" if da[j] <= ZERO else "-")
            plot(np.vstack([[0, 0], cand[j, :, :2]]), color=cmap(pct[j]), lw=1.0, ls=ls, alpha=0.9, zorder=4)
        i = int(sc.argmax())
        xy = np.vstack([[0, 0], cand[i, :, :2]])
        plot(xy, color="k", lw=4.0, zorder=5)
        plot(xy, color=cmap(1.0), lw=2.0, zorder=6, ls="--" if nc[i] <= ZERO else (":" if da[i] <= ZERO else "-"))
        ax.plot(-xy[-1, 1], xy[-1, 0], marker="*", ms=14, color="red", mec="k", zorder=7)
        st = token_stats(sc, nc, da)
        ax.set_title(f"{k}: pick PDMS {arrs[f's64_{k}'][i]:.2f} | AUC safe>coll {st['auc_coll']:.2f}, "
                     f"safe>off {st['auc_off']:.2f} | safe rank pct {st['safe_pct']:.2f}", fontsize=8)
    handles = [Line2D([], [], color="0.3", ls="-", label="safe"), Line2D([], [], color="0.3", ls="--", label="collision"),
               Line2D([], [], color="0.3", ls=":", label="off-road"),
               Line2D([], [], color="k", lw=3, marker="*", mfc="red", label="pick")]
    axes[1].legend(handles=handles, loc="lower right", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="scorer score rank in pool (1 = top)")
    fig.suptitle(title, fontsize=10)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-plot", type=int, default=4)
    ap.add_argument("--control-min", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or HERE.parent / "k0_results" / "scores" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_csv = latest_csv(args.exp_root / f"k0probe_{args.tag}_{args.model}")
    if probe_csv is None:
        raise SystemExit(f"no probe run k0probe_{args.tag}_{args.model} under {args.exp_root}")
    probe = load(probe_csv)
    probe = probe[probe["valid"].astype(bool)]
    logs = token_logs(args.subset_csv)
    groups = pick_tokens(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv", probe,
                         pool_shares(probe_csv), logs, args.model, args.n, args.n, args.control_min, args.seed)
    groups = {g: groups[g] for g in GROUPS}
    print({g: len(v) for g, v in groups.items()}, "probe:", probe_csv, flush=True)

    npz = np.load(probe_csv.with_name(f"{probe_csv.stem}_arrays.npz"))
    pos = {t: i for i, t in enumerate(npz["token"].astype(str))}
    keys = [f"{a}_{k}" for k in INPUTS for a in ("score", "s64", "cand", "sub_no_collision", "sub_drivable_area")]
    missing = [k for k in keys if k not in npz]
    if missing:
        raise SystemExit(f"npz lacks {missing}; rerun the probe (run_t2_probe_v1 from 10/29 on)")
    arrays = {k: npz[k] for k in keys}

    rows, n_bad = [], 0
    for g, toks in groups.items():
        for t in toks:
            a = {k: v[pos[t]] for k, v in arrays.items()}
            row = {"group": g, "token": t, "log": logs[t]}
            for k in INPUTS:
                i = int(a[f"score_{k}"].argmax())
                if i != int(probe.at[t, f"idx_{k}"]) or abs(a[f"s64_{k}"][i] - probe.at[t, f"pdms_{k}"]) > 1e-6:
                    n_bad += 1
                    print(f"WARNING {t} {k}: npz pick {i} pdms {a[f's64_{k}'][i]:.4f} vs CSV "
                          f"{int(probe.at[t, f'idx_{k}'])} {probe.at[t, f'pdms_{k}']:.4f}", flush=True)
                row[f"pdms_{k}"] = float(a[f"s64_{k}"][i])
                row.update({f"{m}_{k}": v for m, v in token_stats(a[f"score_{k}"], a[f"sub_no_collision_{k}"],
                                                                     a[f"sub_drivable_area_{k}"]).items()})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tokens.csv", index=False)

    rng = np.random.default_rng(args.seed)
    lines = [f"# A: scorer scores of dangerous vs safe candidates, {args.model} (tag {args.tag})", "",
             f"Tokens per group: {df['group'].value_counts().to_dict()}; probe `{probe_csv}`; npz/CSV pick "
             f"mismatches: {n_bad}.",
             "Each input on its own 64 candidates. d_* = mean score of collision / off-road candidates − safe ones "
             "(raw and / std of the 64 scores); safe_pct = mean rank percentile of safe candidates (1 top, 0.5 "
             "chance); auc_* = P(safe scores above collision / off-road). Mean over tokens with such candidates "
             "(n in brackets).", "",
             "## Group means", "", "| group | input | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for g in GROUPS:
        for k in INPUTS:
            sub = df[df["group"] == g]
            cells = [f"{sub[f'{m}_{k}'].mean():.3f} ({sub[f'{m}_{k}'].notna().sum()})" for m in METRICS]
            lines.append(f"| {g} | {k} | " + " | ".join(cells) + " |")
    lines += ["", "## Real vs c1 within a group (paired, real − c1, sign-flip p)", "",
              "| group | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for g in GROUPS:
        sub = df[df["group"] == g]
        cells = []
        for m in METRICS:
            d, p, n = sign_flip((sub[f"{m}_real"] - sub[f"{m}_c1"]).to_numpy(float), rng)
            cells.append(f"{d:+.3f} (p {p:.3f}, n {n})")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Failure vs matched control (failure − control, permutation p)", "",
              "| pair | input | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for f, c in MATCHED:
        for k in INPUTS:
            cells = []
            for m in METRICS:
                d, p = perm_test(df.loc[df["group"] == f, f"{m}_{k}"].to_numpy(float),
                                 df.loc[df["group"] == c, f"{m}_{k}"].to_numpy(float), rng)
                cells.append(f"{d:+.3f} (p {p:.3f})")
            lines.append(f"| {f} vs {c} | {k} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    need = {logs[t] for g in GROUPS for t in groups[g][: args.n_plot]}
    scenarios = {}
    for log in sorted(need):
        with open(args.scenario_dir / f"{log}.pkl", "rb") as f:
            scenarios.update(pickle.load(f))
    for g in GROUPS:
        for rank, t in enumerate(groups[g][: args.n_plot]):
            a = {k: v[pos[t]] for k, v in arrays.items()}
            r = df[(df["group"] == g) & (df["token"] == t)].iloc[0]
            plot_pair(scenarios[t], a, t, f"{args.model} | {g} #{rank} | {t} | {logs[t]} | PDMS real "
                      f"{r.pdms_real:.2f} vs c1 {r.pdms_c1:.2f}", out_dir / f"{g}_{rank}_{t}.webp")


if __name__ == "__main__":
    main()

"""Figures 1-7 for the 2026-10-23 meeting (D-K shot curve week).

Numbers are copied from eval/summary_DK_*.md (navtest, x100, log-clustered bootstrap over 136 logs);
training-val curves are parsed from qsub_outputs/ with parse_val_curves.py; figs 6-7 copy
eval/t2_results/t2_summary_full.md and t2_swap_full.md.

    python scripts/analysis/plot_mtg1023_figs.py [--out docs/slides/assets]

Writes mtg1023_fig{1..7}_*.png.
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_val_curves import parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# palette (dataviz reference instance, light mode); colour follows the data family in every figure
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
D1 = "#2a78d6"      # NAVSIM only
D2 = "#eb6834"      # NAVSIM + nuScenes, -mix
TGT = "#1baf7a"     # NAVSIM + nuScenes, -tgt
REF = "#52514e"     # official RAP

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})

FOOT_COMMON = ("navtest, PDMS x100. Single seed except D1-K1 (3 seeds). Every checkpoint was selected on "
               "training-val = 214 NAVSIM val logs with real images. -mix runs carry the SCA sample-0 bev_mask defect.")


def footnote(fig, text, y=0.01):
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=9.5, color=INK2, wrap=True)


# ---------------------------------------------------------------- fig 1: shot curve P(K)
XK = {0: -0.75, 1: 0.0, 10: 1.0, 100: 2.0}

SHOT = {
    "v1": dict(d1_seeds=[80.36, 78.12, 78.76], d1_mean=79.08, d1_sd=1.15, d1_k10=82.83, d1_k100=92.30,
               d2_k0e6=82.30, d2_k0last=77.66, d2_k1=85.73, d2_k10=89.40, tgt_k10=88.66, official=93.80,
               ylim=(74, 96), title="NAVSIM v1 PDMS", k1_label=(0.1, 80.7, "bottom"), k100_below=False),
    "v2": dict(d1_seeds=[73.70, 70.90, 72.90], d1_mean=72.50, d1_sd=1.44, d1_k10=76.87, d1_k100=86.16,
               d2_k0e6=77.20, d2_k0last=74.02, d2_k1=80.68, d2_k10=83.28, tgt_k10=82.66, official=86.67,
               ylim=(68, 90), title="NAVSIM v2 EPDMS", k1_label=(0.1, 71.4, "top"), k100_below=True),
    # navhard two-stage EPDMS (v2 protocol, navhard split); D1-K1 s.d. over 3 seeds
    "navhard": dict(d1_seeds=[28.05, 33.58, 28.09], d1_mean=29.91, d1_sd=3.18, d1_k10=36.12, d1_k100=37.38,
                    d2_k0e6=28.32, d2_k0last=20.72, d2_k1=30.07, d2_k10=32.11, tgt_k10=34.18, official=36.16,
                    ylim=(16, 42), title="navhard two-stage EPDMS", k1_label=(0.1, 25.9, "top"),
                    k100_below=False, k0_short=True),
}


def k_equiv(score, d):
    """K at which the D1 curve (log-linear between K10 and K100) reaches `score`."""
    t = (score - d["d1_k10"]) / (d["d1_k100"] - d["d1_k10"])
    return 1.0 + t, 10 ** (1.0 + t)


def shot_panel(ax, d, annotate_equiv):
    ax.axhline(d["official"], color=REF, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(XK[0] - 0.15, d["official"] + 0.35, f"Official RAP (D1-K100-aug) {d['official']:.2f}",
            fontsize=10, color=INK2, va="bottom")

    # D1: NAVSIM only
    xs = [XK[1], XK[10], XK[100]]
    ys = [d["d1_mean"], d["d1_k10"], d["d1_k100"]]
    ax.plot(xs, ys, color=D1, lw=2, zorder=3)
    for j, s in zip((-0.05, 0.0, 0.05), d["d1_seeds"]):
        ax.plot(XK[1] + j, s, "o", ms=5, mfc=SURFACE, mec=D1, mew=1.3, zorder=4)
    ax.errorbar(XK[1], d["d1_mean"], yerr=d["d1_sd"], fmt="o", ms=9, color=D1, mec=SURFACE, mew=1.5,
                capsize=4, lw=1.5, zorder=5)
    ax.plot(xs[1:], ys[1:], "o", ms=9, color=D1, mec=SURFACE, mew=1.5, zorder=5)
    lx, ly, lva = d["k1_label"]
    ax.text(lx, ly, f"{d['d1_mean']:.2f} ± {d['d1_sd']:.2f}", ha="left", va=lva, fontsize=10, color=INK2)
    ax.text(XK[10] + 0.07, d["d1_k10"] - 0.3, f"{d['d1_k10']:.2f}", ha="left", va="top", fontsize=10, color=INK2)
    if d["k100_below"]:
        ax.text(XK[100] + 0.07, d["d1_k100"] - 0.3, f"{d['d1_k100']:.2f}", ha="left", va="top", fontsize=10,
                color=INK2)
    else:
        ax.text(XK[100] - 0.07, d["d1_k100"] + 0.15, f"{d['d1_k100']:.2f}", ha="right", va="bottom", fontsize=10,
                color=INK2)

    # D2-mix: NAVSIM + nuScenes
    # nudge D2@K1 sideways when it would sit on top of the D1 mean (navhard)
    x1 = XK[1] + (0.07 if abs(d["d2_k1"] - d["d1_mean"]) < 1.5 else 0.0)
    ax.plot([XK[0], x1], [d["d2_k0e6"], d["d2_k1"]], color=D2, lw=1.5, ls=(0, (3, 2)), zorder=3)
    ax.plot([x1, XK[10]], [d["d2_k1"], d["d2_k10"]], color=D2, lw=2, zorder=3)
    ax.plot([x1, XK[10]], [d["d2_k1"], d["d2_k10"]], "o", ms=9, color=D2, mec=SURFACE, mew=1.5, zorder=6)
    ax.plot(XK[0], d["d2_k0e6"], "o", ms=9, mfc=SURFACE, mec=D2, mew=2, zorder=5)
    ax.plot(XK[0], d["d2_k0last"], "X", ms=9, color=D2, mec=SURFACE, mew=1, zorder=5)
    short = d.get("k0_short", False)
    ax.text(XK[0] + 0.08, d["d2_k0e6"], f"{d['d2_k0e6']:.2f}  " + ("e6" if short else "e6 (val-selected)"),
            va="center", fontsize=10, color=INK2)
    ax.text(XK[0] + 0.08, d["d2_k0last"], f"{d['d2_k0last']:.2f}  " + ("last" if short else "last e19 (strict K0)"),
            va="center", fontsize=10, color=INK2)
    ax.text(x1 - 0.07, d["d2_k1"] + 0.3, f"{d['d2_k1']:.2f}", ha="right", va="bottom", fontsize=10, color=INK2)
    ax.text(XK[10] - 0.07, d["d2_k10"] + 0.3, f"{d['d2_k10']:.2f}", ha="right", va="bottom", fontsize=10,
            color=INK2)

    # D2-tgt at K10
    ax.plot(XK[10] + 0.09, d["tgt_k10"], "s", ms=8, color=TGT, mec=SURFACE, mew=1.5, zorder=5)
    ax.text(XK[10] + 0.17, d["tgt_k10"], f"{d['tgt_k10']:.2f}", va="center", fontsize=10, color=INK2)

    # D2-K100 not run
    lo, hi = d["ylim"]
    ax.text(XK[100], lo + 0.6, "D2-K100\nnot run", ha="center", va="bottom", fontsize=10, color=D2,
            style="italic")

    if annotate_equiv:
        for score, label in ((d["d2_k1"], "D2@K1"), (d["d2_k10"], "D2@K10")):
            x_eq, k_eq = k_equiv(score, d)
            x0 = XK[1] if label == "D2@K1" else XK[10]
            ax.annotate("", xy=(x_eq, score), xytext=(x0 + 0.08, score),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1, ls=(0, (2, 2))), zorder=2)
            ax.text(x_eq + 0.05, score - 0.25, f"≈ D1 @ ~{k_eq:.0f}%", fontsize=10, color=INK2, va="top")

    # broken x axis between K0 and K1
    ax.set_xlim(XK[0] - 0.3, XK[100] + 0.3)
    ax.set_ylim(lo, hi)
    ax.set_xticks([XK[k] for k in (0, 1, 10, 100)])
    ax.set_xticklabels(["0\n(scene raster)", "1", "10", "100"])
    xb = (XK[0] + XK[1]) / 2
    for dx in (-0.03, 0.03):
        ax.plot([xb + dx - 0.03, xb + dx + 0.03], [lo - 0.4, lo + 0.4], color=MUTED, lw=1, clip_on=False)
    ax.set_xlabel("K = % of NAVSIM (target-rig) real images in training  (log scale)")
    ax.set_title(d["title"])


def fig1(out):
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.2))
    shot_panel(axes[0], SHOT["v1"], annotate_equiv=True)
    shot_panel(axes[1], SHOT["v2"], annotate_equiv=False)
    shot_panel(axes[2], SHOT["navhard"], annotate_equiv=False)
    handles = [
        Line2D([], [], color=D1, lw=2, marker="o", ms=8, mec=SURFACE, label="D1: NAVSIM only (K1 = 3 seeds, mean ± s.d.)"),
        Line2D([], [], color=D2, lw=2, marker="o", ms=8, mec=SURFACE, label="D2-mix: NAVSIM + nuScenes (native + rendered to NAVSIM rig)"),
        Line2D([], [], color=TGT, lw=0, marker="s", ms=8, mec=SURFACE, label="D2-tgt: NAVSIM + nuScenes rendered to NAVSIM rig only"),
        Line2D([], [], color=REF, lw=1.2, ls=(0, (5, 3)), label="Official RAP weights"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    footnote(fig, FOOT_COMMON.replace("navtest, PDMS x100.", "navtest v1 PDMS / v2 EPDMS and navhard EPDMS, x100.")
             + " navhard = two-stage EPDMS on the navhard split; its D1-K1 seed s.d. is ±3.18, "
             "so most gaps in that panel are within seed noise.\nInferred (left panel arrows): equivalent K by "
             "log-linear interpolation between D1-K10 and D1-K100; only 3 D1 points, single seed. Dashed D2 segment: "
             "the K0 point uses a checkpoint chosen on real-image val.")
    fig.savefig(out / "mtg1023_fig1_shot_curve.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: candidate pool vs selection
POOL = [  # name, family colour, oracle64, mean64, scorer, capture
    ("D1-K1\n(s0)", D1, 93.99, 63.83, 80.36, 0.55),
    ("D1-K10", D1, 94.30, 67.16, 82.83, 0.58),
    ("D1-K100", D1, 98.55, 78.02, 92.30, 0.70),
    (None,) * 6,
    ("D2-K0-scene\n-mix e6", D2, 98.17, 61.11, 82.30, 0.57),
    ("D2-K0-scene\n-mix last", D2, 98.54, 64.04, 77.66, 0.39),
    ("D2-K1\n-mix", D2, 98.23, 70.20, 85.73, 0.55),
    ("D2-K10\n-mix", D2, 98.41, 71.46, 89.40, 0.67),
    ("D2-K10\n-tgt", TGT, 98.21, 70.14, 88.66, 0.66),
    (None,) * 6,
    ("Official\n(D1-K100-aug)", REF, 98.31, 83.06, 93.80, 0.70),
]


def fig2(out):
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ticks, labels = [], []
    for i, (name, c, orc, mean, sc, cap) in enumerate(POOL):
        if name is None:
            continue
        ax.plot([i, i], [mean, orc], color=c, lw=3, solid_capstyle="round", zorder=2)
        ax.plot(i, orc, "^", ms=10, color=c, mec=SURFACE, mew=1.5, zorder=4)
        ax.plot(i, mean, "_", ms=16, mew=3, color=c, zorder=4)
        ax.plot(i, sc, "o", ms=11, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.text(i + 0.14, sc, f"{sc:.2f}", va="center", fontsize=10.5, color=INK)
        ax.text(i + 0.14, orc, f"{orc:.2f}", va="center", fontsize=9.5, color=INK2)
        ax.text(i + 0.14, mean, f"{mean:.2f}", va="center", fontsize=9.5, color=INK2)
        ax.text(i, 56.2, f"capture\n{cap:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
        ticks.append(i)
        labels.append(name)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_xlim(-0.6, len(POOL) - 0.4)
    ax.set_ylim(55, 103)
    ax.set_ylabel("v1 PDMS (x100)")
    ax.set_title("Where the gain comes from: candidate pool (oracle@64) vs selection (scorer)")
    ax.annotate("D1: oracle ≈ 94 until K100", xy=(0.5, 94.6), xytext=(0.5, 100.8), ha="center", fontsize=11,
                color=D1, arrowprops=dict(arrowstyle="-[, widthB=2.2", color=D1, lw=1))
    ax.annotate("D2: oracle 98.2–98.5 at every K, incl. K0", xy=(6, 99.0), xytext=(6, 101.3), ha="center",
                fontsize=11, color=D2, arrowprops=dict(arrowstyle="-[, widthB=15.5", color=D2, lw=1))
    handles = [
        Line2D([], [], lw=0, marker="^", ms=10, color=INK2, mec=SURFACE, label="oracle@64 (best of 64 candidates)"),
        Line2D([], [], lw=0, marker="o", ms=10, color=INK2, mec=SURFACE, label="scorer pick (reported PDMS)"),
        Line2D([], [], lw=0, marker="_", ms=14, mew=3, color=INK2, label="mean of 64 candidates"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    footnote(fig, "capture = (scorer − mean64) / (oracle64 − mean64). " + FOOT_COMMON)
    fig.savefig(out / "mtg1023_fig2_pool_vs_scorer.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: training-val curves
def load_curves():
    q = ROOT / "qsub_outputs"
    p100 = parse(q / "rap-e2a-p100.o9060304")
    p100 = {0: p100[0]}  # e1 in this log is the bar of the resumed epoch before its validation
    p100.update(parse(q / "rap-e2a-p100.o9060312"))
    return [  # label, colour, lw, ls, curve
        ("D2-K0-scene-mix", D2, 2.6, "-", parse(q / "rap-dk.o9062117")),
        ("D2-K1-mix", "#3d3c3a", 1.4, "-", parse(q / "rap-dk.o9062115")),
        ("D2-K10-mix", "#3d3c3a", 1.4, (0, (4, 2)), parse(q / "rap-e3.o8803003")),
        ("D1-K100", "#a3a29d", 1.4, "-", p100),
    ]


def fig3(out):
    curves = load_curves()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw=dict(width_ratios=(1.35, 1)))
    for ax, key, title, ylim in ((axes[0], "val/score", "Scorer-selected trajectory (val/score)", (0.70, 0.94)),
                                 (axes[1], "val/best_score", "Best of the candidates (val/best_score)",
                                  (0.84, 1.0))):
        for label, c, lw, ls, cur in curves:
            ep = sorted(e for e in cur if key in cur[e])
            ys = [cur[e][key] for e in ep]
            ax.plot(ep, ys, color=c, lw=lw, ls=ls, zorder=3 if c == D2 else 2)
            if key == "val/score":
                ax.text(19.4, ys[-1], f"{label}  {ys[-1]:.3f}", va="center", fontsize=10.5,
                        color=INK if c == D2 else INK2)
        ax.set_xlim(0, 19)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.set_xlabel("epoch")
        ax.set_title(title)
    k0 = curves[0][4]
    ax = axes[0]
    ax.plot(6, k0[6]["val/score"], "o", ms=9, mfc=SURFACE, mec=D2, mew=2, zorder=5)
    ax.plot(19, k0[19]["val/score"], "X", ms=9, color=D2, mec=SURFACE, zorder=5, clip_on=False)
    ax.annotate(f"peak {k0[6]['val/score']:.3f} @ e6\n→ ckpt 'e6' (val-selected)", xy=(6, k0[6]["val/score"]),
                xytext=(7.0, 0.715), fontsize=10.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    axes[0].set_ylabel("PDM score on training-val")
    axes[0].set_xlim(0, 19)
    axes[0].spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.97, wspace=0.55, top=0.9, bottom=0.2)
    axes[1].text(5.5, 0.865, "D2-K0 stays at 0.970–0.985 from e3 on,\nlevel with D2-K1/K10-mix:\n"
                 "the best candidate holds, only the pick degrades", fontsize=10.5, color=INK2)
    footnote(fig, "training-val = 214 NAVSIM val logs, real images (so the 'e6' choice uses target real images; "
             "'last' = strict K0). Per-epoch values = last progress-bar line of each epoch (qsub_outputs logs).\n"
             "Inferred: the K0 scorer is supervised only on NAVSIM raster + nuScenes, and its ranking drifts away "
             "from real-image NAVSIM ranking as training goes on.", y=0.01)
    fig.savefig(out / "mtg1023_fig3_val_curves.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: paired differences
SEED_SD = 1.15
PAIRS = [  # group, label, delta, lo, hi
    ("External data at fixed K", "D2-K10-mix − D1-K10", 6.56, 5.73, 7.33),
    ("External data at fixed K", "D2-K1-mix − D1-K1 (s0)", 5.37, 4.32, 6.36),
    ("External data at fixed K", "D2-K1-mix − D1-K1 (s1)", 7.61, 6.53, 8.75),
    ("External data at fixed K", "D2-K1-mix − D1-K1 (s2)", 6.97, 5.70, 8.28),
    ("Cross-K (equivalent shot)", "D2-K1-mix − D1-K10", 2.90, 1.90, 3.90),
    ("Cross-K (equivalent shot)", "D2-K0-scene-mix e6 − D1-K1 (s0)", 1.94, 0.95, 2.84),
    ("Cross-K (equivalent shot)", "D2-K0-scene-mix e6 − D1-K10", -0.54, -1.61, 0.48),
    ("Target real images", "D1-K100 − D1-K10", 9.47, 8.63, 10.29),
    ("Target real images", "D1-K1 (s0) − D1-K10", -2.47, -3.22, -1.78),
    ("Target real images", "D2-K0-scene-mix e6 − D2-K10-mix", -7.10, -7.89, -6.32),
    ("Checkpoint / recipe", "D2-K0-scene-mix last − e6", -4.64, -5.63, -3.56),
    ("Checkpoint / recipe", "Official (aug) − D1-K100", 1.50, 0.90, 2.14),
]


def fig4(out):
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.axvspan(-SEED_SD, SEED_SD, color="#ecebe7", zorder=0)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    y, yt, yl, prev = 0, [], [], None
    for g, label, d, lo, hi in PAIRS:
        if g != prev:
            y -= 0.6 if prev is not None else 0
            ax.text(-12.4, y, g, fontsize=11, fontweight="bold", color=INK, va="center")
            y -= 1
            prev = g
        clear = lo > SEED_SD or hi < -SEED_SD
        ax.plot([lo, hi], [y, y], color=INK, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(d, y, "o", ms=9, mfc=INK if clear else SURFACE, mec=INK, mew=1.8, zorder=4)
        ax.text(11.0, y, f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=10.5, color=INK2)
        yt.append(y)
        yl.append(label)
        y -= 1
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=10.5)
    ax.set_xlim(-12.5, 15.5)
    ax.set_xticks(range(-10, 11, 2))
    ax.set_ylim(y + 0.3, 1.0)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δ v1 PDMS (x100), paired per token, 95% CI (log-clustered bootstrap, 136 logs)")
    ax.set_title("Which differences are larger than seed noise?")
    ax.text(0, 0.75, f"D1-K1 seed s.d. ±{SEED_SD:.2f}", ha="center", va="bottom", fontsize=10, color=INK2)
    handles = [
        Line2D([], [], lw=0, marker="o", ms=9, color=INK, label="CI entirely outside the seed band"),
        Line2D([], [], lw=0, marker="o", ms=9, mfc=SURFACE, mec=INK, mew=1.8, label="CI overlaps the seed band"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    footnote(fig, "Bootstrap CIs cover test-set sampling only, not training seed. Seed band = sample s.d. of 3 "
             "D1-K1 seeds (80.36 / 78.12 / 78.76).\nInferred: a difference of two single-seed runs carries about "
             "√2 × 1.15 ≈ 1.6 of seed noise if both runs have the same seed variance.")
    fig.savefig(out / "mtg1023_fig4_paired_forest.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 5: experiment grid status
COLS = ["K0-strict", "K0-scene", "K1", "K10", "K100"]
ROWS = [
    ("D1", "NAVSIM only\n(K0-strict:\nnuScenes only)"),
    ("D2", "NAVSIM\n+ nuScenes"),
    ("D3", "+ Waymo or\nArgoverse"),
    ("D4", "+ both"),
]
CELLS = {  # (row, col): (status, text)
    (0, 0): ("done", "D1-K0-strict-nat (E1)\n43.8\n(-mix: deferred)"),
    (0, 1): ("na", "not planned"),
    (0, 2): ("done", "79.08 ± 1.15\n(3 seeds)"),
    (0, 3): ("done", "82.83 (E2a)"),
    (0, 4): ("done", "92.30\n(official aug 93.80)"),
    (1, 0): ("na", "n/a\n(strict ⇒ D1)"),
    (1, 1): ("done", "-mix 82.30 (e6*)\n77.66 (last)"),
    (1, 2): ("done", "-mix 85.73"),
    (1, 3): ("done", "-mix 89.40\n-tgt 88.66 · -ft 87.30"),
    (1, 4): ("missing", "MISSING"),
    (2, 4): ("missing", "MISSING"),
    (3, 4): ("missing", "MISSING"),
}
STYLE = {
    "done": dict(fc="#e3eefb", ec=D1, ls="-", tag="done"),
    "missing": dict(fc=SURFACE, ec=D2, ls="--", tag="missing"),
    "todo": dict(fc=SURFACE, ec="#c9c8c3", ls="-", tag="not started"),
    "na": dict(fc="#f1f1ef", ec="#dcdbd6", ls="-", tag=""),
}


def fig5(out):
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_axis_off()
    x0, y0, cw, ch = 2.2, 0.0, 2.0, 1.15
    ncol, nrow = len(COLS), len(ROWS)
    for j, c in enumerate(COLS):
        ax.text(x0 + j * cw + cw / 2, y0 + 0.25, c, ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=INK)
    for i, (r, sub) in enumerate(ROWS):
        yc = y0 - i * ch - ch / 2
        ax.text(0.15, yc, r, ha="left", va="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(0.75, yc, sub, ha="left", va="center", fontsize=9.5, color=INK2)
        for j in range(ncol):
            st, txt = CELLS.get((i, j), ("todo", "—"))
            s = STYLE[st]
            x, y = x0 + j * cw + 0.06, y0 - (i + 1) * ch + 0.06
            ax.add_patch(mpatches.FancyBboxPatch((x, y), cw - 0.12, ch - 0.12, boxstyle="round,pad=0,rounding_size=0.08",
                                                 fc=s["fc"], ec=s["ec"], ls=s["ls"], lw=1.6))
            ax.text(x + (cw - 0.12) / 2, y + (ch - 0.12) / 2 + 0.08, txt, ha="center", va="center", fontsize=10,
                    color=D2 if st == "missing" else (INK if st == "done" else MUTED),
                    fontweight="bold" if st == "missing" else "normal")
            if s["tag"]:
                ax.text(x + cw - 0.2, y + 0.07, s["tag"], ha="right", va="bottom", fontsize=8, color=INK2)
    # axis 1: K100 column, D1 -> D4
    xa = x0 + 4 * cw
    ax.add_patch(mpatches.Rectangle((xa - 0.02, y0 - nrow * ch - 0.02), cw + 0.04, nrow * ch + 0.04, fill=False,
                                    ec=INK, lw=2.4))
    ax.text(xa + cw + 0.15, y0 - 0.2, "Axis ①  (Single target rig)\nD = 1 → 4, target data = full\n"
            "only D1 is done", ha="left", va="top", fontsize=11, color=INK)
    # axis 2: D2 row, K0 -> K100
    ax.add_patch(mpatches.Rectangle((x0 + cw - 0.02, y0 - 2 * ch - 0.02), 4 * cw + 0.04, ch + 0.04, fill=False,
                                    ec=INK, lw=2.4, ls=(0, (6, 3))))
    ax.text(xa + cw + 0.15, y0 - ch - 0.35, "Axis ②  (Single target rig)\nD = 2, K = 0 → full\nK100 end missing",
            ha="left", va="top", fontsize=11, color=INK)
    ax.text(xa + cw + 0.15, y0 - 2 * ch - 0.55,
            "Multi target rig\nx = D (≤ 3), log × rig\nfull cross-render, no K axis\nnot started",
            ha="left", va="top", fontsize=11, color=INK)
    ax.text(x0, y0 - nrow * ch - 0.3,
            "D3/D4 blockers: WOD-E2E has no 3D GT → no raster and no PDM labels; no Argoverse adapter in the "
            "codebase yet.  * e6 checkpoint chosen on real-image val.",
            ha="left", va="top", fontsize=10, color=INK2)
    ax.set_xlim(0, xa + cw + 3.4)
    ax.set_ylim(y0 - nrow * ch - 0.75, y0 + 0.75)
    ax.set_title("Experiment grid: D (independent data sources) × K (% target real images), v1 PDMS",
                 loc="left")
    fig.tight_layout()
    fig.savefig(out / "mtg1023_fig5_grid_status.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 6: T2 rig factors
HEIGHT = "#6250d6"
LONGI = "#1baf7a"
LATERAL = "#8a8984"
T2_FOOT = ("T2: official RAP v1 weights, raster input, navtest 1,501 tokens, calibration updated with the rig. "
           "ΔPDMS vs the identity rig (c1), 95% log-clustered bootstrap CI.\nOfficial raster convention renders the "
           "camera 0.8 m above the nominal calibration (raster F0 ≈ 2.29 m vs real ≈ 1.49 m), so raster results do "
           "not transfer directly to real images.")
T2_SWEEP = {  # factor: [(displacement m, Δ, lo, hi)]
    "camera height": [(-0.7, -15.3, -18.2, -12.7), (-0.5, -11.1, -13.0, -9.2), (-0.3, -5.6, -6.7, -4.4),
                      (0.0, 0.0, 0.0, 0.0), (0.3, -5.8, -7.5, -4.2), (0.5, -10.4, -12.7, -8.0),
                      (1.0, -10.7, -12.9, -8.5)],
    "longitudinal": [(-1.0, -0.5, -1.0, -0.1), (0.0, 0.0, 0.0, 0.0), (1.0, -0.4, -0.8, -0.0)],
    "lateral": [(-0.5, -0.6, -1.1, -0.1), (0.0, 0.0, 0.0, 0.0), (0.5, 0.1, -0.3, 0.6)],
}
T2_SPLIT = [  # preset, Δ new, Δ fix_all (selection only), Δ oracle@64, Δ mean64
    ("height\n−0.7 m", -15.3, -3.3, -0.6, -15.1),
    ("height\n+1.0 m", -10.7, -3.4, -1.3, -10.5),
    ("focal\n×0.8", -4.9, -1.4, 0.1, -6.4),
    ("focal\n×1.25", -3.9, -1.1, -0.3, -4.7),
    ("longitudinal\n+1.0 m", -0.4, -0.0, -0.0, 0.1),
]


def fig6(out):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw=dict(width_ratios=(1.1, 1)))
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    for (name, pts), c, ms in zip(T2_SWEEP.items(), (HEIGHT, LONGI, LATERAL), (9, 8, 8)):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=c, lw=2, zorder=3)
        for x, d, lo, hi in pts:
            if x == 0:
                continue
            ax.plot([x, x], [lo, hi], color=c, lw=1.5, zorder=3)
            ax.plot(x, d, "o", ms=ms, color=c, mec=SURFACE, mew=1.5, zorder=4)
    for x, d, _, _ in T2_SWEEP["camera height"]:
        if x != 0:
            left = x == 0.5
            ax.text(x - 0.06 if left else x + 0.05, d - (0.5 if left else 0), f"{d:+.1f}".replace("-", "−"),
                    va="center", ha="right" if left else "left", fontsize=10, color=INK2)
    ax.text(-1.1, -8.0, "camera height", color=HEIGHT, fontsize=11, fontweight="bold")
    ax.text(0.62, -2.4, "longitudinal ±1.0 m: −0.5 / −0.4\nlateral ±0.5 m: −0.6 / +0.1", color=INK2,
            fontsize=10.5)
    ax.set_xlim(-1.15, 1.25)
    ax.set_ylim(-19, 2)
    ax.set_xlabel("camera displacement (m)  ·  height: up = +, longitudinal: forward = +")
    ax.set_ylabel("Δ PDMS vs identity rig (x100)")
    ax.set_title("Only height hurts: dose-dependent on both sides")
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=8, mec=SURFACE, label=n)
               for n, c in zip(T2_SWEEP, (HEIGHT, LONGI, LATERAL))]
    ax.legend(handles=handles, loc="lower right", fontsize=10.5)

    ax = axes[1]
    series = [("total (Δ new)", 1, INK), ("selection only (candidates fixed)", 2, "#a3a29d"),
              ("mean of 64 candidates", 4, HEIGHT), ("best of 64 (oracle)", 3, LONGI)]
    w = 0.19
    for k, (label, col, c) in enumerate(series):
        xs = [i + (k - 1.5) * w for i in range(len(T2_SPLIT))]
        ys = [row[col] for row in T2_SPLIT]
        ax.bar(xs, ys, width=w - 0.03, color=c, label=label, zorder=3)
        if col in (1, 2):
            for x, y in zip(xs, ys):
                if abs(y) >= 0.95:
                    ax.text(x, y - 0.3, f"{y:.1f}".replace("-", "−"), ha="center", va="top", fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)
    ax.set_xticks(range(len(T2_SPLIT)))
    ax.set_xticklabels([r[0] for r in T2_SPLIT], fontsize=10.5)
    ax.set_ylim(-19, 2)
    ax.set_title("The height loss is in generation, not coverage")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    footnote(fig, T2_FOOT + "  'Selection only' = c1 candidates teacher-forced, new images read in every round.")
    fig.savefig(out / "mtg1023_fig6_t2_factors.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 7: feature-swap rescue
SWAP = [  # preset, loss (points), rescue share %, lo, hi, is_height
    ("height +1.0 m  (loss 10.7)", 10.7, 1, -6, 8, True),
    ("height −0.7 m  (loss 15.3)", 15.3, 2, -3, 7, True),
    ("focal ×0.8  (loss 4.9)", 4.9, 24, 9, 39, False),
    ("focal ×1.25  (loss 3.9)", 3.9, 23, 6, 40, False),
]


def fig7(out):
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.axvspan(-10, 25, color="#ecebe7", zorder=0)
    ax.axvline(25, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.axvline(50, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for i, (name, loss, d, lo, hi, is_h) in enumerate(SWAP):
        y = -i - (0.5 if not is_h else 0)
        c = HEIGHT if is_h else LATERAL
        ax.plot([lo, hi], [y, y], color=c, lw=2.5, solid_capstyle="round", zorder=3)
        ax.plot(d, y, "o", ms=11, color=c, mec=SURFACE, mew=2, zorder=4)
        ax.text(hi + 1.5, y, f"{d}%  [{lo}, {hi}]".replace("-", "−"), va="center", fontsize=11, color=INK)
        ax.text(-11.5, y, name, va="center", ha="right", fontsize=11, color=INK)
    ax.text(-11.5, -2.0, "positive control", ha="right", va="center", fontsize=10, color=INK2, style="italic")
    ax.text(25.5, 0.75, "ENC-DEAD if both height CIs end ≤ 25%", fontsize=10, color=INK2, va="bottom")
    ax.text(50.5, 0.75, "ENC-MATTERS if both start ≥ 50%", fontsize=10, color=INK2, va="bottom")
    ax.set_xlim(-10, 70)
    ax.set_ylim(-4.3, 1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the PDMS loss rescued by swapping in the identity-rig feature of the same surface (%)")
    ax.set_title("Same-surface features do not rescue the height loss (pre-registered verdict: ENC-DEAD)")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.88, bottom=0.27)
    footnote(fig, "swap_all: F0+L0+R0 feature cells whose centre shows a surface also seen under c1 take the c1 "
             "feature of that surface; decoder and calibration unchanged. Only ~11–12% of F0 cells are swapped "
             "(drawn surfaces);\nbackground cells keep the new-rig feature. Unwarped control (swap_all_bg) rescues "
             "40% [21, 58] at +1.0 m but 6% at −0.7 m — not a bound, unexplained. Raster, official v1 weights, "
             "1,501 tokens.")
    fig.savefig(out / "mtg1023_fig7_swap_rescue.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/slides/assets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        f(a.out)
    print("k_equiv v1:", [round(k_equiv(s, SHOT["v1"])[1], 1) for s in (85.73, 89.40)])

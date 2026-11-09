"""Updated D x K grid and shot curve for the 2026-11-06 meeting.

Same style as plot_mtg1023_figs.py (fig5 grid, fig1 shot curve); only the data change.
New since 10/23 (all at K1): fixed 23k external-raster budget cells (nuScenes / WOMD / both, seeds 0-1)
and the step-matched D1-K1 control (seed 0). Numbers copied from
eval/summary_DK_K1grid_s0_20261104.md, summary_DK_K1grid_s1_20261105.md,
summary_DK_K1stepmatch_e9_20261105.md, summary_DK_D3K1_20261031.md (navtest, x100).

    python scripts/analysis/plot_mtg1106_figs.py [--out docs/slides/assets]

Writes mtg1106_fig_grid.png and mtg1106_fig_shot_curve.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_mtg1023_figs as base  # noqa: E402  (also applies the 10/23 rcParams)

D1, D2, TGT, REF = base.D1, base.D2, base.TGT, base.REF
INK, INK2, MUTED, SURFACE = base.INK, base.INK2, base.MUTED, base.SURFACE
WOMD = "#8a4fd6"    # NAVSIM + WOMD (rendered to NAVSIM rig)
D3C = "#a0632b"     # NAVSIM + nuScenes + WOMD, 11.5k each (brown: distinct from the green nuScenes line)
XK = base.XK

# ------------------------------------------------------------------ new K1 points (s0, s1)
NEW = {
    "v1": dict(nusc=(83.20, 81.82), womd=(84.62, 82.95), d3=(83.79, 82.35), step=80.86),
    "v2": dict(nusc=(77.63, 76.78), womd=(78.71, 76.71), d3=(78.01, 76.44), step=74.66),
    "navhard": dict(nusc=(26.96, 29.72), womd=(27.19, 30.56), d3=(26.55, 33.16), step=24.63),
}


def mean(p):
    return sum(p) / len(p) + 1e-9  # round half up (83.785 -> 83.79)


# lines from 10/23 (single runs except D1-K1 = 3-seed mean ± s.d.)
OLD = {
    "v1": dict(d1=(79.08, 82.83, 92.30), d1_sd=1.15, mix=(85.73, 89.40), k0=(82.30, 77.66), tgt10=88.66, off=93.80),
    "v2": dict(d1=(72.50, 76.87, 86.16), d1_sd=1.44, mix=(80.68, 83.28), k0=(77.20, 74.02), tgt10=82.66, off=86.67),
    "navhard": dict(d1=(29.91, 36.12, 37.38), d1_sd=3.18, mix=(30.07, 32.11), k0=(28.32, 20.72), tgt10=34.18,
                    off=36.16),
}
PANEL = {
    "v1": dict(title="NAVSIM v1 · PDMS", ylim=(74, 96)),
    "v2": dict(title="NAVSIM v2 · EPDMS", ylim=(68, 90)),
    "navhard": dict(title="navhard · EPDMS", ylim=(16, 42)),
}
X = {0: -0.8, 1: 0.0, 10: 1.0, 100: 2.0}
DX = dict(step=-0.13, mix=0.0, nusc=0.13, d3=0.26, womd=0.39)   # spread of the markers at K1
MS = 10


def shot_panel(ax, key):
    o, n, p = OLD[key], NEW[key], PANEL[key]
    lo, hi = p["ylim"]

    def lab(x, y, dy=0.0, ha="left", dx=0.07):
        ax.text(x + (dx if ha == "left" else -dx), y + dy, f"{y:.1f}", ha=ha, va="center", fontsize=10, color=INK2)

    ax.axhline(o["off"], color=REF, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(X[0] - 0.3, o["off"] + (hi - lo) * 0.012, f"{o['off']:.1f}", ha="left", va="bottom", fontsize=10,
            color=INK2)
    nh = key == "navhard"   # navhard: D1 and the orange line cross, so a few labels move
    # NAVSIM only
    ax.plot([X[1], X[10], X[100]], o["d1"], color=D1, lw=2.2, zorder=3)
    ax.errorbar(X[1], o["d1"][0], yerr=o["d1_sd"], fmt="o", ms=MS, color=D1, mec=SURFACE, mew=1.5, capsize=4,
                lw=1.4, zorder=5)
    ax.plot([X[10], X[100]], o["d1"][1:], "o", ms=MS, color=D1, mec=SURFACE, mew=1.5, zorder=5)
    lab(X[1], o["d1"][0], dy=-(hi - lo) * 0.05, ha="right" if nh else "left")
    lab(X[10], o["d1"][1], dy=(hi - lo) * 0.035 if nh else -(hi - lo) * 0.035, ha="right" if nh else "left")
    lab(X[100], o["d1"][2], dy=(hi - lo) * 0.04 if nh else 0.0, ha="right")
    # + nuScenes real + raster (dashed to K0: that checkpoint was picked on real-image val)
    ax.plot([X[0], X[1]], [o["k0"][0], o["mix"][0]], color=D2, lw=1.6, ls=(0, (3, 2)), zorder=3)
    ax.plot([X[1], X[10]], o["mix"], color=D2, lw=2.2, zorder=3)
    ax.plot([X[1], X[10]], o["mix"], "o", ms=MS, color=D2, mec=SURFACE, mew=1.5, zorder=6)
    ax.plot(X[0], o["k0"][0], "o", ms=MS, mfc=SURFACE, mec=D2, mew=2, zorder=5)
    ax.plot(X[0], o["k0"][1], "X", ms=MS, color=D2, mec=SURFACE, mew=1, zorder=5)
    lab(X[0], o["k0"][0])
    lab(X[0], o["k0"][1])
    lab(X[1], o["mix"][0], dy=(hi - lo) * 0.035, ha="right")
    lab(X[10], o["mix"][1], dy=-(hi - lo) * 0.035 if nh else (hi - lo) * 0.035, ha="left" if nh else "right")
    # + nuScenes raster only: new K1 point joined to the 10/23 K10 point
    ax.plot([X[1] + DX["nusc"], X[10] + 0.07], [mean(n["nusc"]), o["tgt10"]], color=TGT, lw=2.2, zorder=3)
    ax.plot([X[1] + DX["nusc"], X[10] + 0.07], [mean(n["nusc"]), o["tgt10"]], "s", ms=MS - 1, color=TGT,
            mec=SURFACE, mew=1.5, zorder=6)
    lab(X[10] + 0.07, o["tgt10"])
    # other new K1 points
    ax.plot(X[1] + DX["womd"], mean(n["womd"]), "D", ms=MS - 1, color=WOMD, mec=SURFACE, mew=1.5, zorder=6)
    ax.plot(X[1] + DX["d3"], mean(n["d3"]), "^", ms=MS + 1, color=D3C, mec=SURFACE, mew=1.5, zorder=6)
    ax.plot(X[1] + DX["step"], n["step"], "D", ms=MS - 2, mfc=SURFACE, mec=D1, mew=1.8, zorder=6)
    # axes
    ax.set_xlim(X[0] - 0.35, X[100] + 0.3)
    ax.set_ylim(lo, hi)
    ax.set_xticks([X[k] for k in (0, 1, 10, 100)])
    ax.set_xticklabels(["0", "1", "10", "100"])
    xb = (X[0] + X[1]) / 2
    for dxb in (-0.03, 0.03):
        ax.plot([xb + dxb - 0.03, xb + dxb + 0.03], [lo - 0.012 * (hi - lo), lo + 0.012 * (hi - lo)], color=MUTED,
                lw=1, clip_on=False)
    ax.set_xlabel("NAVSIM real images in training (%)")
    ax.set_title(p["title"])


def shot_legend(fig, ncol):
    h = [
        Line2D([], [], color=D1, lw=2.2, marker="o", ms=MS, mec=SURFACE, label="NAVSIM only"),
        Line2D([], [], color=D2, lw=2.2, marker="o", ms=MS, mec=SURFACE, label="+ nuScenes (real + raster)"),
        Line2D([], [], color=TGT, lw=2.2, marker="s", ms=MS - 1, mec=SURFACE, label="+ nuScenes (raster)"),
        Line2D([], [], color=WOMD, lw=0, marker="D", ms=MS - 1, mec=SURFACE, label="+ WOMD (raster)"),
        Line2D([], [], color=D3C, lw=0, marker="^", ms=MS + 1, mec=SURFACE, label="+ nuScenes + WOMD (raster)"),
        Line2D([], [], color=D1, lw=0, marker="D", ms=MS - 2, mfc=SURFACE, mec=D1, mew=1.8,
               label="NAVSIM only, trained as long"),
        Line2D([], [], color=D2, lw=0, marker="o", ms=MS, mfc=SURFACE, mec=D2, mew=2, label="K0: best epoch"),
        Line2D([], [], color=D2, lw=0, marker="X", ms=MS, mec=SURFACE, label="K0: last epoch"),
        Line2D([], [], color=REF, lw=1.2, ls=(0, (5, 3)), label="Official RAP"),
    ]
    fig.legend(handles=h, loc="upper center", ncol=ncol, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.6)


def fig_shot(out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    for ax, key in zip(axes, ("v1", "v2")):
        shot_panel(ax, key)
    shot_legend(fig, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out / "mtg1106_fig_shot_curve.png", dpi=200)
    plt.close(fig)
    # navhard kept as a separate backup figure
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    shot_panel(ax, "navhard")
    shot_legend(fig, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(out / "mtg1106_fig_shot_curve_navhard.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ grid
ROWS = [
    ("D1", "NAVSIM only\n(K0-strict:\nnuScenes only)"),
    ("D2", "NAVSIM\n+ nuScenes\nor WOMD"),
    ("D3", "NAVSIM\n+ nuScenes\n+ WOMD"),
    ("D4", "+ a 4th source"),
]
CELLS = dict(base.CELLS)
CELLS.update({
    (0, 2): ("new", "79.08 ± 1.15 (3 seeds)\nsame steps: 80.86"),
    (1, 1): ("done", "-mix 82.30 (e6*)\n77.66 (last)"),
    (1, 2): ("new", "-mix 85.73\nnuSc 82.51\nWOMD 83.79"),
    (2, 2): ("new", "-mix 86.09\nnuSc+WOMD 83.07"),
})
STYLE = dict(base.STYLE)
STYLE["new"] = dict(fc="#e4f5ee", ec=TGT, ls="-", tag="new since 10/23")


def fig_grid(out):
    base.ROWS, base.CELLS, base.STYLE = ROWS, CELLS, STYLE
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_axis_off()
    x0, y0, cw, ch = 2.2, 0.0, 2.0, 1.15
    cols = base.COLS
    ncol, nrow = len(cols), len(ROWS)
    for j, c in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, y0 + 0.25, c, ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
    for i, (r, sub) in enumerate(ROWS):
        yc = y0 - i * ch - ch / 2
        ax.text(0.15, yc, r, ha="left", va="center", fontsize=14, fontweight="bold", color=INK)
        ax.text(0.75, yc, sub, ha="left", va="center", fontsize=9.5, color=INK2)
        for j in range(ncol):
            st, txt = CELLS.get((i, j), ("todo", "—"))
            s = STYLE[st]
            x, y = x0 + j * cw + 0.06, y0 - (i + 1) * ch + 0.06
            ax.add_patch(mpatches.FancyBboxPatch((x, y), cw - 0.12, ch - 0.12,
                                                 boxstyle="round,pad=0,rounding_size=0.08",
                                                 fc=s["fc"], ec=s["ec"], ls=s["ls"], lw=1.6))
            ax.text(x + (cw - 0.12) / 2, y + (ch - 0.12) / 2 + 0.08, txt, ha="center", va="center",
                    fontsize=9.5 if st == "new" else 10,
                    color=D2 if st == "missing" else (INK if st in ("done", "new") else MUTED),
                    fontweight="bold" if st == "missing" else "normal")
            if s["tag"]:
                ax.text(x + cw - 0.2, y + 0.07, s["tag"], ha="right", va="bottom", fontsize=8,
                        color=TGT if st == "new" else INK2)
    xa = x0 + 4 * cw
    ax.add_patch(mpatches.Rectangle((xa - 0.02, y0 - nrow * ch - 0.02), cw + 0.04, nrow * ch + 0.04, fill=False,
                                    ec=INK, lw=2.4))
    ax.text(xa + cw + 0.15, y0 - 0.2, "Axis ①  (Single target rig)\nD = 1 → 4, target data = full\n"
            "only D1 is done", ha="left", va="top", fontsize=11, color=INK)
    ax.add_patch(mpatches.Rectangle((x0 + cw - 0.02, y0 - 2 * ch - 0.02), 4 * cw + 0.04, ch + 0.04, fill=False,
                                    ec=INK, lw=2.4, ls=(0, (6, 3))))
    ax.text(xa + cw + 0.15, y0 - ch - 0.35, "Axis ②  (Single target rig)\nD = 2, K = 0 → full\nK100 end missing",
            ha="left", va="top", fontsize=11, color=INK)
    ax.text(xa + cw + 0.15, y0 - 2 * ch - 0.55,
            "Multi target rig\nx = D (≤ 3), log × rig\nfull cross-render, no K axis\nnot started",
            ha="left", va="top", fontsize=11, color=INK)
    ax.text(x0, y0 - nrow * ch - 0.3,
            "Since 10/23 only the K1 column moved. nuSc / WOMD / nuSc+WOMD = same 23k external raster rows and "
            "2,520 steps, mean of 2 seeds;\n-mix cells: 1 seed, larger budget. WOMD has no camera images → raster "
            "only, rendered to the NAVSIM rig.\nStill no source with a different rig and real images (WOD-E2E: no 3D GT; "
            "no Argoverse adapter).  * e6 chosen on real-image val.",
            ha="left", va="top", fontsize=9.5, color=INK2)
    ax.set_xlim(0, xa + cw + 3.4)
    ax.set_ylim(y0 - nrow * ch - 1.15, y0 + 0.75)
    ax.set_title("Experiment grid: D (independent data sources) × K (% target real images), v1 PDMS", loc="left")
    fig.tight_layout()
    fig.savefig(out / "mtg1106_fig_grid.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Result 1: K1 cells, two seeds
# (label, hypothesis, colour, [(seed, v1, gain vs same-seed D1-K1, CI lo, CI hi) ...]); CI = paired, log-clustered
R1_CELLS = [
    ("D1-K1\n(baseline)", "reference", D1, [(0, 80.36, None, None, None), (1, 78.12, None, None, None)]),
    ("D1-K1,\nsame steps", "H1", D1, [(0, 80.86, 0.50, -0.47, 1.61)]),
    ("+ nuScenes\n23k", "H3", TGT, [(0, 83.20, 2.84, 1.86, 3.85), (1, 81.82, 3.70, 2.53, 4.79)]),
    ("+ WOMD\n23k", "H3", WOMD, [(0, 84.62, 4.26, 3.41, 5.19), (1, 82.95, 4.83, 3.82, 5.73)]),
    ("+ nuScenes + WOMD\n11.5k each", "H2", D3C, [(0, 83.79, 3.43, 2.22, 4.62), (1, 82.35, 4.23, 3.35, 5.06)]),
]


def fig_result1(out):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    w = 0.34
    for i, (name, hyp, col, runs) in enumerate(R1_CELLS):
        for seed, v, g, lo, hi in runs:
            x = i + (seed - 0.5) * w * 1.05
            ax.bar(x, v, width=w, color=col, alpha=1.0 if seed == 0 else 0.45, ec=SURFACE, lw=1, zorder=2)
            if lo is not None:
                ax.errorbar(x, v, yerr=[[g - lo], [hi - g]], fmt="none", ecolor=INK, elinewidth=1.3, capsize=4,
                            zorder=4)
            top = v + (hi - g if hi is not None else 0)
            lab = f"{v:.2f}" if g is None else f"{v:.2f}\n(+{g:.2f})"
            ax.text(x, top + 0.25, lab, ha="center", va="bottom", fontsize=9.5, color=INK, linespacing=1.1)
        if len(runs) == 1:
            ax.text(i + 0.5 * w * 1.05, 74.4, "seed 1\nnot run", ha="center", va="bottom", fontsize=8.5,
                    color=MUTED, style="italic")
        ax.text(i, 95.2, hyp, ha="center", va="center", fontsize=12, color=col, fontweight="bold")
    for y, lab in ((82.83, "D1-K10 (10 % real) 82.83"), (92.30, "D1-K100 (100 % real) 92.30")):
        ax.axhline(y, color=REF, lw=1.2, ls=(0, (5, 3)), zorder=1)
        ax.text(-0.55, y + 0.25, lab, ha="left", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(range(len(R1_CELLS)))
    ax.set_xticklabels([c[0] for c in R1_CELLS], fontsize=10.5)
    ax.set_xlim(-0.6, len(R1_CELLS) - 0.4)
    ax.set_ylim(74, 96.5)
    ax.set_ylabel("navtest v1 PDMS (x100)")
    ax.set_title("Result 1: K1 (1 % real) + 23k external raster, same 2,520 steps")
    handles = [mpatches.Patch(color=INK2, alpha=1.0, label="seed 0"),
               mpatches.Patch(color=INK2, alpha=0.45, label="seed 1"),
               Line2D([], [], color=INK, lw=1.3, marker="|", ms=8, label="95 % CI of the gain over D1-K1 (same seed)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    base.footnote(fig,
                  "Bars: v1 PDMS of each run; (+x) = gain over D1-K1 of the same seed. Whiskers: paired 95 % CI of that gain "
                  "(log-clustered bootstrap, 136 logs), drawn around the bar top —\nthey say whether the gain is real, "
                  "not the run's absolute uncertainty (eval/summary_DK_K1grid_*, summary_DK_D3vsD1_20261106.md)."
                  "\nDashed lines: single seed; D1-K10 trained ~680 steps. "
                  "Seeds also change the 1 % split, so compare within a seed.")
    fig.savefig(out / "mtg1106_fig_result1.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Result 2: generation vs selection
POOL = [  # name, colour, oracle64, mean64, pick
    ("D1-K1\ns0", D1, 93.99, 63.83, 80.36), ("D1-K1\ns1", D1, 93.34, 64.88, 78.12),
    ("same\nsteps", D1, 95.56, 70.35, 80.86), None,
    ("+ nuSc\ns0", TGT, 97.24, 71.31, 83.20), ("+ nuSc\ns1", TGT, 97.57, 65.07, 81.82),
    ("+ WOMD\ns0", WOMD, 98.09, 68.13, 84.62), ("+ WOMD\ns1", WOMD, 98.00, 62.39, 82.95),
    ("+ both\ns0", D3C, 97.67, 66.42, 83.79), ("+ both\ns1", D3C, 97.09, 66.74, 82.35), None,
    ("D1-K10", D1, 94.30, 67.16, 82.83), ("D1-K100", D1, 98.55, 78.02, 92.30),
]


def fig_result2(out):
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ticks, labels = [], []
    for i, row in enumerate(POOL):
        if row is None:
            continue
        name, c, orc, mean_, sc = row
        ax.plot([i, i], [sc, orc], color=c, lw=6, alpha=0.25, solid_capstyle="butt", zorder=2)
        ax.plot([i, i], [mean_, orc], color=c, lw=1.2, zorder=1)
        ax.plot(i, orc, "^", ms=10, color=c, mec=SURFACE, mew=1.5, zorder=4)
        ax.plot(i, mean_, "_", ms=14, mew=2.5, color=c, zorder=4)
        ax.plot(i, sc, "o", ms=10, color=c, mec=SURFACE, mew=2, zorder=5)
        ax.text(i + 0.15, sc, f"{sc:.1f}", va="center", fontsize=9.5, color=INK)
        ax.text(i + 0.15, orc, f"{orc:.1f}", va="center", fontsize=9, color=INK2)
        ax.text(i, 56.0, f"regret\n{orc - sc + 1e-9:.1f}", ha="center", va="bottom", fontsize=9.5,
                color=INK if c != D1 or name.startswith("D1-K100") else INK2,
                fontweight="bold" if name.startswith("D1-K100") else "normal")
        ticks.append(i)
        labels.append(name)
    ax.axvspan(-0.5, 9.5, color="#f4f3f0", zorder=0)
    ax.text(4.5, 101.6, "1 % real images: regret 13.5–15.8, whatever raster is added", ha="center", fontsize=11,
            color=INK)
    ax.text(11.5, 101.6, "more real images", ha="center", fontsize=11, color=INK)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, len(POOL) - 0.4)
    ax.set_ylim(55, 103.5)
    ax.set_ylabel("navtest v1 PDMS (x100)")
    ax.set_title("Result 2: external raster lifts the best candidate, not the pick")
    handles = [
        Line2D([], [], lw=0, marker="^", ms=10, color=INK2, mec=SURFACE, label="oracle@64 (best of 64)"),
        Line2D([], [], lw=0, marker="o", ms=10, color=INK2, mec=SURFACE, label="scorer pick (reported PDMS)"),
        Line2D([], [], lw=0, marker="_", ms=14, mew=2.5, color=INK2, label="mean of 64"),
        Line2D([], [], lw=6, alpha=0.25, color=INK2, label="regret = oracle − pick"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    base.footnote(fig, "navtest v1, x100, single runs (s0 / s1 = training seed; the seed also changes the 1 % split). "
                       "+ nuSc / + WOMD = 23k rows rendered to the NAVSIM rig; + both = 11.5k each; all K1 runs 2,520 steps "
                       "except D1-K1 (~720).")
    fig.savefig(out / "mtg1106_fig_result2.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Ablation: D1-R1 training-val curves
R1C = "#c43c6c"


def fig_ablation(out):
    from parse_val_curves import parse
    q = base.ROOT / "qsub_outputs"
    p100 = {0: parse(q / "rap-e2a-p100.o9060304")[0]}
    p100.update(parse(q / "rap-e2a-p100.o9060312"))
    curves = [  # label, colour, lw, curve
        ("D1-K100  (100 % real)", "#3d3c3a", 1.8, p100),
        ("D1-R1  (1 % real + 99 % labeled raster)", R1C, 2.8, parse(q / "rap-r2r.o9266260")),
        ("K1 + WOMD 23k  (1 % real + unlabeled raster)", WOMD, 1.6, parse(q / "rap-dk.o9211333")),
        ("D1-K1  (1 % real)", D1, 1.6, parse(q / "rap-dk.o9062116")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, key, title, ylim in ((axes[0], "val/best_score", "Generation: best of 64 (val)", (0.80, 1.0)),
                                 (axes[1], "val/score", "Selection: the scorer's pick (val)", (0.70, 0.94))):
        for label, c, lw, cur in curves:
            ep = sorted(e for e in cur if key in cur[e])
            ax.plot(ep, [cur[e][key] for e in ep], color=c, lw=lw, zorder=3 if c == R1C else 2)
        ax.set_xlim(0, 19)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.set_xlabel("epoch")
        ax.set_title(title)
    r1 = curves[1][3]
    last = max(r1)
    axes[1].annotate(f"D1-R1 flat at ~0.83 since epoch 3\n(epoch {last} of 20, still training)",
                     xy=(last, r1[last]["val/score"]), xytext=(8.5, 0.735), fontsize=10.5, color=R1C,
                     arrowprops=dict(arrowstyle="->", color=R1C, lw=1))
    axes[1].text(10.5, 0.925, "D1-K100 keeps rising → 0.915", fontsize=10.5, color="#3d3c3a")
    axes[0].text(6, 0.83, "D1-R1 ≈ D1-K100 ≈ + WOMD: 0.97–0.98", fontsize=10.5, color=INK2)
    axes[0].set_ylabel("PDM score on training-val")
    handles = [Line2D([], [], color=c, lw=lw, label=lab) for lab, c, lw, _ in curves]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    base.footnote(fig, "Training-val = 214 NAVSIM val logs, real images; one seed; preliminary — navtest numbers come "
                       "after training. D1-R1 = RAP's real-ratio protocol: same 85,016 scenes and steps as D1-K100, "
                       "1 % keep real images, 99 % raster only (with PDM labels). Alignment on.")
    fig.savefig(out / "mtg1106_fig_ablation.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=base.ROOT / "docs/slides/assets")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in (fig_grid, fig_shot, fig_result1, fig_result2, fig_ablation):
        f(a.out)
    print("wrote mtg1106_fig_{grid,shot_curve,shot_curve_navhard,result1,result2,ablation}.png to", a.out)

"""Exp-0 / T1 step 8 figures from run_t1_probe_v1.py outputs (per-token CSV + _arrays.npz).

Writes eval/t1_results/figs/t1p_<tag>/*.webp (inside RAP/, synced back; png is not):
  1 dose_response    ΔPDMS vs vertical content shift (feature rows) for img / crop / feat / img_oob /
                     fix_all, pitch presets at their image-centre shift
  2 sca_waypoints    F0 SCA attention mass per waypoint, refiner round 1 (same initial proposals for
                     every variant): out / black fill / padding / content
  3 ladder           cosine to raw from image features through the 4 refiner rounds, plus ΔPDMS
  4 cos_heatmap      F0 28 x 48 cosine of real shifted vs moved crop features (FPN, ViT)
  5 overview         Δ img vs Δ img_oob for every preset

    python tools/plot_t1_probe.py --tag official
"""
import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
from summarize_t1 import EXP, SUBSET, boot_ci, latest_csv, load, token_logs  # noqa: E402
from rap_eval_agents.rig_transform import feature_shift, get_spec  # noqa: E402

VARIANTS = {  # name: (label, colour, linestyle)
    "img": ("img (= T1)", "#d62728", "-"),
    "fix_all": ("fix_all (raw candidates)", "#9467bd", ":"),
    "feat": ("feat (equivariant backbone)", "#ff7f0e", "--"),
    "img_oob": ("img_oob (fill = outside)", "#1f77b4", "-"),
    "crop": ("crop (content control)", "#2ca02c", "-"),
}
PITCH_AS_PPY = {"pitch_up_m10": "ppy_m272", "pitch_up_m5": "ppy_m135", "pitch_up_p5": "ppy_p135"}
CATS = [("out of image", "#bdbdbd"), ("black fill", "#252525"), ("padding row", "#d9b26f"), ("content", "#41ab5d")]
WAYPOINT_S = np.arange(1, 9) * 0.5


def save(fig, path: Path):
    """webp via Pillow (png / jpg are not synced); falls back to matplotlib's own pdf."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    try:
        from PIL import Image
        Image.open(io.BytesIO(buf.getvalue())).save(path, "WEBP", quality=90)
    except Exception as e:
        path = path.with_suffix(".pdf")
        print(f"webp failed ({e!r}); writing {path}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)


class Runs:
    def __init__(self, tag, exp_root, subset_csv, n_boot):
        self.logs, self.n_boot, self.rng = token_logs(subset_csv), n_boot, np.random.default_rng(0)
        self.df, self.npz = {}, {}
        prefix = f"t1probe_{tag}_"
        for d in sorted(exp_root.glob(prefix + "*")):
            csv = latest_csv(d) if d.is_dir() else None
            if csv is None:
                continue
            name = d.name[len(prefix):]
            df = load(csv)
            self.df[name] = df[df["valid"].astype(bool)]
            npz = csv.with_name(csv.stem + "_arrays.npz")
            if npz.exists():
                self.npz[name] = np.load(npz)
        if not self.df:
            raise SystemExit(f"no {prefix}* runs under {exp_root}")

    def delta(self, preset, v):
        """(mean, lo, hi) of ΔPDMS points, log-clustered bootstrap; None if missing."""
        df = self.df.get(preset)
        if df is None or f"pdms_{v}" not in df:
            return None
        m, lo, hi = boot_ci(df[f"pdms_{v}"] - df["pdms_raw"], self.logs, self.n_boot, self.rng)
        return 100 * m, 100 * lo, 100 * hi


def rows_of(preset):
    s = feature_shift(get_spec(preset))
    return None if s is None or s[1] != 0 else s[0]


def fig_dose_response(R, out):
    vert = sorted((p for p in R.df if rows_of(p) is not None), key=rows_of)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, (label, col, ls) in VARIANTS.items():
        pts = [(rows_of(p), R.delta(p, v), float(rows_of(p)).is_integer()) for p in vert]
        pts = [(x, d, i) for x, d, i in pts if d is not None]
        if not pts:
            continue
        xs = np.array([x for x, _, _ in pts])
        m = np.array([d[0] for _, d, _ in pts])
        err = np.array([[d[0] - d[1], d[2] - d[0]] for _, d, _ in pts]).T
        ax.plot(xs, m, ls, color=col, label=label, lw=1.6)
        for integer, mk in ((True, "o"), (False, "s")):
            sel = np.array([i == integer for _, _, i in pts])
            if sel.any():
                ax.errorbar(xs[sel], m[sel], yerr=err[:, sel], fmt=mk, color=col, ms=5, capsize=2,
                            mfc=col if integer else "white")
    for p, ppy in PITCH_AS_PPY.items():
        for v, mk in (("img", "*"), ("img_oob", "*")):
            d = R.delta(p, v) if p in R.df else None
            if d is not None:
                ax.plot(rows_of(ppy), d[0], mk, color=VARIANTS[v][1], ms=13, mec="k", mew=0.6)
    ax.plot([], [], "*", color="#999", ms=13, mec="k", mew=0.6, label="pitch −10 / −5 / +5° (img, img_oob)")
    ax.plot([], [], "o", color="#999", label="whole feature rows")
    ax.plot([], [], "s", color="#999", mfc="white", label="fractional (T1 ppy_*)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.4, ls=":")
    ax.set_xlabel("content shift on the feature map (rows; − = up)   1 row = 40 px @1080 = 16 px @432")
    ax.set_ylabel("ΔPDMS vs raw (points)")
    ax.set_title("T1: pure vertical translation, exact calibration (official RAP v1, 1,501 tokens, 95% CI)")
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    save(fig, out / "dose_response.webp")


def fig_sca_waypoints(R, out):
    stacked = [p for p in ("ppy_m7r", "pitch_up_m10") if p in R.npz and "sca_img_oob" in R.npz[p]]
    vert = sorted((p for p in R.npz if rows_of(p) is not None and "sca_img" in R.npz[p]), key=rows_of)
    fig, axes = plt.subplots(1, len(stacked) + 1, figsize=(5.2 * (len(stacked) + 1), 4.3))
    axes = np.atleast_1d(axes)
    vs = ("raw", "img", "img_oob")
    for ax, p in zip(axes, stacked):
        w = 0.27
        for j, v in enumerate(vs):
            a = np.nanmean(R.npz[p][f"sca_{v}"][:, 0, :8, 1:], axis=0)  # (8, 4) round 1
            bottom = np.zeros(8)
            for c, (cname, col) in enumerate(CATS):
                if c == 0:  # out of image (~0.8-0.95) would flatten the rest; it is 1 - bar height
                    continue
                ax.bar(WAYPOINT_S + (j - 1) * w * 0.5, a[:, c], w * 0.5, bottom=bottom, color=col,
                       edgecolor="white", lw=0.3, label=cname if j == 0 else None)
                bottom += a[:, c]
        ax.set_xticks(WAYPOINT_S)
        ax.set_xlabel("waypoint (s)   bars left to right: raw, img, img_oob")
        ax.set_ylabel("F0 attention mass inside the image (rest: outside)")
        ax.set_title(f"{p}: F0 SCA sampling, refiner round 1")
    axes[0].legend(fontsize=7, loc="lower right")
    ax = axes[-1]
    cmap = plt.get_cmap("coolwarm")
    for p in vert:
        a = np.nanmean(R.npz[p]["sca_img"][:, 0, :8, :], axis=0)
        r = rows_of(p)
        ax.plot(WAYPOINT_S, a[:, 2] + a[:, 3], "o-" if float(r).is_integer() else "s--", ms=4,
                color=cmap(0.5 + r / 16), label=f"{p} ({r:+.2f} rows)")
    a = np.nanmean(R.npz[vert[0]]["sca_raw"][:, 0, :8, :], axis=0)
    ax.plot(WAYPOINT_S, a[:, 2] + a[:, 3], "k-", lw=2, label="raw")
    ax.set_xlabel("waypoint (s)")
    ax.set_ylabel("F0 attention mass on black fill + padding")
    ax.set_title("img, refiner round 1")
    ax.legend(fontsize=7)
    save(fig, out / "sca_waypoints.webp")


def fig_ladder(R, out):
    presets = [p for p in ("ppy_m7r", "ppy_m3r", "pitch_up_m10") if p in R.df]
    fig, axes = plt.subplots(1, len(presets), figsize=(5.3 * len(presets), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    stages = ["ViT", "FPN", "query r1", "r2", "r3", "r4 (scorer in)"]
    for ax, p in zip(axes, presets):
        df = R.df[p]
        for v, (label, col, ls) in VARIANTS.items():
            if f"pdms_{v}" not in df:
                continue
            img_cols = {"img": ("vit_aligned_f0", "fpn_aligned_f0"), "crop": (None, "fpn_crop_vs_raw_f0")}.get(v, (None, None))
            y = [df[c].mean() if c and c in df else np.nan for c in img_cols]
            y += [df[f"qcos_r{r}_{v}"].mean() if f"qcos_r{r}_{v}" in df else np.nan for r in range(1, 5)]
            d = R.delta(p, v)
            ax.plot(range(len(stages)), y, ls, marker="o", ms=4, color=col,
                    label=f"{label}: ΔPDMS {d[0]:+.1f}" if d else label)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=20)
        ax.axvline(1.5, color="k", lw=0.4, ls=":")
        ax.set_title(p)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("cosine to raw (image: aligned, F0; queries: slot-matched)")
    fig.suptitle("Where the divergence starts: image features vs trajectory queries per refiner round", fontsize=10)
    save(fig, out / "ladder.webp")


def fig_cos_heatmap(R, out):
    presets = [p for p in ("ppy_m7r", "ppy_m3r", "ppy_m272", "pp_p5") if p in R.npz and "cosmap_fpn_f0" in R.npz[p]]
    if not presets:
        return
    fig, axes = plt.subplots(2, len(presets), figsize=(3.6 * len(presets), 4.4), squeeze=False)
    for j, p in enumerate(presets):
        for i, key in enumerate(("cosmap_fpn_f0", "cosmap_vit_f0")):
            with np.errstate(invalid="ignore"):
                m = np.nanmean(R.npz[p][key], axis=0)
            im = axes[i, j].imshow(m, vmin=0.8, vmax=1.0, cmap="viridis")
            axes[i, j].set_title(f"{p} {key.split('_')[1].upper()} (mean {np.nanmean(m):.3f})", fontsize=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([0, 13, 27])
    fig.colorbar(im, ax=axes, shrink=0.7, label="cos(real shifted, moved crop)")
    fig.suptitle("8b: F0 feature equivariance per position (white = not compared: fill / padding / partial patches)",
                 fontsize=9)
    save(fig, out / "cos_heatmap.webp")


def fig_overview(R, out):
    order = ["pitch_up_m10", "pitch_up_m5", "pitch_up_p5", "ppy_m7r", "ppy_m272", "ppy_m3r", "ppy_m135", "ppy_m1r",
             "ppy_p1r", "ppy_p3r", "ppy_p135", "pp_p5", "pp_m5", "focal_0.8", "focal_1.25", "drop_black_f0"]
    order = [p for p in order if p in R.df]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    y = np.arange(len(order))
    for k, (v, off) in enumerate((("img", -0.18), ("img_oob", 0.18))):
        ds = [R.delta(p, v) for p in order]
        m = np.array([d[0] if d else np.nan for d in ds])
        err = np.array([[d[0] - d[1], d[2] - d[0]] if d else [np.nan, np.nan] for d in ds]).T
        ax.barh(y + off, m, 0.34, xerr=err, color=VARIANTS[v][1], label=VARIANTS[v][0], capsize=2,
                error_kw={"lw": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("ΔPDMS vs raw (points, 95% CI)")
    ax.set_title("T1 drops with and without SCA samples on black fill / padding (img_oob)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, out / "overview.webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="official")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "t1_results" / "figs")
    args = ap.parse_args()
    R = Runs(args.tag, args.exp_root, args.subset_csv, args.n_boot)
    out = args.out_dir / f"t1p_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    print("presets:", ", ".join(sorted(R.df)))
    for fn in (fig_overview, fig_dose_response, fig_sca_waypoints, fig_ladder, fig_cos_heatmap):
        fn(R, out)


if __name__ == "__main__":
    main()

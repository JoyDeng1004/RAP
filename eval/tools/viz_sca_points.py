"""Exp-0 / T1 step 8: where RAP's spatial cross-attention samples F0, raw vs rig presets (refiner round 1).

Round 1 reads the images at the same initial proposals for every rig, so differences come from the
rig alone. Sampling locations = projected 3D anchors (4 heights x 4 box corners per waypoint) plus
learned deformable offsets, one per (head, anchor); weights = attention weights.

Writes to eval/t1_results/figs/sca_points/ (webp, synced back):
  density_<presets>.webp   attention-weighted sampling density on a canvas extending the image by half
                           its size on every side, per waypoint group; anchors (unweighted) for raw;
                           black-fill outline; mass shares below / above / beside / on fill / in image
  example_<preset>_<token>.webp   one token: F0 image, anchors (x) and samples (size = weight,
                           colour = waypoint time) of the raw-selected candidate, raw vs preset

Needs a GPU (RAPV2Agent). Tokens come from the probe CSV of the first preset:
  - density: --n-density random valid tokens
  - examples: the --n-examples tokens where img_oob recovers the most

    python tools/viz_sca_points.py --rap-root $RAP --agent-config hydra/rap_v2_official_agent.yaml \
        --presets ppy_m7r pitch_up_m10
"""
import argparse
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
sys.path.insert(0, str(HERE.parent / "v1"))
from summarize_t1 import EXP, latest_csv, load  # noqa: E402

SLOT_F0 = 1
W_PAD, H_PAD, H_IMG = 768, 448, 432
X_RANGE, Y_RANGE = (-W_PAD / 2, W_PAD * 1.5), (-H_PAD / 2, H_PAD * 1.5)
BIN = 8  # px
GROUPS = [("0.5–1.0 s", [0, 1]), ("1.5–2.5 s", [2, 3, 4]), ("3.0–4.0 s", [5, 6, 7])]


def save(fig, path: Path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    try:
        from PIL import Image
        Image.open(io.BytesIO(buf.getvalue())).save(path, "WEBP", quality=88)
    except Exception as e:
        path = path.with_suffix(".pdf")
        print(f"webp failed ({e!r}); writing {path}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("written:", path, flush=True)


def round1_f0(records, n_poses):
    """From record_sca records of one decode: F0 anchors and samples of refiner round 1.
    Returns dict q -> (anchors (D, 2) px, locs (layers*heads*P, 2) px, weights (same,)); weights are
    averaged over layers and divided by heads, so one query's weights sum to 1."""
    ps, attn = records[0], [r for r in records[1:3] if r[0] == "attn"]
    assert ps[0] == "ps" and len(attn) == 2 and (len(records) < 4 or records[3][0] == "ps"), [r[0] for r in records[:4]]
    ref, mask = ps[1][SLOT_F0, 0].float().cpu().numpy(), ps[2][SLOT_F0, 0].cpu().numpy()  # (Q, D, 2), (Q, D)
    idx = np.nonzero(mask.sum(-1))[0]
    out = {}
    scale = np.array([W_PAD, H_PAD])
    for j, q in enumerate(idx):
        locs, ws = [], []
        for _, loc, w in attn:
            l = loc[SLOT_F0, j].float().cpu().numpy().reshape(-1, 2)  # (heads*levels*P, 2)
            ww = w[SLOT_F0, j].float().cpu().numpy().reshape(-1) / w.shape[2] / len(attn)
            locs.append(l)
            ws.append(ww)
        out[int(q)] = (ref[q] * scale, np.concatenate(locs) * scale, np.concatenate(ws))
    return out


def side_shares(locs, ws, fill):
    """Mass shares: below / above / left+right of the padded image, on black fill, padding row, image content."""
    x, y = locs[:, 0], locs[:, 1]
    below, above = y >= H_PAD, y < 0
    side = ~below & ~above & ((x < 0) | (x >= W_PAD))
    inside = ~below & ~above & ~side
    xi, yi = np.clip(x, 0, W_PAD - 1).astype(int), np.clip(y, 0, H_PAD - 1).astype(int)
    pad = inside & (y >= H_IMG)
    on_fill = inside & ~pad & fill[yi, xi]
    content = inside & ~pad & ~on_fill
    tot = ws.sum()
    return {k: float(ws[m].sum() / tot) for k, m in
            (("below", below), ("above", above), ("side", side), ("fill", on_fill), ("pad", pad), ("content", content))}


def draw_frame(ax, fill=None):
    ax.plot([0, W_PAD, W_PAD, 0, 0], [0, 0, H_PAD, H_PAD, 0], "w-", lw=1.0)
    ax.plot([0, W_PAD], [H_IMG, H_IMG], "w:", lw=0.8)
    if fill is not None and fill.any():
        ax.contour(np.arange(W_PAD), np.arange(H_PAD), fill.astype(float), levels=[0.5], colors="red", linewidths=0.9)
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(Y_RANGE[1], Y_RANGE[0])
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True)
    ap.add_argument("--presets", nargs="+", default=["ppy_m7r", "pitch_up_m10"])
    ap.add_argument("--probe-tag", default="official")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--n-density", type=int, default=48)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "t1_results" / "figs" / "sca_points")
    args = ap.parse_args()

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from run_pdm_score_v1 import load_split
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents.rig_transform import apply_rig, get_spec

    csv = latest_csv(args.exp_root / f"t1probe_{args.probe_tag}_{args.presets[0]}")
    df = load(csv)
    df = df[df["valid"].astype(bool)]
    rng = np.random.default_rng(0)
    dens_tokens = list(rng.choice(df.index.to_numpy(), size=min(args.n_density, len(df)), replace=False))
    gain = (df["pdms_img_oob"] - df["pdms_img"]).sort_values(ascending=False)
    ex_tokens = list(gain.index[: args.n_examples])
    print("examples (img_oob - img):", {t: round(float(gain[t]), 3) for t in ex_tokens}, flush=True)

    agent = instantiate(OmegaConf.load(args.agent_config))
    agent.initialize()
    model = agent._model
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    split, scene_filter = load_split(args.rap_root, "navtest")
    scene_filter.tokens = sorted(set(dens_tokens) | set(ex_tokens))
    loader = SceneLoader(data_path=data_root / "navsim_logs" / split, sensor_blobs_path=data_root / "sensor_blobs" / split,
                         scene_filter=scene_filter, sensor_config=agent.get_sensor_config())

    def run(token, preset):
        ai = loader.get_agent_input_from_token(token)
        apply_rig(ai.cameras[-1], get_spec(preset))
        f = agent._features(ai)
        with torch.no_grad():
            imf, _ = rp.encode(model, f)
            with rp.record_sca(model) as rec:
                o = rp.decode(model, f, imf)
        n_poses = o["proposals"].shape[2]
        return f, round1_f0(rec, n_poses), n_poses, int(o["score"][0].argmax())

    variants = ["raw"] + args.presets
    nbx, nby = int((X_RANGE[1] - X_RANGE[0]) / BIN), int((Y_RANGE[1] - Y_RANGE[0]) / BIN)
    H = {(v, g): np.zeros((nby, nbx)) for v in variants for g in range(len(GROUPS))}
    A = {g: np.zeros((nby, nbx)) for g in range(len(GROUPS))}
    S = {(v, g): [] for v in variants for g in range(len(GROUPS))}
    fills = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, token in enumerate(dict.fromkeys(dens_tokens + ex_tokens)):
        per = {v: run(token, v) for v in variants}
        for v, (f, r1, n_poses, _) in per.items():
            fill = rp.black_mask(f, SLOT_F0).cpu().numpy()
            fills.setdefault(v, fill)
            if token not in dens_tokens:
                continue
            for g, (_, ts) in enumerate(GROUPS):
                qs = [q for q in r1 if q % n_poses in ts]
                if not qs:
                    continue
                locs = np.concatenate([r1[q][1] for q in qs])
                ws = np.concatenate([r1[q][2] for q in qs])
                H[v, g] += np.histogram2d(locs[:, 1], locs[:, 0], bins=(nby, nbx), range=(Y_RANGE, X_RANGE), weights=ws)[0]
                S[v, g].append((side_shares(locs, ws, fill), ws.sum()))
                if v == "raw":
                    anc = np.concatenate([r1[q][0] for q in qs])
                    anc = anc[np.isfinite(anc).all(1)]
                    A[g] += np.histogram2d(anc[:, 1], anc[:, 0], bins=(nby, nbx), range=(Y_RANGE, X_RANGE))[0]
        if token in ex_tokens:
            fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 3.6))
            sel = per["raw"][3]
            for ax, v in zip(axes, variants):
                f, r1, n_poses, _ = per[v]
                img = f["camera_feature"][0, SLOT_F0].float().cpu().numpy().transpose(1, 2, 0)
                img = img * np.array(rp.RAP_STD) + np.array(rp.RAP_MEAN)
                ax.imshow(np.clip(img, 0, 255).astype(np.uint8), extent=(0, W_PAD, H_PAD, 0))
                for t in range(min(8, n_poses)):
                    q = sel * n_poses + t
                    if q not in r1:
                        continue
                    anc, locs, ws = r1[q]
                    col = plt.get_cmap("plasma")(t / 7)
                    ax.scatter(locs[:, 0], locs[:, 1], s=600 * ws, color=col, alpha=0.6, lw=0)
                    ok = np.isfinite(anc).all(1) & (np.abs(anc) < 1e4).all(1)
                    ax.scatter(anc[ok, 0], anc[ok, 1], marker="x", s=14, color=col, lw=0.9)
                draw_frame(ax, rp.black_mask(f, SLOT_F0).cpu().numpy())
                sh = side_shares(np.concatenate([r1[q][1] for q in r1 if q // n_poses == sel and q % n_poses < 8] or [np.zeros((1, 2))]),
                                 np.concatenate([r1[q][2] for q in r1 if q // n_poses == sel and q % n_poses < 8] or [np.zeros(1)]),
                                 rp.black_mask(f, SLOT_F0).cpu().numpy())
                ax.set_title(f"{v}: below {sh['below']:.2f} | fill {sh['fill']:.2f} | content {sh['content']:.2f}", fontsize=8)
            fig.suptitle(f"token {token}: F0 SCA round 1, raw-selected candidate #{sel} "
                         f"(x = anchors, dots = samples, size = weight, colour = 0.5→4 s; red = black fill; dotted = padding)",
                         fontsize=8)
            save(fig, args.out_dir / f"example_{args.presets[0]}_{token}.webp")
        print(f"{i + 1} tokens done", flush=True)

    cols = ["raw anchors"] + [f"{v} samples" for v in variants]
    fig, axes = plt.subplots(len(GROUPS), len(cols), figsize=(4.4 * len(cols), 2.9 * len(GROUPS)), squeeze=False)
    vmax = max(h.max() for h in H.values())
    for g, (gname, _) in enumerate(GROUPS):
        ax = axes[g, 0]
        ax.imshow(A[g] + 1e-9, extent=(*X_RANGE, Y_RANGE[1], Y_RANGE[0]), cmap="magma",
                  norm=LogNorm(vmin=max(A[g].max() * 1e-4, 1e-3), vmax=A[g].max()), aspect="auto")
        draw_frame(ax)
        ax.set_ylabel(gname)
        for c, v in enumerate(variants, start=1):
            ax = axes[g, c]
            ax.imshow(H[v, g] + 1e-12, extent=(*X_RANGE, Y_RANGE[1], Y_RANGE[0]), cmap="magma",
                      norm=LogNorm(vmin=vmax * 1e-5, vmax=vmax), aspect="auto")
            draw_frame(ax, fills.get(v))
            tot = sum(w for _, w in S[v, g])
            sh = {k: sum(s[k] * w for s, w in S[v, g]) / tot for k in S[v, g][0][0]} if tot else {}
            if sh:
                ax.set_title(f"below {sh['below']:.2f} above {sh['above']:.2f} side {sh['side']:.2f}\n"
                             f"fill {sh['fill']:.2f} pad {sh['pad']:.2f} content {sh['content']:.2f}", fontsize=7)
        if g == 0:
            for c, name in enumerate(cols):
                axes[0, c].text(0.5, 1.32, name, transform=axes[0, c].transAxes, ha="center", fontsize=10)
    fig.suptitle(f"F0 SCA round 1 over {len(dens_tokens)} tokens: attention-weighted sampling density "
                 "(white = padded image 768x448, dotted = padding row, red = black fill; canvas extends half an image each side)",
                 fontsize=9, y=1.02)
    save(fig, args.out_dir / f"density_{'_'.join(args.presets)}.webp")


if __name__ == "__main__":
    main()

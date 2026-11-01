"""K0 real-vs-raster diagnosis, V1: per-patch feature cosine between the real input and the official raster
(c1, same calibration) on the tokens where only the real input picks a 0-PDMS trajectory, against
control tokens where both inputs pick well.

Tokens (from summarize_k0_probe.py's <tag>_tail_tokens.csv and the probe run of the same checkpoint):
  collision  real pick fails no_collision (at-fault collision), c1 pick >= 0.5, worst regret first
  offroad    real pick fails drivable_area only, c1 pick >= 0.5
  control    real and c1 picks both >= --control-min, random
  hard_coll  difficulty-matched control of collision: real and c1 picks >= --control-min, nearest share of
             the real input's 64 candidates failing no_collision (probe npz sub_no_collision_real), greedy
             without replacement, hardest failure token first
  hard_off   the same for offroad on the drivable_area share
  at most one token per log in each group. --match-only stops after printing the matching (CPU only).

Per token, forward only (real and c1 inputs, as run_t2_probe_v1.py with preset ident):
  cosine of the FPN features (camera / level embeddings stripped) and of the raw DINOv3 patch tokens,
  per 16 px patch and camera; the 64 candidates of the real input scored with their PDMS gates; the
  picks of both inputs. The picks are checked against the probe CSV (idx_real / idx_c1).

Writes to eval/k0_results/figs/<model>/ (webp, synced back):
  <group>_<rank>_<token>.webp  rows F0 / L0 / R0 / B0: real | raster | FPN cos | ViT cos; right: BEV
                               with map, boxes, the real input's 64 candidates (red = collision,
                               orange = off-road), real pick (red) and c1 pick (green)
  mean_cos.webp                group-mean FPN cos maps per camera, and failure − matched control
  summary.md / tokens.csv      mean cos per group, camera, drawn vs background patches; failure vs
                               control comparisons (matched and random) with permutation p-values

Needs a GPU (RAPV2Agent):

    python tools/viz_k0_failures.py --rap-root $RAP --agent-config hydra/rap_v2_agent.yaml \
        --scenario-dir <stage A dir> --metric-cache <subset cache> --tag k0b --model D2-K0-scene-mix_last
"""
import argparse
import io
import lzma
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
sys.path.insert(0, str(HERE.parent / "v1"))
from summarize_t1 import EXP, SUBSET, latest_csv, load, token_logs  # noqa: E402

SHOW = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")  # display order; RAP slots are t2_render.CAMS
GROUPS = ("collision", "offroad", "control", "hard_coll", "hard_off")
# (failure group, control group, pool share column the pair is matched on)
PAIRS = (("collision", "hard_coll", "pool_collision"), ("offroad", "hard_off", "pool_offroad"),
         ("collision", "control", "pool_collision"), ("offroad", "control", "pool_offroad"))
N_PERM = 20000
ZERO = 1e-6
PATCH = 16
VMIN = 0.3


def save(fig, path: Path):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    try:
        from PIL import Image
        Image.open(io.BytesIO(buf.getvalue())).save(path, "WEBP", quality=85)
    except Exception as e:
        path = path.with_suffix(".pdf")
        print(f"webp failed ({e!r}); writing {path}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("written:", path, flush=True)


def one_per_log(df: pd.DataFrame, n: int) -> list:
    return list(df[~df["log"].duplicated()].index[:n])


def pool_shares(probe_csv: Path) -> pd.DataFrame:
    """Per token, share of the real input's 64 candidates failing each PDMS gate (probe npz)."""
    a = np.load(probe_csv.with_name(f"{probe_csv.stem}_arrays.npz"))
    if "sub_no_collision_real" not in a:
        raise SystemExit(f"{probe_csv}: npz has no sub-metrics; rerun the probe (run_t2_probe_v1 from 10/29 on)")
    return pd.DataFrame({"pool_collision": (a["sub_no_collision_real"] <= ZERO).mean(1),
                         "pool_offroad": (a["sub_drivable_area_real"] <= ZERO).mean(1)},
                        index=pd.Index(a["token"].astype(str)))


def match(target: pd.Series, pool: pd.DataFrame, col: str) -> list:
    """Greedy nearest neighbour on ``col`` without replacement, hardest target first, one token per log."""
    used, logs_used, out = set(), set(), {}
    for tok, v in target.sort_values(ascending=False).items():
        cand = pool[~pool.index.isin(used) & ~pool["log"].isin(logs_used)]
        if cand.empty:
            break
        best = (cand[col] - v).abs().idxmin()
        used.add(best)
        logs_used.add(pool.at[best, "log"])
        out[tok] = best
    return [out[t] for t in target.index if t in out]  # in the target's order


def pick_tokens(tail_csv: Path, probe: pd.DataFrame, shares: pd.DataFrame, logs: pd.Series, model: str,
                n_gate: int, n_ctrl: int, control_min: float, seed: int) -> dict:
    t = pd.read_csv(tail_csv)
    t = t[t["model"] == model].set_index("token")
    if t.empty:
        raise SystemExit(f"no real-only failures of {model} in {tail_csv}")
    coll = t[t["pick_no_collision_real"] <= ZERO]
    off = t[(t["pick_drivable_area_real"] <= ZERO) & (t["pick_no_collision_real"] > ZERO)]
    ok = probe[(probe["pdms_real"] >= control_min) & (probe["pdms_c1"] >= control_min)
               & ~probe.index.isin(t.index)].copy()
    ok["log"] = logs.reindex(ok.index)
    ok = ok.join(shares).sample(frac=1.0, random_state=seed)
    groups = {"collision": one_per_log(coll, n_gate), "offroad": one_per_log(off, n_gate),
              "control": one_per_log(ok, n_ctrl)}
    rest = ok[~ok.index.isin(groups["control"])]
    groups["hard_coll"] = match(shares.loc[groups["collision"], "pool_collision"], rest, "pool_collision")
    rest = rest[~rest.index.isin(groups["hard_coll"])]
    groups["hard_off"] = match(shares.loc[groups["offroad"], "pool_offroad"], rest, "pool_offroad")
    return groups


def perm_test(a: np.ndarray, b: np.ndarray, rng, n: int = N_PERM) -> tuple:
    """Difference of means a − b and its two-sided permutation p-value."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan"), float("nan")
    obs, x = a.mean() - b.mean(), np.concatenate([a, b])
    idx = np.argsort(rng.random((n, len(x))), axis=1)
    d = x[idx[:, :len(a)]].mean(1) - x[idx[:, len(a):]].mean(1)
    return float(obs), float((np.sum(np.abs(d) >= abs(obs) - 1e-12) + 1) / (n + 1))


def to_img(f, slot: int, rp) -> np.ndarray:
    img = f["camera_feature"][0, slot].float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * np.array(rp.RAP_STD) + np.array(rp.RAP_MEAN), 0, 255).astype(np.uint8)


def drawn_patches(f, slot: int, rp) -> np.ndarray:
    """(28, 48) patches of a raster input holding any non-black pixel."""
    black = rp.black_mask(f, slot).cpu().numpy()
    return (~black).reshape(rp.FEAT_H, PATCH, rp.FEAT_W, PATCH).any((1, 3))


def to_ego(xy_world: np.ndarray, heading: float) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return xy_world @ np.array([[c, -s], [s, c]])  # R_ego^T applied to row vectors


def draw_bev(ax, scenario: dict, cand: np.ndarray, gates: dict, i_real: int, cand_c1: np.ndarray, i_c1: int):
    h = scenario["ego_heading"]
    lim = dict(x=(-12, 60), y=(-25, 25))

    def plot(xy, **kw):  # ego x forward -> up, y left -> left
        ax.plot(-xy[:, 1], xy[:, 0], **kw)

    for feat in scenario["map_features"].values():
        if "polygon" in feat and len(feat["polygon"]) > 2:
            p = to_ego(np.asarray(feat["polygon"], np.float64)[:, :2], h)
            ax.fill(-p[:, 1], p[:, 0], color="0.9", lw=0, zorder=0)
        if "polyline" in feat and len(feat["polyline"]) > 1:
            plot(to_ego(np.asarray(feat["polyline"], np.float64)[:, :2], h), color="0.7", lw=0.5, zorder=1)
    for b in np.asarray(scenario["anns"]["gt_boxes_world"], np.float64).reshape(-1, 7):
        c = to_ego(b[None, :2], h)[0]
        yaw = b[6] - h
        dx, dy = b[3] / 2, b[4] / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        xy = corners @ rot.T + c
        ax.fill(-xy[:, 1], xy[:, 0], color="tab:blue", alpha=0.35, lw=1.0, ec="tab:blue", zorder=2)
    ego = np.array([[4.0, 1.0], [4.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [4.0, 1.0]])
    plot(ego, color="k", lw=1.2, zorder=3)
    for j, tr_ in enumerate(cand):
        xy = np.vstack([[0, 0], tr_[:, :2]])
        col = "tab:red" if gates["no_collision"][j] <= ZERO else (
            "tab:orange" if gates["drivable_area"][j] <= ZERO else "0.55")
        plot(xy, color=col, lw=0.5, alpha=0.6, zorder=4)
    plot(np.vstack([[0, 0], cand[i_real, :, :2]]), color="red", lw=2.4, zorder=6, label="real pick")
    plot(np.vstack([[0, 0], cand_c1[i_c1, :, :2]]), color="limegreen", lw=2.4, ls="--", zorder=6, label="c1 pick")
    ax.set_xlim(-lim["y"][1], -lim["y"][0])
    ax.set_ylim(*lim["x"])
    ax.set_aspect("equal")
    ax.set_xlabel("← left   lateral (m)   right →", fontsize=8)
    ax.set_ylabel("forward (m)", fontsize=8)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("BEV, real candidates: red collision / orange off-road", fontsize=8)


def plot_token(imgs: dict, cos: dict, row: dict, scenario: dict, cand_r, sub_r, i_r, cand_c, i_c, title: str,
               ext, path: Path):
    """Rows F0 / L0 / R0 / B0: real | raster | FPN cos | ViT cos (over the grey real image); right: BEV."""
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.04, hspace=0.12)
    im = None
    for i, c in enumerate(SHOW):
        img_r, img_c = imgs[c]
        gray = img_r.mean(-1)
        panels = [(img_r, None, "real"), (img_c, None, "raster c1"),
                  (gray, cos[c]["fpn"], f"FPN cos {row[f'fpn_{c}']:.2f}"),
                  (gray, cos[c]["vit"], f"ViT cos {row[f'vit_{c}']:.2f}")]
        for j, (base, m, name) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(base, cmap="gray" if base.ndim == 2 else None, extent=ext)
            if m is not None:
                im = ax.imshow(m, cmap="magma", vmin=VMIN, vmax=1.0, alpha=0.75, extent=ext, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{c[4:].upper()} {name}", fontsize=8)
    draw_bev(fig.add_subplot(gs[:, 4]), scenario, cand_r, sub_r, i_r, cand_c, i_c)
    fig.colorbar(im, ax=fig.axes[:16], shrink=0.5, pad=0.01, label="cos(real, raster)")
    fig.suptitle(title, fontsize=10)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="the RAPV2Agent yaml the probe ran with")
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="summarize_k0_probe tag whose tail_tokens.csv to read")
    ap.add_argument("--model", required=True, help="model name in that CSV, e.g. D2-K0-scene-mix_last")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n-per-gate", type=int, default=4)
    ap.add_argument("--n-controls", type=int, default=4)
    ap.add_argument("--n-mean", type=int, default=0, help="extra tokens per group for mean_cos only (0: none)")
    ap.add_argument("--control-min", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--match-only", action="store_true", help="print the token groups and their matching, no GPU")
    args = ap.parse_args()
    out_dir = args.out_dir or HERE.parent / "k0_results" / "figs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_csv = latest_csv(args.exp_root / f"k0probe_{args.tag}_{args.model}")
    if probe_csv is None:
        raise SystemExit(f"no probe run k0probe_{args.tag}_{args.model} under {args.exp_root}")
    probe = load(probe_csv)
    probe = probe[probe["valid"].astype(bool)]
    logs = token_logs(args.subset_csv)
    shares = pool_shares(probe_csv)
    n_plot = {g: args.n_per_gate if g in ("collision", "offroad") else args.n_controls for g in GROUPS}
    n_all = {g: max(n, args.n_mean) for g, n in n_plot.items()}
    groups = pick_tokens(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv", probe, shares, logs,
                         args.model, n_all["collision"], n_all["control"], args.control_min, args.seed)
    print({g: len(v) for g, v in groups.items()}, "probe:", probe_csv, flush=True)
    for fail, ctrl, col in PAIRS:
        a, b = shares.loc[groups[fail], col], shares.loc[groups[ctrl], col]
        print(f"match {fail:9s} vs {ctrl:9s} on {col}: mean {a.mean():.3f} vs {b.mean():.3f}, "
              f"median {a.median():.3f} vs {b.median():.3f} (n {len(a)} / {len(b)})", flush=True)
    if args.match_only:
        return

    import torch
    from hydra.utils import instantiate
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from omegaconf import OmegaConf
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from oracle64_v1 import score_candidates
    from run_pdm_score_v1 import load_split
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents import t2_render as tr

    agent = instantiate(OmegaConf.load(args.agent_config))
    agent.initialize()
    model = agent._model
    rmod = tr.renderer_module(args.rap_root)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())
    mc_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths

    tokens = list(dict.fromkeys(t for g in GROUPS for t in groups[g]))
    tok_logs = sorted({logs[t] for t in tokens})
    scenarios = {}
    for log in tok_logs:
        with open(args.scenario_dir / f"{log}.pkl", "rb") as f:
            scenarios.update(pickle.load(f))
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    split, scene_filter = load_split(args.rap_root, "navtest")
    scene_filter.log_names, scene_filter.tokens = tok_logs, tokens
    loader = SceneLoader(data_path=data_root / "navsim_logs" / split, sensor_blobs_path=data_root / "sensor_blobs" / split,
                         scene_filter=scene_filter, sensor_config=agent.get_sensor_config())

    def features(token, raster: bool):
        ai = loader.get_agent_input_from_token(token)  # _features edits the cameras: fresh input each time
        if raster:
            images, _ = tr.render(rmod, scenarios[token], tr.IDENT)
            for name in tr.CAMS:
                cam = getattr(ai.cameras[-1], name)
                cam.image = images[name].astype(np.float32)
                tr.update_model_camera(cam, tr.IDENT)
        return agent._features(ai)

    valid = torch.zeros(rp.FEAT_H, rp.FEAT_W, dtype=torch.bool)
    valid[: rp.CONTENT_H] = True
    rows, maps = [], {g: {c: [] for c in SHOW} for g in GROUPS}
    for g in GROUPS:
        for rank, token in enumerate(groups[g]):
            with lzma.open(mc_paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            with torch.no_grad():
                f_r, f_c = features(token, False), features(token, True)
                (imf_r, vit_r), (imf_c, vit_c) = rp.encode(model, f_r, want_vit=True), rp.encode(model, f_c, want_vit=True)
                o_r, o_c = rp.decode(model, f_r, imf_r), rp.decode(model, f_c, imf_c)
                s_r, s_c = rp.strip_embeds(model, imf_r[0]), rp.strip_embeds(model, imf_c[0])
                v = valid.to(s_r.device)
                cos = {c: {"fpn": rp.cos_map(s_r, s_c, v, tr.CAMS.index(c)).cpu().numpy(),
                           "vit": rp.cos_map(vit_r, vit_c, v, tr.CAMS.index(c)).cpu().numpy()} for c in SHOW}
            cand_r = o_r["proposals"][0, :, :8].float().cpu().numpy()
            cand_c = o_c["proposals"][0, :, :8].float().cpu().numpy()
            i_r, i_c = int(o_r["score"][0].argmax()), int(o_c["score"][0].argmax())
            pool_r, sub_r = score_candidates(metric_cache, cand_r, simulator, scorer, return_subscores=True)
            pool_c = score_candidates(metric_cache, cand_c, simulator, scorer)
            p = probe.loc[token]
            if (i_r, i_c) != (int(p["idx_real"]), int(p["idx_c1"])) or abs(pool_r[i_r] - p["pdms_real"]) > 1e-6:
                print(f"WARNING {token}: picks ({i_r}, {i_c}) pdms {pool_r[i_r]:.4f} vs probe "
                      f"({int(p['idx_real'])}, {int(p['idx_c1'])}) {p['pdms_real']:.4f}", flush=True)

            row = {"group": g, "token": token, "log": logs[token], "pdms_real": pool_r[i_r], "pdms_c1": pool_c[i_c],
                   "oracle_real": pool_r.max(), "pick_collision": sub_r["no_collision"][i_r] <= ZERO,
                   "pick_offroad": sub_r["drivable_area"][i_r] <= ZERO,
                   "pool_collision": float((sub_r["no_collision"] <= ZERO).mean()),
                   "pool_offroad": float((sub_r["drivable_area"] <= ZERO).mean())}
            for c in SHOW:
                slot = tr.CAMS.index(c)
                drawn = drawn_patches(f_c, slot, rp)
                for kind in ("fpn", "vit"):
                    m = cos[c][kind]
                    row[f"{kind}_{c}"] = float(np.nanmean(m))
                    row[f"{kind}_{c}_drawn"] = float(np.nanmean(np.where(drawn, m, np.nan)))
                    row[f"{kind}_{c}_bg"] = float(np.nanmean(np.where(~drawn, m, np.nan)))
                maps[g][c].append(cos[c]["fpn"])
            rows.append(row)
            if rank >= n_plot[g]:
                continue

            imgs = {c: (to_img(f_r, tr.CAMS.index(c), rp), to_img(f_c, tr.CAMS.index(c), rp)) for c in SHOW}
            title = (f"{args.model} | {g} #{rank} | {token} | {logs[token]}\n"
                     f"PDMS real {pool_r[i_r]:.2f} (collision {row['pick_collision']}, off-road "
                     f"{row['pick_offroad']}) vs c1 {pool_c[i_c]:.2f}; oracle {pool_r.max():.2f}; real pool "
                     f"collision {row['pool_collision']:.0%} / off-road {row['pool_offroad']:.0%}")
            plot_token(imgs, cos, row, scenarios[token], cand_r, sub_r, i_r, cand_c, i_c, title,
                       (0, rp.FEAT_W * PATCH, rp.FEAT_H * PATCH, 0), out_dir / f"{g}_{rank}_{token}.webp")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tokens.csv", index=False)

    diffs = PAIRS[:2]
    fig, axes = plt.subplots(len(GROUPS) + len(diffs), len(SHOW),
                             figsize=(4.2 * len(SHOW), 2.6 * (len(GROUPS) + len(diffs))))
    mean = {g: {c: np.nanmean(np.stack(maps[g][c]), 0) for c in SHOW} for g in GROUPS if maps[g]["cam_f0"]}
    im = imd = None
    for j, c in enumerate(SHOW):
        for i, g in enumerate(GROUPS):
            ax = axes[i, j]
            if g in mean:
                im = ax.imshow(mean[g][c], cmap="magma", vmin=VMIN, vmax=1.0, interpolation="nearest")
            ax.set_title(f"{c[4:].upper()} {g} (n={len(maps[g][c])})", fontsize=8)
        for i, (g, h, _) in enumerate(diffs):
            ax = axes[len(GROUPS) + i, j]
            if g in mean and h in mean:
                imd = ax.imshow(mean[g][c] - mean[h][c], cmap="RdBu", vmin=-0.2, vmax=0.2,
                                interpolation="nearest")
            ax.set_title(f"{c[4:].upper()} {g} − {h}", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes[: len(GROUPS)].ravel().tolist(), shrink=0.6, label="mean FPN cos")
    if imd is not None:
        fig.colorbar(imd, ax=axes[len(GROUPS):].ravel().tolist(), shrink=0.6, label="Δ cos")
    fig.suptitle(f"{args.model}: mean FPN cos(real, raster) per patch", fontsize=10)
    save(fig, out_dir / "mean_cos.webp")

    cols = [f"{k}_{c}{s}" for k in ("fpn", "vit") for c in SHOW for s in ("", "_drawn", "_bg")]
    agg = df.groupby("group")[cols].mean().reindex([g for g in GROUPS if g in set(df["group"])])
    lines = [f"# V1 {args.model} (tag {args.tag})", "",
             f"Tokens per group: {df['group'].value_counts().to_dict()}; probe `{probe_csv}`.",
             "Mean cos(real, raster) over content patches; drawn = patches with any raster pixel, bg = black in "
             "the raster.", ""]
    for kind in ("fpn", "vit"):
        lines += [f"## {kind.upper()}", "", "| group | " + " | ".join(
            f"{c[4:].upper()} all / drawn / bg" for c in SHOW) + " |", "|---|" + "---|" * len(SHOW)]
        for g, r in agg.iterrows():
            cells = [" / ".join(f"{r[f'{kind}_{c}{s}']:.3f}" for s in ("", "_drawn", "_bg")) for c in SHOW]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")

    rng = np.random.default_rng(args.seed)
    lines += ["## Failure vs control", "",
              f"Δ = failure − control mean cos, two-sided permutation p ({N_PERM} permutations), F0 / L0 / R0; "
              "B0 is black in the official raster and left out. Matched = hard_coll / hard_off. Many tests: read "
              "p < 0.01 as a lead, not a finding.", "",
              "| failure vs control | matched on: failure / control | feature | patches | F0 Δ (p) | L0 Δ (p) | R0 Δ (p) |",
              "|---|---|---|---|---|---|---|"]
    for fail, ctrl, col in PAIRS:
        a, b = df[df["group"] == fail], df[df["group"] == ctrl]
        if a.empty or b.empty:
            continue
        mq = f"{col} {a[col].mean():.2f} / {b[col].mean():.2f}"
        for kind in ("fpn", "vit"):
            for sfx, name in (("", "all"), ("_drawn", "drawn"), ("_bg", "bg")):
                cells = []
                for c in SHOW[:3]:
                    d, pv = perm_test(a[f"{kind}_{c}{sfx}"].to_numpy(float), b[f"{kind}_{c}{sfx}"].to_numpy(float), rng)
                    cells.append(f"{d:+.3f} ({pv:.3f})")
                lines.append(f"| {fail} vs {ctrl} | {mq} | {kind.upper()} | {name} | " + " | ".join(cells) + " |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

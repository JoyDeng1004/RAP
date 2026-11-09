"""K0 real-vs-raster diagnosis, B: which single box decides the pick? On the real-only failure tokens (collision,
offroad: viz_k0_failures.pick_tokens, the V1 / V3 tokens), swap only the F0 patches one box covers from real to
c1 features, re-decode, PDM-score the 64 candidates and record whether the pick is rescued (PDMS >= 0.5).

Boxes and their patches: t2_render.surface_map on CAM_F0 under the identity rig, i.e. the per-pixel ray cast
against gt_boxes_world in the raster's own camera (with the official render shift), so a patch belongs to the box
the c1 raster actually draws there; occlusion between boxes is resolved by the nearest hit. Projecting the boxes
with lidar2img would miss the render shift and land beside the drawn box. A box covers a 16 px patch when at least
MIN_PATCH_FRAC of the patch's canvas pixels show it. Occlusion = 1 − visible / unoccluded footprint (ray cast of
the box alone on a stride-STRIDE grid). Position in the ego frame: distance of the box centre, lateral offset y;
lane = same (|y| < LANE_HALF), adjacent (< 3 LANE_HALF) or far, a geometric proxy without the lane graph; nearest =
closest box ahead (x > 0) among the F0-visible ones; lead = closest box ahead in the same lane.

Null control per box: the same number of random F0 content patches (--n-null draws, seeded by token and box).
Also: all F0 box patches at once (the V3 f0_obj analogue, from the ray-cast boxes).

Writes eval/k0_results/boxswap/<model>/boxes.csv (one row per box), tokens.csv, summary.md and per-token figures
(F0 real image, each box's patches outlined red = rescues alone, grey = does not; distance / lane / occlusion).

    python tools/boxswap_k0_failures.py --rap-root $RAP --agent-config hydra/rap_v2_agent.yaml \
        --metric-cache <subset cache> --scenario-dir <stage A dir> --tag k0b --model D2-K0-scene-mix_last
"""
import argparse
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
sys.path.insert(0, str(HERE.parent / "v1"))
from summarize_t1 import EXP, SUBSET, latest_csv, load, token_logs  # noqa: E402
from viz_k0_failures import PATCH, ZERO, pick_tokens, pool_shares, save, to_ego, to_img  # noqa: E402

GROUPS = ("collision", "offroad")
OK = 0.5
MIN_PATCH_FRAC = 0.1
STRIDE = 4
LANE_HALF = 1.8
CANVAS_PER_PATCH = 40  # 16 input px at the 0.4 canvas -> input scale
DIST_BINS = (0, 10, 20, 40, np.inf)


def box_patches(obj: np.ndarray, kind_box: np.ndarray, crop: int, n_rows: int, n_cols: int) -> dict:
    """{box id: (n_rows, n_cols) bool} of patches whose canvas pixels show the box for >= MIN_PATCH_FRAC."""
    view = np.where(kind_box, obj, -1)[crop: crop + n_rows * CANVAS_PER_PATCH, : n_cols * CANVAS_PER_PATCH]
    blocks = view.reshape(n_rows, CANVAS_PER_PATCH, n_cols, CANVAS_PER_PATCH).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(n_rows, n_cols, -1)
    out = {}
    for i in np.unique(view[view >= 0]):
        frac = (blocks == i).mean(-1)
        m = frac >= MIN_PATCH_FRAC
        if m.any():
            out[int(i)] = m
    return out


def box_attrs(scenario: dict, i: int) -> dict:
    b = np.asarray(scenario["anns"]["gt_boxes_world"], np.float64).reshape(-1, 7)[i]
    xy = to_ego(b[None, :2], scenario["ego_heading"])[0]
    y = float(xy[1])
    lane = "same" if abs(y) < LANE_HALF else ("adjacent" if abs(y) < 3 * LANE_HALF else "far")
    names = np.asarray(scenario["anns"].get("gt_names", []))
    return {"x": float(xy[0]), "y": y, "dist": float(np.hypot(*xy)), "lane": lane,
            "cls": str(names[i]) if i < len(names) else "?"}


def summarize(boxes: pd.DataFrame, toks: pd.DataFrame, n_null: int, out: Path, title: str):
    def rate(s):
        return f"{100 * s.mean():.0f}% ({len(s)})" if len(s) else "–"

    null_cols = [f"null{j}_ok" for j in range(n_null)]
    boxes = boxes.copy()
    boxes["null_ok"] = boxes[null_cols].mean(1)
    boxes["dist_bin"] = pd.cut(boxes["dist"], DIST_BINS, right=False,
                               labels=[f"{a}-{b} m" for a, b in zip(DIST_BINS[:-1], DIST_BINS[1:])])
    boxes["occ_bin"] = pd.cut(boxes["occlusion"], (-0.01, 0.2, 0.6, 1.0), labels=["<20%", "20-60%", ">60%"])
    lines = [f"# {title}", "", f"Tokens: {toks['group'].value_counts().to_dict()}; boxes visible in F0: {len(boxes)}. "
             f"Cell = share of swaps that rescue the pick (PDMS ≥ {OK}) (n boxes); null = same number of random F0 "
             f"patches, mean over {n_null} draw(s).", ""]
    for name, col in (("distance", "dist_bin"), ("lane (|y| proxy)", "lane"), ("nearest box ahead", "nearest"),
                      ("lead box (nearest ahead in the same lane)", "lead"),
                      ("occlusion", "occ_bin"), ("class", "cls")):
        lines += [f"## By {name}", "", "| " + name + " | group | box swap | null |", "|---|---|---|---|"]
        for (v, g), sub in boxes.groupby([col, "group"], observed=True):
            lines.append(f"| {v} | {g} | {rate(sub['ok'])} | {rate(sub['null_ok'])} |")
        lines.append("")
    lines += ["## Per token", "", "| group | n | some single box rescues | lead box rescues (tokens with a lead) | "
              "nearest box rescues | best null rescues | all F0 boxes rescue | mean boxes / token |",
              "|---|---|---|---|---|---|---|---|"]
    for g, sub in toks.groupby("group"):
        lines.append(f"| {g} | {len(sub)} | {rate(sub['any_box'])} | {rate(sub['lead_ok'].dropna().astype(bool))} | "
                     f"{rate(sub['nearest_ok'].dropna().astype(bool))} | {rate(sub['any_null'])} | "
                     f"{rate(sub['all_boxes_ok'])} | {sub['n_boxes'].mean():.1f} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def plot_token(img: np.ndarray, boxes: pd.DataFrame, masks: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
    for _, b in boxes.iterrows():
        m = masks[int(b["box"])]
        rr, cc = np.nonzero(m)
        x0, x1, y0, y1 = cc.min() * PATCH, (cc.max() + 1) * PATCH, rr.min() * PATCH, (rr.max() + 1) * PATCH
        col = "red" if b["ok"] else "0.6"
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=2.2 if b["ok"] else 1.2,
                                        ls="-" if b["nearest"] else "--"))
        above = y0 >= 14
        ax.text(x0 + 2, y0 - 2 if above else y1 + 2, f"{b['dist']:.0f}m {b['lane'][:3]} occ{b['occlusion']:.0%}"
                + (" LEAD" if b["lead"] else "") + (" N" if b["nearest"] else "") + (" *" if b["null_ok_any"] else ""),
                color="white", fontsize=7, va="bottom" if above else "top", bbox=dict(fc=col, alpha=0.8, lw=0, pad=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title + "\nred = swapping this box alone rescues; grey = not; solid = nearest box ahead; "
                 "LEAD = nearest ahead in the same lane; * = a random patch set of the same size also rescues", fontsize=9)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True)
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=0, help="smoke test: first N tokens of each group")
    ap.add_argument("--max-boxes", type=int, default=20, help="nearest visible boxes per token")
    ap.add_argument("--n-null", type=int, default=1)
    ap.add_argument("--n-plot", type=int, default=8)
    ap.add_argument("--control-min", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or HERE.parent / "k0_results" / "boxswap" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_csv = latest_csv(args.exp_root / f"k0probe_{args.tag}_{args.model}")
    if probe_csv is None:
        raise SystemExit(f"no probe run k0probe_{args.tag}_{args.model} under {args.exp_root}")
    probe = load(probe_csv)
    probe = probe[probe["valid"].astype(bool)]
    logs = token_logs(args.subset_csv)
    groups = pick_tokens(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv", probe,
                         pool_shares(probe_csv), logs, args.model, args.n, args.n, args.control_min, args.seed)
    groups = {g: (groups[g][: args.max_tokens] if args.max_tokens else groups[g]) for g in GROUPS}
    print({g: len(v) for g, v in groups.items()}, "probe:", probe_csv, flush=True)

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
    from swap_k0_failures import decode_rounds, mixed

    agent = instantiate(OmegaConf.load(args.agent_config))
    agent.initialize()
    model = agent._model
    n_rounds = len(model._trajectory_head)
    rmod = tr.renderer_module(args.rap_root)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())
    mc_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    slot = tr.CAMS.index("cam_f0")
    n_rows, n_cols = rp.CONTENT_H, rp.FEAT_W

    tokens = list(dict.fromkeys(t for g in groups for t in groups[g]))
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

    def run(fb, imf_list, metric_cache):
        o = decode_rounds(model, fb, imf_list)
        cand = o["proposals"][0, :, :8].float().cpu().numpy()
        pool, sub = score_candidates(metric_cache, cand, simulator, scorer, return_subscores=True)
        i = int(o["score"][0].argmax())
        return float(pool[i]), bool(sub["no_collision"][i] <= ZERO), bool(sub["drivable_area"][i] <= ZERO)

    content = np.zeros((rp.FEAT_H, rp.FEAT_W), bool)
    content[: rp.CONTENT_H] = True
    box_rows, tok_rows, n_bad = [], [], 0
    for g, toks in groups.items():
        for rank, token in enumerate(toks):
            t0 = time.time()
            scen = scenarios[token]
            with lzma.open(mc_paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            surf = tr.surface_map(rmod, scen, tr.IDENT, channels=("CAM_F0",))["CAM_F0"]
            masks = box_patches(surf["obj"], surf["kind"] == tr.KIND_BOX, tr.CROP, n_rows, n_cols)
            # occlusion: visible pixels vs the box ray-cast alone, on a stride grid
            all_boxes = tr._boxes(scen)
            rr, cc = np.mgrid[tr.CROP: tr.CANVAS_H - tr.CROP: STRIDE, 0: tr.CANVAS_W: STRIDE]
            rr, cc = rr.ravel().astype(np.float64), cc.ravel().astype(np.float64)
            centre, dirs = tr._pixel_rays(surf["T"], surf["K"], rr, cc)
            vis_grid = np.where(surf["kind"] == tr.KIND_BOX, surf["obj"], -1)[rr.astype(int), cc.astype(int)]
            attrs = {}
            for i in masks:
                alone = np.isfinite(tr._hit_boxes(centre, dirs, [all_boxes[i]], surf["T"], surf["K"], rr, cc))
                attrs[i] = dict(box_attrs(scen, i), occlusion=float(1 - (vis_grid == i).sum() / max(alone.sum(), 1)))
            ahead = [i for i in masks if attrs[i]["x"] > 0]
            nearest = min(ahead, key=lambda i: attrs[i]["dist"]) if ahead else None
            same = [i for i in ahead if attrs[i]["lane"] == "same"]
            lead = min(same, key=lambda i: attrs[i]["dist"]) if same else None
            keep = sorted(masks, key=lambda i: attrs[i]["dist"])[: args.max_boxes]

            with torch.no_grad():
                f_r, f_c = features(token, False), features(token, True)
                imf_r, imf_c = rp.encode(model, f_r)[0], rp.encode(model, f_c)[0]
                n_cam = imf_r[0].shape[0]
                base = {k: run(fb, [imf] * n_rounds, metric_cache)
                        for k, fb, imf in (("real", f_r, imf_r), ("c1", f_c, imf_c))}
                for k in ("real", "c1"):
                    if abs(base[k][0] - probe.at[token, f"pdms_{k}"]) > 1e-6:
                        n_bad += 1
                        print(f"WARNING {token}: pdms_{k} {base[k][0]:.4f} vs probe {probe.at[token, f'pdms_{k}']:.4f}",
                              flush=True)

                def swap(m2d):
                    m = torch.zeros(n_cam, rp.FEAT_H * rp.FEAT_W, dtype=torch.bool)
                    full = np.zeros((rp.FEAT_H, rp.FEAT_W), bool)
                    full[: m2d.shape[0]] = m2d
                    m[slot] = torch.from_numpy(full.reshape(-1))
                    return run(f_r, [mixed(imf_r, imf_c, m)] * n_rounds, metric_cache)

                union = np.zeros((n_rows, n_cols), bool)
                for i in masks:
                    union |= masks[i]
                all_ok = swap(union)[0] >= OK if masks else False
                rows_tok = []
                for i in keep:
                    pd_, coll, off = swap(masks[i])
                    row = {"group": g, "token": token, "box": i, **attrs[i], "nearest": i == nearest, "lead": i == lead,
                           "n_patches": int(masks[i].sum()), "pdms": pd_, "ok": pd_ >= OK, "coll": coll, "off": off}
                    k = int(masks[i].sum())
                    for j in range(args.n_null):
                        rng = np.random.default_rng([args.seed, j, i, int(token[:8], 16)])
                        flat = np.flatnonzero(content[:n_rows].reshape(-1))
                        nm = np.zeros(n_rows * n_cols, bool)
                        nm[rng.choice(flat, size=min(k, len(flat)), replace=False)] = True
                        row[f"null{j}_ok"] = swap(nm.reshape(n_rows, n_cols))[0] >= OK
                    row["null_ok_any"] = any(row[f"null{j}_ok"] for j in range(args.n_null))
                    rows_tok.append(row)
            box_rows += rows_tok
            bt = pd.DataFrame(rows_tok)
            tok_rows.append({"group": g, "token": token, "log": logs[token], "pdms_real": base["real"][0],
                             "pdms_c1": base["c1"][0], "n_boxes": len(masks), "n_tested": len(keep),
                             "any_box": bool(bt["ok"].any()) if len(bt) else False,
                             "nearest_ok": bool(bt.loc[bt["nearest"], "ok"].any()) if nearest in keep else np.nan,
                             "has_lead": lead is not None,
                             "lead_ok": bool(bt.loc[bt["lead"], "ok"].any()) if lead in keep else np.nan,
                             "any_null": bool(bt["null_ok_any"].any()) if len(bt) else False,
                             "all_boxes_ok": all_ok, "t_token": time.time() - t0})
            print(f"[{g}] {token} boxes {len(masks)} tested {len(keep)} rescued "
                  f"{int(bt['ok'].sum()) if len(bt) else 0} null {int(bt['null_ok_any'].sum()) if len(bt) else 0} "
                  f"all {all_ok} ({time.time() - t0:.1f}s)", flush=True)
            if rank < args.n_plot and len(bt):
                plot_token(to_img(f_r, slot, rp), bt, masks,
                           f"{args.model} | {g} #{rank} | {token} | real {base['real'][0]:.2f} c1 {base['c1'][0]:.2f} | "
                           f"all F0 boxes rescue: {all_ok}", out_dir / f"{g}_{rank}_{token}.webp")

    boxes, toks = pd.DataFrame(box_rows), pd.DataFrame(tok_rows)
    boxes.to_csv(out_dir / "boxes.csv", index=False)
    toks.to_csv(out_dir / "tokens.csv", index=False)
    print(f"pdms mismatches vs probe: {n_bad}", flush=True)
    summarize(boxes, toks, args.n_null, out_dir / "summary.md", f"B single-box swaps, {args.model} (tag {args.tag})")


if __name__ == "__main__":
    main()

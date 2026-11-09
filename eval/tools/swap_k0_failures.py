"""K0 real-vs-raster diagnosis, V3: causal feature swaps between the real input and the official raster (c1, same
calibration) on the real-only failure tokens and their difficulty-matched controls (viz_k0_failures.pick_tokens,
same seed, so the same tokens as V1).

The FPN features of both inputs share one layout, (camera, 28 x 48 patches, batch, channels), and one lidar2img,
so a swap replaces positions of one input's features by the other's and leaves everything else as is.
Every variant decodes its own 64 candidates, which are PDM-scored (oracle64_v1.score_candidates); the pick is the
scorer's argmax.

  real / c1            the two inputs (checked against the probe CSV)
  sel_c1feat           real proposals teacher-forced, c1 features: the scorer re-ranks the real pool reading c1
  sel_realfeat         the reverse, on the c1 pool
  rescue_<part>        real features with <part> taken from c1
  break_<part>         c1 features with <part> taken from real
  <part>: cam_f0 / cam_l0 / cam_r0 / cam_b0 (whole camera; B0 is black in the official raster), obj / line / bg
  (patches of every camera whose c1 raster is >= OBJ_FRAC non-black, partly non-black, or black: boxes and lights,
  thin map lines, empty), f0_obj / f0_rest (F0's obj patches / the rest of F0), r4 (only the last refiner round
  reads the other input), r123 (rounds 1-3 do; break_r4 / break_r123 would repeat rescue_r123 / rescue_r4)
  null controls, N_RAND draws each, seeded by token: rand<i> = as many content patches as obj, uniform over all
  cameras; f0_rand<i> = as many F0 content patches as f0_obj, uniform over F0

Writes eval/k0_results/swap/<model>/tokens.csv (PDMS and gates per variant) and summary.md (per group: mean PDMS
and share of tokens with PDMS >= 0.5 per variant).

    python tools/swap_k0_failures.py --rap-root $RAP --agent-config hydra/rap_v2_agent.yaml \
        --metric-cache <subset cache> --scenario-dir <stage A dir> --tag k0b --model D2-K0-scene-mix_last
"""
import argparse
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
sys.path.insert(0, str(HERE.parent / "v1"))
from summarize_t1 import EXP, SUBSET, latest_csv, load, token_logs  # noqa: E402
from viz_k0_failures import PATCH, ZERO, pick_tokens, pool_shares  # noqa: E402

GROUPS = ("collision", "offroad", "hard_coll", "hard_off")
CAMS_SWAP = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")
REGIONS = ("obj", "line", "bg")
ROUNDS = ("r4", "r123")
OBJ_FRAC = 0.3
N_RAND = 3
ATOL = 1e-5
OK = 0.5


def variants() -> list:
    rand = [f"rand{i}" for i in range(N_RAND)] + [f"f0_rand{i}" for i in range(N_RAND)]
    rescue = list(CAMS_SWAP) + list(REGIONS) + ["f0_obj", "f0_rest"] + rand + list(ROUNDS)
    brk = list(CAMS_SWAP) + list(REGIONS) + ["f0_obj"] + [f"f0_rand{i}" for i in range(N_RAND)]
    return (["real", "c1", "sel_c1feat", "sel_realfeat"] + [f"rescue_{p}" for p in rescue]
            + [f"break_{p}" for p in brk])


def random_masks(obj: np.ndarray, slot_f0: int, content: np.ndarray, token: str, seed: int) -> dict:
    """Null masks with as many patches as ``obj`` (all cameras) / as obj within F0, over content patches."""
    n_cam = obj.shape[0]
    out = {}
    for i in range(N_RAND):
        rng = np.random.default_rng([seed, i, int(token[:8], 16)])
        flat = np.flatnonzero(np.tile(content, n_cam))
        m = np.zeros(obj.size, bool)
        m[rng.choice(flat, size=min(int(obj.sum()), len(flat)), replace=False)] = True
        out[f"rand{i}"] = m.reshape(obj.shape)
        f0 = np.zeros_like(obj)
        cols = np.flatnonzero(content)
        f0[slot_f0, rng.choice(cols, size=min(int(obj[slot_f0].sum()), len(cols)), replace=False)] = True
        out[f"f0_rand{i}"] = f0
    return out


def region_masks(f_c1, cams, rp) -> dict:
    """(N_cam, H*W) bool masks of obj / line / bg patches from the c1 raster's non-black pixel fraction."""
    frac = np.stack([(~rp.black_mask(f_c1, s).cpu().numpy()).reshape(rp.FEAT_H, PATCH, rp.FEAT_W, PATCH).mean((1, 3))
                     for s in range(len(cams))]).reshape(len(cams), -1)
    return {"obj": frac >= OBJ_FRAC, "line": (frac > 0) & (frac < OBJ_FRAC), "bg": frac == 0}


def mixed(imf_base, imf_other, mask):
    """``imf_base`` with the (camera, patch) positions in ``mask`` taken from ``imf_other``."""
    ff = imf_base[0].clone()
    m = mask.to(ff.device)
    ff[m] = imf_other[0][m]
    return (ff,) + tuple(imf_base[1:])


def decode_rounds(model, feats, imfs, force=None):
    """rp.decode with one image feature per refiner round (and optional teacher-forced proposals)."""
    import torch
    heads = model._trajectory_head
    ego = feats["ego_status"][:, -1]
    if model.b2d:
        ego = ego.clone()
        ego[:, 1:3] = 0
    bev = model.hist_encoding(ego)[:, None] + model.init_feature.weight[None]
    props = []
    for i, (refine, imf) in enumerate(zip(heads, imfs)):
        if force is None:
            bev, props = refine(bev, props, imf)
        else:
            props.append(force[i])
            bev = refine.Bev_refiner(force[i], bev, imf)
    proposals = props[-1]
    pred_logit, pred_logit2 = model.scorer(proposals, bev)[:2]
    if pred_logit2 is not None:
        score = (torch.sigmoid(pred_logit) + torch.sigmoid(pred_logit2))[:, :, -1] / 2
    else:
        score = torch.sigmoid(pred_logit)[:, :, -1]
    return {"proposals": proposals, "score": score, "props": props}


def summarize(df: pd.DataFrame, out: Path, title: str):
    vs = variants()
    groups = [g for g in GROUPS if g in set(df["group"])]
    n = df["group"].value_counts().to_dict()
    lines = [f"# {title}", "",
             f"Tokens per group: {n}. Cell = mean PDMS ×100 / share of tokens with PDMS ≥ {OK} (%). Failure groups "
             "start at real = 0, so their share is the rescue rate; matched controls start ≥ 0.9, so a drop is a "
             "break.", "", "| variant | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for v in vs:
        if f"pdms_{v}" not in df:
            continue
        cells = []
        for g in groups:
            x = df.loc[df["group"] == g, f"pdms_{v}"]
            cells.append(f"{100 * x.mean():.1f} / {100 * (x >= OK).mean():.0f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", "Gates of the pick on the failure groups (% collision / % off-road):", "",
              "| variant | " + " | ".join(g for g in groups if g in ("collision", "offroad")) + " |",
              "|---|" + "---|" * len([g for g in groups if g in ("collision", "offroad")])]
    for v in vs:
        if f"coll_{v}" not in df:
            continue
        cells = [f"{100 * df.loc[df['group'] == g, f'coll_{v}'].mean():.0f} / "
                 f"{100 * df.loc[df['group'] == g, f'off_{v}'].mean():.0f}" for g in groups if g in ("collision", "offroad")]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", f"Null controls ({N_RAND} random draws; cell = mean over draws of the PDMS ≥ {OK} share (%), "
              "[min, max] over draws), next to the structured swap of the same size:", "",
              "| swap | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for d, structured, null in (("rescue", "obj", "rand"), ("rescue", "f0_obj", "f0_rand"), ("break", "f0_obj", "f0_rand")):
        cols = [f"pdms_{d}_{null}{i}" for i in range(N_RAND)]
        if f"pdms_{d}_{structured}" not in df or not all(c in df for c in cols):
            continue
        for name, cs in ((f"{d}_{structured}", [f"pdms_{d}_{structured}"]), (f"{d}_{null}", cols)):
            cells = []
            for g in groups:
                rates = [100 * (df.loc[df["group"] == g, c] >= OK).mean() for c in cs]
                cells.append(f"{np.mean(rates):.0f}" + (f" [{min(rates):.0f}, {max(rates):.0f}]" if len(cs) > 1 else ""))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if "n_obj" in df:
        lines += ["", "Patches swapped per token (mean): obj " + f"{df['n_obj'].mean():.0f}, F0 obj {df['n_f0_obj'].mean():.0f}"
                  f" (of {4 * 27 * 48} / {27 * 48} content patches)."]
    shares = [c for c in df if c.startswith("share_")]
    if shares:
        lines += ["", "Mean share of patches per region (all cameras): " +
                  ", ".join(f"{c[6:]} {df[c].mean():.1%}" for c in shares)]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


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
    ap.add_argument("--n", type=int, default=30, help="tokens per group (the V1 groups with --n-mean of the same size)")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--max-tokens", type=int, default=0, help="smoke test: first N tokens of each group")
    ap.add_argument("--control-min", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or HERE.parent / "k0_results" / "swap" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_csv = latest_csv(args.exp_root / f"k0probe_{args.tag}_{args.model}")
    if probe_csv is None:
        raise SystemExit(f"no probe run k0probe_{args.tag}_{args.model} under {args.exp_root}")
    probe = load(probe_csv)
    probe = probe[probe["valid"].astype(bool)]
    logs = token_logs(args.subset_csv)
    groups = pick_tokens(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv", probe,
                         pool_shares(probe_csv), logs, args.model, args.n, args.n, args.control_min, args.seed)
    groups = {g: (groups[g][: args.max_tokens] if args.max_tokens else groups[g]) for g in args.groups}
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

    agent = instantiate(OmegaConf.load(args.agent_config))
    agent.initialize()
    model = agent._model
    n_rounds = len(model._trajectory_head)
    rmod = tr.renderer_module(args.rap_root)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())
    mc_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths

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

    rows = []
    for g, toks in groups.items():
        for token in toks:
            t0 = time.time()
            with lzma.open(mc_paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            row = {"group": g, "token": token, "log": logs[token]}
            with torch.no_grad():
                f_r, f_c = features(token, False), features(token, True)
                if not torch.equal(f_r["lidar2img"], f_c["lidar2img"]):
                    raise AssertionError(f"{token}: c1 calibration != real calibration")
                imf_r, imf_c = rp.encode(model, f_r)[0], rp.encode(model, f_c)[0]
                if imf_r[0].shape[1] != rp.FEAT_H * rp.FEAT_W:
                    raise AssertionError(f"unexpected feature layout {tuple(imf_r[0].shape)}")
                outs = {"real": decode_rounds(model, f_r, [imf_r] * n_rounds),
                        "c1": decode_rounds(model, f_c, [imf_c] * n_rounds)}
                ref = rp.decode(model, f_r, imf_r)
                if not (torch.allclose(outs["real"]["proposals"], ref["proposals"], rtol=0, atol=ATOL)
                        and torch.allclose(outs["real"]["score"], ref["score"], rtol=0, atol=ATOL)):
                    raise AssertionError(f"{token}: decode_rounds(real) != rp.decode(real)")
                outs["sel_c1feat"] = decode_rounds(model, f_c, [imf_c] * n_rounds, force=outs["real"]["props"])
                outs["sel_realfeat"] = decode_rounds(model, f_r, [imf_r] * n_rounds, force=outs["c1"]["props"])

                n_cam = imf_r[0].shape[0]
                masks = {c: torch.zeros(n_cam, rp.FEAT_H * rp.FEAT_W, dtype=torch.bool) for c in CAMS_SWAP}
                for c in CAMS_SWAP:
                    masks[c][tr.CAMS.index(c)] = True
                reg = region_masks(f_c, tr.CAMS, rp)
                slot_f0 = tr.CAMS.index("cam_f0")
                content = np.zeros((rp.FEAT_H, rp.FEAT_W), bool)
                content[: rp.CONTENT_H] = True
                content = content.reshape(-1)
                f0 = np.zeros_like(reg["obj"])
                f0[slot_f0] = content
                reg["f0_obj"] = reg["obj"] & f0
                reg["f0_rest"] = f0 & ~reg["obj"]
                reg.update(random_masks(reg["obj"] & np.tile(content, (n_cam, 1)), slot_f0, content, token, args.seed))
                for k, m in reg.items():
                    masks[k] = torch.from_numpy(m)
                    if not k[-1].isdigit():
                        row[f"share_{k}"] = float(m.mean())
                row["n_obj"], row["n_f0_obj"] = int(reg["obj"].sum()), int(reg["f0_obj"].sum())
                vs = set(variants())
                for d, (base, other, fb) in {"rescue": (imf_r, imf_c, f_r), "break": (imf_c, imf_r, f_c)}.items():
                    for p, m in masks.items():
                        if f"{d}_{p}" in vs:
                            outs[f"{d}_{p}"] = decode_rounds(model, fb, [mixed(base, other, m)] * n_rounds)
                outs["rescue_r4"] = decode_rounds(model, f_r, [imf_r] * (n_rounds - 1) + [imf_c])
                outs["rescue_r123"] = decode_rounds(model, f_r, [imf_c] * (n_rounds - 1) + [imf_r])

            pools = {}
            for v, o in outs.items():
                cand = o["proposals"][0, :, :8].float().cpu().numpy()
                if v == "sel_c1feat":
                    pool, sub = pools["real"]
                elif v == "sel_realfeat":
                    pool, sub = pools["c1"]
                else:
                    pool, sub = score_candidates(metric_cache, cand, simulator, scorer, return_subscores=True)
                    pools[v] = (pool, sub)
                i = int(o["score"][0].argmax())
                row[f"pdms_{v}"] = float(pool[i])
                row[f"coll_{v}"] = bool(sub["no_collision"][i] <= ZERO)
                row[f"off_{v}"] = bool(sub["drivable_area"][i] <= ZERO)
                row[f"oracle_{v}"] = float(pool.max())
            p = probe.loc[token]
            for k in ("real", "c1"):
                if abs(row[f"pdms_{k}"] - p[f"pdms_{k}"]) > 1e-6:
                    print(f"WARNING {token}: pdms_{k} {row[f'pdms_{k}']:.4f} vs probe {p[f'pdms_{k}']:.4f}", flush=True)
            row["t_token"] = time.time() - t0
            rows.append(row)
            print(f"[{g}] {token} real {row['pdms_real']:.2f} c1 {row['pdms_c1']:.2f} sel_c1feat "
                  f"{row['pdms_sel_c1feat']:.2f} ({row['t_token']:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tokens.csv", index=False)
    summarize(df, out_dir / "summary.md", f"V3 feature swaps, {args.model} (tag {args.tag})")


if __name__ == "__main__":
    main()

"""Exp-0 / T1 step 8 probes for one rig preset, NAVSIM v1, same token set / pool as run_oracle64_v1.py.

Per token (all forward-only; every candidate set is PDM-scored with oracle64_v1.score_candidates):
  raw       raw input: 64 proposals, scorer pick
  img       the preset applied to the images (== the T1 run): regenerated proposals, scorer pick
  fix_last  8a: raw proposals, last refiner round + scorer read the new images
  fix_all   8a: raw proposals teacher-forced in every round, all rounds read the new images
  crop      pure translations only: the crop control (same visible content, original position)
  feat      8c (pure translations only): the decoder gets the crop-control features moved by the
            shift (an exactly equivariant backbone) with the shifted calibration
  img_oob   boundary test: img, but SCA samples on black fill / padding count as outside the image
            (contribute 0, like points that project outside the raw image); rig_probe.fill_as_outside
  raw_oob   its control: raw with the same rule (only the padding row and stray black pixels move)
For the figures every token also stores: query cosine to raw after each refiner round, slot-matched
proposal ADE per round and pool coverage; F0 SCA attention mass per round and waypoint on
out-of-image / black fill / padding / content (rig_probe.sca_stats); the candidate arrays; and,
for pure translations, F0 28 x 48 similarity maps.

And, for pure translations, 8b feature statistics: cosine / relative L2 between the real shifted
features and the moved crop features (FPN output and raw ViT tokens), the same without moving
(unaligned), crop vs raw on the kept content, and raw vs resample_ctrl (interpolation noise floor).

Checks per token (failure -> row invalid with assert_fail=True; run exits non-zero after the CSV):
  - decode(encode(raw)) == model(raw, return_score=True) (proposals and scores)
  - fixed-candidate variants return exactly the raw proposals
  - recomputed PDMS of the raw / img picks == official pdm_score (< 1e-6)

Outputs <ts>.csv (per-token rows + average row) and <ts>_arrays.npz.

    python run_t1_probe_v1.py --rap-root $RAP --agent-config rap_v2_official_agent.yaml \
        --metric-cache <subset cache> --preset ppy_m7r --output-dir <dir> --workers 4
"""
import argparse
import contextlib
import logging
import lzma
import os
import pickle
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from run_pdm_score_v1 import load_split

logger = logging.getLogger("run_t1_probe_v1")

ATOL_MODEL = 1e-5
SLOT_F0 = 1  # RAP camera order b0, f0, l0, r0
CAND_VARIANTS = ("raw", "img", "crop", "feat", "raw_oob", "img_oob")  # variants with their own candidate sets
SCORE_VARIANTS = ("raw", "img", "fix_last", "fix_all", "crop", "feat", "raw_oob", "img_oob")
SCA_VARIANTS = ("raw", "img", "fix_all", "crop", "feat", "raw_oob", "img_oob")  # SCA sampling statistics recorded for these


def score_logs_probe(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from oracle64_v1 import check_idx, score_candidates
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents.rap_v2_agent import RAPV2Agent
    from rap_eval_agents.rig_transform import apply_rig, feature_shift, get_spec

    job = args[0]
    thread_id = str(uuid.uuid4())
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())

    agent = instantiate(OmegaConf.load(job["agent_config"]))
    if type(agent) is not RAPV2Agent:
        raise SystemExit(f"use the plain RAPV2Agent config (the probe applies the rig itself), got {type(agent)}")
    agent.initialize()
    model = agent._model

    spec = get_spec(job["preset"])
    shift = feature_shift(spec)
    crop_spec = get_spec("crop_" + job["preset"]) if shift is not None else None
    noise_spec = get_spec("resample_ctrl")

    scene_filter = job["scene_filter"]
    scene_filter.log_names = [a["log_file"] for a in args]
    scene_filter.tokens = [t for a in args for t in a["tokens"]]
    scene_loader = SceneLoader(
        data_path=Path(job["navsim_log_path"]),
        sensor_blobs_path=Path(job["sensor_blobs_path"]),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    paths = job["metric_cache_paths"]
    tokens = [t for t in scene_filter.tokens if t in paths and t in scene_loader.scene_frames_dicts]

    def features(token, s):
        # RAPV2Agent._features edits the cameras in place: a fresh AgentInput for every variant.
        agent_input = scene_loader.get_agent_input_from_token(token)
        apply_rig(agent_input.cameras[-1], s)
        return agent._features(agent_input)

    def cands(out):
        return (out["proposals"][0, :, :8].float().cpu().numpy(), out["score"][0].float().cpu().numpy())

    rows = []
    for idx, token in enumerate(tokens):
        if idx % 20 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        row: Dict[str, Any] = {"token": token, "valid": True, "assert_fail": False, "error": ""}
        t0 = time.time()
        try:
            with lzma.open(paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            with torch.no_grad():
                def run(name, f, imf, oob=False, **kw):
                    oob_ctx = rp.fill_as_outside(model, f) if oob else contextlib.nullcontext()
                    if name not in SCA_VARIANTS:
                        with oob_ctx:
                            return rp.decode(model, f, imf, **kw)
                    # record_sca outside fill_as_outside: it records the locations actually sampled
                    with rp.record_sca(model) as rec, oob_ctx:  # record_sca leaves outputs unchanged
                        o = rp.decode(model, f, imf, **kw)
                    row[f"_sca_{name}"] = rp.sca_stats(rec, f, o["proposals"].shape[2], SLOT_F0).numpy()
                    return o

                f_raw = features(token, get_spec("raw"))
                imf_raw, vit_raw = rp.encode(model, f_raw, want_vit=shift is not None)
                o_raw = run("raw", f_raw, imf_raw)
                ref = model(f_raw, None, return_score=True)
                if not (torch.allclose(o_raw["proposals"], ref["trajectory"], rtol=0, atol=ATOL_MODEL)
                        and torch.allclose(o_raw["score"], ref["score"], rtol=0, atol=ATOL_MODEL)):
                    raise AssertionError("decode(encode(raw)) != model(raw)")

                f_new = features(token, spec)
                imf_new, vit_new = rp.encode(model, f_new, want_vit=shift is not None)
                outs = {
                    "raw": o_raw,
                    "img": run("img", f_new, imf_new),
                    "fix_last": run("fix_last", f_new, imf_new, inject_last=(o_raw["bev_in"][-1], o_raw["props"][-1])),
                    "fix_all": run("fix_all", f_new, imf_new, force=o_raw["props"]),
                    "raw_oob": run("raw_oob", f_raw, imf_raw, oob=True),
                    "img_oob": run("img_oob", f_new, imf_new, oob=True),
                }
                for k in ("fix_last", "fix_all"):
                    if not torch.equal(outs[k]["proposals"], o_raw["proposals"]):
                        raise AssertionError(f"{k} changed the raw proposals")

                if shift is not None:
                    dy, dx = shift
                    f_crop = features(token, crop_spec)
                    imf_crop, vit_crop = rp.encode(model, f_crop, want_vit=True)
                    outs["crop"] = run("crop", f_crop, imf_crop)
                    moved, valid = rp.translate(imf_crop[0], imf_new[0], dy, dx)
                    outs["feat"] = run("feat", f_new, rp.with_calibration(imf_new, moved, f_new))

                    # 8b statistics (embeddings removed: they are constant per camera)
                    s_new, s_crop, s_raw = (rp.strip_embeds(model, x[0]) for x in (imf_new, imf_crop, imf_raw))
                    s_moved = rp.translate(s_crop, s_new, dy, dx)[0]
                    v_moved = rp.translate(vit_crop, vit_new, dy, dx)[0]
                    rlo, rhi = rp.content_range(dy, rp.CONTENT_H)
                    clo, chi = rp.content_range(dx, rp.FEAT_W)
                    kept = torch.zeros_like(valid)
                    kept[rlo:rhi + 1, clo:chi + 1] = True
                    body = torch.zeros_like(valid)
                    body[:rp.CONTENT_H] = True
                    f_noise = features(token, noise_spec)
                    s_noise = rp.strip_embeds(model, rp.encode(model, f_noise)[0][0])
                    stats = {
                        "fpn_aligned": rp.masked_cos(s_new, s_moved, valid),
                        "fpn_unaligned": rp.masked_cos(s_new, s_crop, valid),
                        "vit_aligned": rp.masked_cos(vit_new, v_moved, valid),
                        "vit_unaligned": rp.masked_cos(vit_new, vit_crop, valid),
                        "fpn_crop_vs_raw": rp.masked_cos(s_crop, s_raw, kept),
                        "fpn_noise": rp.masked_cos(s_noise, s_raw, body),
                        "rl2_fpn_aligned": rp.masked_rel_l2(s_new, s_moved, valid),
                        "rl2_fpn_crop_vs_raw": rp.masked_rel_l2(s_crop, s_raw, kept),
                        "rl2_fpn_noise": rp.masked_rel_l2(s_noise, s_raw, body),
                    }
                    for k, v in stats.items():
                        v = v.cpu().numpy()
                        row[f"{k}_f0"] = float(v[SLOT_F0])
                        row[f"{k}_mean"] = float(v.mean())
                    row["n_valid_pos"] = int(valid.sum())
                    # F0 per-position maps for the 28 x 48 heatmaps (NaN outside the compared positions)
                    row["_cosmap_fpn_f0"] = rp.cos_map(s_new, s_moved, valid, SLOT_F0).cpu().numpy()
                    row["_cosmap_vit_f0"] = rp.cos_map(vit_new, v_moved, valid, SLOT_F0).cpu().numpy()
                    row["_cosmap_crop_raw_f0"] = rp.cos_map(s_crop, s_raw, kept, SLOT_F0).cpu().numpy()

                # layer-wise divergence from raw: queries after each refiner round (slot-matched, so
                # comparable across rigs), proposals per round, and pool coverage (slot-free)
                r_q = o_raw["bev_in"][1:] + [o_raw["bev_out"]]
                for k, o in outs.items():
                    if k == "raw":
                        continue
                    if o["bev_in"]:
                        for r, q in enumerate(o["bev_in"][1:] + [o["bev_out"]], start=1):
                            row[f"qcos_r{r}_{k}"] = rp.query_cos(q, r_q[r - 1])
                    else:  # fix_last: only the last round ran
                        row[f"qcos_r4_{k}"] = rp.query_cos(o["bev_out"], r_q[-1])
                    if k in CAND_VARIANTS:
                        for r, (p_new, p_raw) in enumerate(zip(o["props"], o_raw["props"]), start=1):
                            row[f"pade_r{r}_{k}"] = rp.proposal_ade(p_new, p_raw)
                        row[f"cover_{k}"] = rp.pool_cover(o_raw["proposals"], o["proposals"])
                        row[f"cover_rev_{k}"] = rp.pool_cover(o["proposals"], o_raw["proposals"])

            arr = {k: cands(o) for k, o in outs.items()}
            c_raw, sc_raw = arr["raw"]
            s64 = {k: score_candidates(metric_cache, arr[k][0], simulator, scorer) for k in CAND_VARIANTS if k in arr}
            for k in SCORE_VARIANTS:
                if k not in arr:
                    continue
                c, sc = arr[k]
                pool = s64[k] if k in s64 else s64["raw"]  # fixed-candidate variants re-rank the raw pool
                sel = int(sc.argmax())
                row[f"pdms_{k}"] = float(pool[sel])
                row[f"idx_{k}"] = sel
                row[f"_score_{k}"] = sc
                if k in s64:
                    row[f"_cand_{k}"] = c  # (64, 8, 3) for the candidate BEV plots
                    row[f"oracle_{k}"] = float(pool.max())
                    row[f"mean64_{k}"] = float(pool.mean())
                    row[f"_s64_{k}"] = pool
            for k in ("raw", "img"):
                check_idx(metric_cache, arr[k][0], s64[k], row[f"idx_{k}"], simulator, scorer, k)
        except Exception as e:
            logger.warning("probe failed for token %s", token)
            traceback.print_exc()
            row.update(valid=False, assert_fail=isinstance(e, AssertionError), error=repr(e)[:300])
        row["t_token"] = time.time() - t0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ RAPV2Agent (not the rig agent)")
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--preset", required=True, help="rig_transform.PRESETS name")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-log", type=int, default=0, help="smoke test: first N tokens of each log")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import navsim
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from rap_eval_agents.rig_transform import feature_shift, get_spec

    nav = str(Path(navsim.__file__).resolve())
    if not nav.startswith(str(args.rap_root.resolve())):
        raise SystemExit(f"navsim resolves to {nav}, expected RAP's v1 copy under {args.rap_root}")
    shift = feature_shift(get_spec(args.preset))  # unknown names fail here, before any GPU work
    if shift is not None:
        get_spec("crop_" + args.preset)
    logger.info("navsim=%s preset=%s feature_shift(rows, cols)=%s", nav, args.preset, shift)

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_split, scene_filter = load_split(args.rap_root, args.split)
    navsim_log_path = data_root / "navsim_logs" / data_split
    sensor_blobs_path = data_root / "sensor_blobs" / data_split

    scene_loader = SceneLoader(
        data_path=navsim_log_path, sensor_blobs_path=None,
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    per_log = {log: [t for t in toks if t in metric_cache_paths]
               for log, toks in scene_loader.get_tokens_list_per_log().items()}
    per_log = {log: toks for log, toks in per_log.items() if toks}
    if args.per_log:
        per_log = {log: sorted(toks)[: args.per_log] for log, toks in per_log.items()}
    logger.info("tokens=%d logs=%d", sum(len(t) for t in per_log.values()), len(per_log))

    job = dict(
        agent_config=str(Path(args.agent_config).resolve()),
        preset=args.preset,
        scene_filter=scene_filter,
        navsim_log_path=str(navsim_log_path),
        sensor_blobs_path=str(sensor_blobs_path),
        metric_cache_paths=metric_cache_paths,
    )
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]

    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    rows = worker_map(worker, score_logs_probe, data_points)

    arrays: Dict[str, List[np.ndarray]] = {}
    arr_tokens = []
    for r in rows:
        if r["valid"]:
            arr_tokens.append(r["token"])
        keys = [k for k in r if k.startswith("_")]
        for k in keys:
            v = r.pop(k)
            if r["valid"]:
                arrays.setdefault(k[1:], []).append(v)

    df = pd.DataFrame(rows)
    n_ok = int(df["valid"].sum())
    n_assert = int(df["assert_fail"].sum())
    average = df.drop(columns=["token", "valid", "assert_fail", "error"] + [c for c in df if c.startswith("idx_")],
                      errors="ignore").mean(skipna=True)
    average["token"] = "average"
    average["valid"] = bool(df["valid"].all())
    average["assert_fail"] = n_assert > 0
    df.loc[len(df)] = average

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out = args.output_dir / f"{stamp}.csv"
    df.to_csv(out)
    if arr_tokens:
        np.savez(args.output_dir / f"{stamp}_arrays.npz", token=np.array(arr_tokens),
                 **{k: np.stack(v) for k, v in arrays.items() if len(v) == len(arr_tokens)})
    msg = " ".join(f"{c}={average[c]:.4f}" for c in df.columns
                   if c.startswith(("pdms_", "oracle_", "mean64_", "fpn_aligned_", "vit_aligned_")) and c in average)
    logger.info("Finished: preset=%s valid=%d failed=%d assert_fail=%d | %s | t_token=%.2fs csv=%s",
                args.preset, n_ok, len(df) - 1 - n_ok, n_assert, msg, average["t_token"], out)
    if n_assert:
        raise SystemExit(f"{n_assert} tokens failed a probe consistency check; see 'error' column in {out}")


if __name__ == "__main__":
    main()

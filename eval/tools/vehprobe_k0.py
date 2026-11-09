"""K0 real-vs-raster diagnosis, C: linear probes for "is there a vehicle in this F0 patch" and "how far is it" on
the patch features of one checkpoint, real and raster (c1) input alike. Question: are K0's real-image features
worse at vehicles than K1's (then K0's problem is seeing vehicles in real images, not only that the raster carries
more), or does K1's real input come close to its raster?

Features per F0 patch (27 content rows x 48): the FPN output without camera / level embeddings (what the decoder
reads) and the raw DINOv3 patch tokens (frozen and shared by every checkpoint: a sanity check, its probe must
match across models).
Labels from t2_render.surface_map on CAM_F0 (ray cast of gt_boxes_world in the raster's camera, official render
shift included; the shift is the official fix that aligns the raster with the real images, so the same labels
serve both inputs; any residual misalignment hits K0 and K1 alike, which is the comparison that matters):
  positive  >= POS_FRAC of the patch's canvas pixels show a vehicle box (gt_names == "vehicle")
  negative  no box pixel of any class in the patch and its 8 neighbours
  distance  ego-frame distance of the vehicle box covering most of a positive patch
Per token all positives and up to NEG_PER_POS x as many negatives (at least MIN_NEG), seeded by token, so every
model sees the same patches.

Tokens: the probe CSV's tokens minus the failure / control groups, split by log into train / test (log hash);
evaluation also on the failure groups (collision + offroad) and the matched controls (hard_coll + hard_off).
Probes (per feature x input): standardised logistic regression (AUC) and ridge on log distance (MAE in metres,
median relative error), trained on train; plus c1 -> real transfer (probe trained on c1, applied to real).

GPU mode (one checkpoint, RAP_CKPT):
    python tools/vehprobe_k0.py --rap-root $RAP --agent-config hydra/rap_v2_agent.yaml \
        --scenario-dir <stage A dir> --tag k0b --label D2-K0-scene-mix_last
  -> eval/k0_results/vehprobe/<label>/metrics.csv, preds.csv
Compare (CPU): python tools/vehprobe_k0.py --compare D2-K0-scene-mix_last D2-K1-mix_e13
  -> eval/k0_results/vehprobe/compare.md (AUC / MAE per model, model differences with log-cluster bootstrap CIs)
"""
import argparse
import os
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "py"))
sys.path.insert(0, str(HERE.parent / "v1"))
from summarize_t1 import EXP, SUBSET, latest_csv, load, token_logs  # noqa: E402
from viz_k0_failures import pick_tokens, pool_shares, to_ego  # noqa: E402

FAIL_MODEL = "D2-K0-scene-mix_last"  # whose failures define the failure / control sets
POS_FRAC = 0.3
NEG_PER_POS = 2
MIN_NEG = 20
CANVAS_PER_PATCH = 40
FEATS = ("fpn", "vit")
INPUTS = ("real", "c1")
SETS = ("test", "failure", "control")
N_BOOT = 1000


def patch_labels(obj: np.ndarray, kind_box: np.ndarray, is_vehicle: np.ndarray, dist: np.ndarray, crop: int,
                 n_rows: int, n_cols: int):
    """Per patch (n_rows, n_cols): vehicle pixel fraction, any-box pixel presence, distance of the dominant vehicle."""
    view = np.where(kind_box, obj, -1)[crop: crop + n_rows * CANVAS_PER_PATCH, : n_cols * CANVAS_PER_PATCH]
    blocks = view.reshape(n_rows, CANVAS_PER_PATCH, n_cols, CANVAS_PER_PATCH).transpose(0, 2, 1, 3).reshape(n_rows, n_cols, -1)
    veh_ids = np.flatnonzero(is_vehicle)
    veh = np.isin(blocks, veh_ids)
    frac = veh.mean(-1)
    any_box = (blocks >= 0).any(-1)
    d = np.full((n_rows, n_cols), np.nan)
    for r, c in zip(*np.nonzero(frac > 0)):
        ids, cnt = np.unique(blocks[r, c][veh[r, c]], return_counts=True)
        d[r, c] = dist[ids[cnt.argmax()]]
    return frac, any_box, d


def sample_patches(frac: np.ndarray, any_box: np.ndarray, token: str):
    """Indices (flat) of positive patches and of sampled clean negatives (no box in the 3 x 3 neighbourhood)."""
    pad = np.pad(any_box, 1)
    near = np.zeros_like(any_box)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            near |= pad[dr: dr + any_box.shape[0], dc: dc + any_box.shape[1]]
    pos = np.flatnonzero(frac >= POS_FRAC)
    neg_all = np.flatnonzero(~near)
    rng = np.random.default_rng(int(token[:8], 16))
    k = min(len(neg_all), max(MIN_NEG, NEG_PER_POS * len(pos)))
    return pos, np.sort(rng.choice(neg_all, size=k, replace=False))


def split_of(log: str) -> str:
    return "test" if zlib.crc32(log.encode()) % 5 == 0 else "train"


def fit_eval(data: pd.DataFrame, X: dict, out_rows: list, preds: list, label: str):
    """data: one row per patch (token, log, set, y, dist); X[(feat, input)] aligned (n, C) arrays."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = (data["set"] == "train").to_numpy()
    pos = (data["y"] == 1).to_numpy()
    clf, reg = {}, {}
    for key, x in X.items():
        clf[key] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(x[tr], data["y"][tr])
        reg[key] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x[tr & pos], np.log(data["dist"][tr & pos]))
    for feat in FEATS:
        for inp, src in (("real", "real"), ("c1", "c1"), ("real_from_c1", "c1")):
            x = X[(feat, "real" if inp != "c1" else "c1")]
            p = clf[(feat, src)].predict_proba(x)[:, 1]
            dhat = np.exp(reg[(feat, src)].predict(x))
            for s in SETS:
                m = (data["set"] == s).to_numpy()
                if not m.any() or data["y"][m].nunique() < 2:
                    continue
                mp = m & pos
                err = np.abs(dhat[mp] - data["dist"][mp].to_numpy())
                out_rows.append({"model": label, "feat": feat, "input": inp, "set": s, "n": int(m.sum()),
                                 "n_pos": int(mp.sum()), "auc": roc_auc_score(data["y"][m], p[m]),
                                 "dist_mae": float(err.mean()),
                                 "dist_rel": float(np.median(err / data["dist"][mp].to_numpy()))})
            preds.append(pd.DataFrame({"feat": feat, "input": inp, "p": p, "dhat": dhat, "row": np.arange(len(data))}))


def cluster_boot(df: pd.DataFrame, stat, n: int = N_BOOT, seed: int = 0):
    """stat on df, and a 95% CI from resampling logs."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    by_log = defaultdict(list)
    for i, lg in enumerate(df["log"].to_numpy()):
        by_log[lg].append(i)
    logs = list(by_log)
    vals = []
    for _ in range(n):
        idx = np.concatenate([by_log[logs[j]] for j in rng.integers(0, len(logs), len(logs))])
        try:
            vals.append(stat(df.iloc[idx]))
        except ValueError:
            continue
    return stat(df), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def compare(labels: list, root: Path):
    from sklearn.metrics import roc_auc_score
    mets = pd.concat([pd.read_csv(root / lb / "metrics.csv") for lb in labels])
    preds = {lb: pd.read_csv(root / lb / "preds.csv") for lb in labels}
    lines = [f"# C vehicle probes: {' vs '.join(labels)}", "",
             "AUC = vehicle vs clean background per F0 patch; MAE / rel = distance error on vehicle patches (m / "
             "median relative). input real_from_c1 = probe trained on c1 features, applied to real.", "",
             "| model | feat | input | set | n (pos) | AUC | dist MAE | dist rel |", "|---|---|---|---|---|---|---|---|"]
    for _, r in mets.sort_values(["feat", "input", "set", "model"]).iterrows():
        lines.append(f"| {r.model} | {r.feat} | {r.input} | {r.set} | {r.n} ({r.n_pos}) | {r.auc:.3f} | "
                     f"{r.dist_mae:.1f} | {r.dist_rel:.2f} |")
    if len(labels) == 2:
        a, b = labels
        pa, pb = preds[a], preds[b]
        same = pa[["token", "patch", "y"]].equals(pb[["token", "patch", "y"]])
        lines += ["", f"## {a} − {b} (same patches: {same}; log-cluster bootstrap 95% CI)", "",
                  "| feat | input | set | ΔAUC [CI] | ΔMAE m [CI] |", "|---|---|---|---|---|"]
        if same:
            for feat in FEATS:
                for inp in ("real", "c1", "real_from_c1"):
                    for s in SETS:
                        m = (pa["set"] == s).to_numpy()
                        if not m.any():
                            continue
                        d = pa.loc[m, ["log", "y", "dist"]].copy()
                        d["pa"], d["pb"] = pa.loc[m, f"p_{feat}_{inp}"].to_numpy(), pb.loc[m, f"p_{feat}_{inp}"].to_numpy()
                        d["ea"] = np.abs(pa.loc[m, f"d_{feat}_{inp}"].to_numpy() - d["dist"].to_numpy())
                        d["eb"] = np.abs(pb.loc[m, f"d_{feat}_{inp}"].to_numpy() - d["dist"].to_numpy())
                        if d["y"].nunique() < 2:
                            continue
                        da = cluster_boot(d, lambda x: roc_auc_score(x["y"], x["pa"]) - roc_auc_score(x["y"], x["pb"]))
                        dm = cluster_boot(d[d["y"] == 1], lambda x: (x["ea"] - x["eb"]).mean())
                        lines.append(f"| {feat} | {inp} | {s} | {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] | "
                                     f"{dm[0]:+.2f} [{dm[1]:+.2f}, {dm[2]:+.2f}] |")
    (root / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path)
    ap.add_argument("--agent-config")
    ap.add_argument("--scenario-dir", type=Path)
    ap.add_argument("--tag", default="k0b")
    ap.add_argument("--label", help="name of the checkpoint in RAP_CKPT (output folder)")
    ap.add_argument("--exp-root", type=Path, default=EXP)
    ap.add_argument("--subset-csv", type=Path, default=SUBSET)
    ap.add_argument("--n", type=int, default=30, help="tokens per failure / control group")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--control-min", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=HERE.parent / "k0_results" / "vehprobe")
    ap.add_argument("--compare", nargs="+", default=None, help="labels to compare (CPU only)")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare, args.root)
    out_dir = args.root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_csv = latest_csv(args.exp_root / f"k0probe_{args.tag}_{FAIL_MODEL}")
    probe = load(probe_csv)
    probe = probe[probe["valid"].astype(bool)]
    logs = token_logs(args.subset_csv)
    groups = pick_tokens(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv", probe,
                         pool_shares(probe_csv), logs, FAIL_MODEL, args.n, args.n, args.control_min, args.seed)
    tail = set(pd.read_csv(HERE.parent / "k0_results" / f"k0_probe_{args.tag}_tail_tokens.csv")["token"])
    sets = {t: "failure" for g in ("collision", "offroad") for t in groups[g]}
    sets.update({t: "control" for g in ("hard_coll", "hard_off") for t in groups[g]})
    rest = [t for t in probe.index if t not in sets and t not in tail]
    rng = np.random.default_rng(args.seed)
    rest = list(rng.permutation(rest))
    tr_ = [t for t in rest if split_of(logs[t]) == "train"][: args.n_train]
    te_ = [t for t in rest if split_of(logs[t]) == "test"][: args.n_test]
    sets.update({t: "train" for t in tr_})
    sets.update({t: "test" for t in te_})
    print({s: sum(v == s for v in sets.values()) for s in ("train",) + SETS}, flush=True)

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from run_pdm_score_v1 import load_split
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents import t2_render as tr

    agent = instantiate(OmegaConf.load(args.agent_config))
    agent.initialize()
    model = agent._model
    rmod = tr.renderer_module(args.rap_root)
    slot = tr.CAMS.index("cam_f0")
    n_rows, n_cols = rp.CONTENT_H, rp.FEAT_W

    tokens = list(sets)
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

    rows, X = [], {(f, i): [] for f in FEATS for i in INPUTS}
    t0 = time.time()
    for n_done, token in enumerate(tokens):
        scen = scenarios[token]
        surf = tr.surface_map(rmod, scen, tr.IDENT, channels=("CAM_F0",))["CAM_F0"]
        b = np.asarray(scen["anns"]["gt_boxes_world"], np.float64).reshape(-1, 7)
        names = np.asarray(scen["anns"].get("gt_names", []), dtype=str)
        dist = np.linalg.norm(to_ego(b[:, :2], scen["ego_heading"]), axis=1) if len(b) else np.zeros(0)
        frac, any_box, d = patch_labels(surf["obj"], surf["kind"] == tr.KIND_BOX, names == "vehicle", dist, tr.CROP,
                                        n_rows, n_cols)
        pos, neg = sample_patches(frac, any_box, token)
        idx = np.concatenate([pos, neg])
        if not len(idx):
            continue
        with torch.no_grad():
            for inp, raster in (("real", False), ("c1", True)):
                f = features(token, raster)
                imf, vit = rp.encode(model, f, want_vit=True)
                fpn = rp.strip_embeds(model, imf[0])[slot, :, 0].reshape(rp.FEAT_H, rp.FEAT_W, -1)[:n_rows]
                v = vit[slot, :, 0].reshape(rp.FEAT_H, rp.FEAT_W, -1)[:n_rows]
                X[("fpn", inp)].append(fpn.reshape(n_rows * n_cols, -1)[idx].float().cpu().numpy())
                X[("vit", inp)].append(v.reshape(n_rows * n_cols, -1)[idx].float().cpu().numpy())
        rows.append(pd.DataFrame({"token": token, "log": logs[token], "set": sets[token], "patch": idx,
                                  "y": np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)],
                                  "dist": d.reshape(-1)[idx], "veh_frac": frac.reshape(-1)[idx]}))
        if n_done % 25 == 0:
            print(f"{n_done}/{len(tokens)} tokens, {time.time() - t0:.0f}s", flush=True)
    data = pd.concat(rows, ignore_index=True)
    X = {k: np.concatenate(v) for k, v in X.items()}
    print(f"patches: {len(data)} (positives {int(data['y'].sum())}); by set {data['set'].value_counts().to_dict()}",
          flush=True)

    out_rows, preds = [], []
    fit_eval(data, X, out_rows, preds, args.label)
    mets = pd.DataFrame(out_rows)
    mets.to_csv(out_dir / "metrics.csv", index=False)
    wide = data.copy()
    for p in preds:
        key = f"{p['feat'].iloc[0]}_{p['input'].iloc[0]}"
        wide[f"p_{key}"] = p["p"].to_numpy()
        wide[f"d_{key}"] = p["dhat"].to_numpy()
    wide[wide["set"] != "train"].to_csv(out_dir / "preds.csv", index=False)
    print(mets.to_string(index=False))


if __name__ == "__main__":
    main()

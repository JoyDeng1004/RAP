#!/usr/bin/env python3
"""Audit cache shapes, camera validity, blank images, and projection centers.

When CUDA is available, also compare DINOv3 features across cache sources.
"""
import argparse
import gzip
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

CAMS = ["b0", "f0", "l0", "r0"]


def load_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def sample_tokens(cache, n, seed, logs=None):
    rng = random.Random(seed)
    logs = logs or sorted(p.name for p in cache.iterdir() if p.is_dir())
    token_dirs = []
    for log in rng.sample(logs, min(len(logs), n)):
        toks = [t for t in (cache / log).iterdir() if (t / "rap_feature.gz").is_file()]
        if toks:
            token_dirs.append(rng.choice(toks))
    return token_dirs


def cam_centres(l2i):
    # lidar2img = S @ K @ [R|t]; the centre c satisfies P[:3,:3] c + P[:3,3] = 0.
    P = np.asarray(l2i, dtype=np.float64)
    return np.stack([-np.linalg.solve(p[:3, :3], p[:3, 3]) for p in P])


def blank(img):
    # Check channels separately and exclude padding from normalized images.
    return bool(img[:, :400].float().flatten(1).std(1).max() < 1e-4)


def probe_cache(name, cache, token_dirs):
    rec = {"n": len(token_dirs), "shapes": {}, "camera_valid": 0,
           "blank_rendered": [0] * 4, "blank_real": [0] * 4,
           "centre_rendered": [], "centre_real": [], "traj_shape": None, "ego_status_shape": None}
    for td in token_dirs:
        feat = load_gz(td / "rap_feature.gz")
        tgt = load_gz(td / "rap_target.gz")
        if not rec["shapes"]:
            rec["shapes"] = {k: list(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in feat.items()}
            rec["traj_shape"] = list(tgt["trajectory"].shape)
            rec["ego_status_shape"] = list(feat["ego_status"].shape)
        rec["camera_valid"] += int(bool(feat["camera_valid"]))
        for i in range(4):
            rec["blank_rendered"][i] += blank(feat["rendered_camera_feature"][i])
            rec["blank_real"][i] += blank(feat["camera_feature"][i])
        rec["centre_rendered"].append(cam_centres(feat["rendered_lidar2img"]))
        rec["centre_real"].append(cam_centres(feat["lidar2img"]))
    n = max(rec["n"], 1)
    rec["camera_valid_rate"] = rec.pop("camera_valid") / n
    for k in ("blank_rendered", "blank_real"):
        rec[k] = {c: v / n for c, v in zip(CAMS, rec[k])}
    for k in ("centre_rendered", "centre_real"):
        arr = np.stack(rec[k])
        rec[k] = {c: {"mean": arr[:, i].mean(0).round(3).tolist(), "std": arr[:, i].std(0).round(3).tolist()}
                  for i, c in enumerate(CAMS)}
    return rec


@torch.no_grad()
def dino_probe(named_dirs, ckpt, n_dino):
    import timm
    model = timm.create_model("vit_huge_plus_patch16_dinov3_qkvb", pretrained=True, num_classes=0,
                              pretrained_cfg_overlay=dict(file=ckpt)).cuda().eval().half()
    pooled, out = {}, {}
    for name, dirs in named_dirs.items():
        for branch, key in (("rendered", "rendered_camera_feature"), ("real", "camera_feature")):
            feats = []
            for td in dirs[:n_dino]:
                feat = load_gz(td / "rap_feature.gz")
                if branch == "real" and not bool(feat["camera_valid"]):
                    continue
                img = feat[key].cuda().half()
                tok = model.forward_features(img)
                H, W = img.shape[-2:]
                out.setdefault(f"{name}/{branch}", {
                    "input": list(img.shape), "tokens": list(tok.shape),
                    "expected_tokens": 5 + (H // 16) * (W // 16)})
                assert torch.isfinite(tok).all(), f"non-finite DINO output in {td}"
                feats.append(tok[:, 5:].float().mean(1))
            if feats:
                pooled[f"{name}/{branch}"] = torch.stack(feats).mean(0)
    keys = sorted(pooled)
    cos = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sim = torch.nn.functional.cosine_similarity(pooled[a], pooled[b], dim=-1)
            cos[f"{a} vs {b}"] = {c: round(float(s), 3) for c, s in zip(CAMS, sim)}
    return {"shapes": out, "cosine_per_cam": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap", default=".")
    ap.add_argument("--n", type=int, default=300, help="sampled tokens per cache (one per log)")
    ap.add_argument("--n-dino", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/step3_probe.json")
    args = ap.parse_args()
    rap = Path(args.rap).resolve()
    caches = {
        "A_nativerig": rap / "cache/rap_nusc_nativerig",
        "C_navsimrig": rap / "cache/rap_nusc_navsimrig",
        "D_rap_ego": rap / "cache/rap_ego",
    }
    named_dirs = {k: sample_tokens(v, args.n, args.seed) for k, v in caches.items()}
    # D mixes paired and raster-only logs; split so the real-branch stats are not diluted.
    d_valid = [td for td in named_dirs["D_rap_ego"] if bool(load_gz(td / "rap_feature.gz")["camera_valid"])]
    named_dirs["D_paired"] = d_valid
    report = {k: probe_cache(k, caches.get(k, caches["D_rap_ego"]), v) for k, v in named_dirs.items()}
    if torch.cuda.is_available() and args.n_dino > 0:
        ckpt = str(rap / "ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
        report["dino"] = dino_probe({k: named_dirs[k] for k in ("A_nativerig", "C_navsimrig", "D_paired")},
                                    ckpt, args.n_dino)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

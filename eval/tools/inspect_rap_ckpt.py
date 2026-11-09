"""Check whether a RAP checkpoint loads into this fork's RAPModel (timm DINOv3). Read-only, CPU.

Official RAP (vita-epfl/RAP 5fd8630a) builds the backbone with HF transformers
(AutoModel dinov3-vith16plus); this fork uses timm. Parameter names differ, so
the backbone is matched by exact tensor content instead of by name. If every
backbone weight in the checkpoint equals a pretrained timm weight, the backbone
was frozen during training and can be skipped when loading.

    python inspect_rap_ckpt.py --ckpt <file.ckpt> --rap-root $RAP --dino-ckpt <dinov3 .pth>
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import torch

BB = "_backbone.img_backbone."


def digest(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().float().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rap-root", required=True)
    ap.add_argument("--dino-ckpt", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    print("[top] keys:", sorted(ck.keys()) if isinstance(ck, dict) else type(ck))
    for k in ("epoch", "global_step", "pytorch-lightning_version"):
        if isinstance(ck, dict) and k in ck:
            print(f"[top] {k}: {ck[k]}")
    hp = ck.get("hyper_parameters") if isinstance(ck, dict) else None
    if hp:
        print("[top] hyper_parameters keys:", list(hp)[:30])
    sd = ck.get("state_dict", ck)

    init_keys = [k for k in sd if k.endswith("init_feature.weight")]
    assert len(init_keys) == 1, init_keys
    prefix = init_keys[0][: -len("init_feature.weight")]
    rows = sd[init_keys[0]].shape[0]
    num_poses = rows // 64
    horizon = num_poses * 0.5
    print(f"[layout] prefix='{prefix}' init_feature={tuple(sd[init_keys[0]].shape)} -> num_poses={num_poses} time_horizon={horizon}")
    print("[layout] key groups:", Counter(".".join(k.split(".")[:3]) for k in sd).most_common(12))
    outside = [k for k in sd if not k.startswith(prefix)]
    print(f"[layout] keys outside prefix: {len(outside)} {outside[:5]}")

    ck_model = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    ck_bb = {k: v for k, v in ck_model.items() if k.startswith(BB)}
    print(f"[backbone] ckpt backbone tensors: {len(ck_bb)}; sample names: {list(ck_bb)[:4]}")
    print(f"[backbone] ckpt dtypes: {Counter(str(v.dtype) for v in ck_bb.values())}")

    sys.path.append(str(Path(args.rap_root) / "navsim" / "agents"))
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from rap_dino.navsim_config import RAPConfig
    from rap_dino.rap_model import RAPModel

    model = RAPModel(RAPConfig(
        trajectory_sampling=TrajectorySampling(time_horizon=horizon, interval_length=0.5),
        dino_ckpt_path=args.dino_ckpt, distill_feature=True, pdm_scorer=True,
    ))
    msd = model.state_dict()

    # Everything except the backbone must match by name and shape.
    m_rest = {k: v for k, v in msd.items() if not k.startswith(BB)}
    c_rest = {k: v for k, v in ck_model.items() if not k.startswith(BB)}
    missing = sorted(set(m_rest) - set(c_rest))
    unexpected = sorted(set(c_rest) - set(m_rest))
    shape_bad = [(k, tuple(c_rest[k].shape), tuple(m_rest[k].shape))
                 for k in set(m_rest) & set(c_rest) if c_rest[k].shape != m_rest[k].shape]
    print(f"[heads] model={len(m_rest)} ckpt={len(c_rest)} missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_bad)}")
    for name, lst in (("missing", missing), ("unexpected", unexpected), ("shape", shape_bad)):
        if lst:
            print(f"  {name}: {lst[:10]}")

    # Backbone: name-independent content match against timm's pretrained weights.
    # timm fuses q/k/v into one qkv tensor while HF keeps q_proj/k_proj/v_proj, so also index the thirds.
    timm_digests = set()
    for k, v in msd.items():
        if not k.startswith(BB):
            continue
        timm_digests.add(digest(v))
        if "qkv" in k and v.shape[0] % 3 == 0:
            timm_digests.update(digest(c) for c in v.chunk(3, dim=0))
    unmatched = [(k, tuple(v.shape)) for k, v in ck_bb.items() if digest(v) not in timm_digests]
    weights = {k: v for k, v in ck_bb.items() if v.dim() >= 2}
    unmatched_w = [(k, s) for k, s in unmatched if k in weights]
    print(f"[backbone] exact matches with pretrained timm weights: {len(ck_bb) - len(unmatched)}/{len(ck_bb)} "
          f"(>=2-D weights: {len(weights) - len(unmatched_w)}/{len(weights)})")
    for k, s in unmatched[:20]:
        print(f"  unmatched: {k} {s}")

    heads_ok = not missing and not unexpected and not shape_bad
    names_same = set(ck_bb) == {k for k in msd if k.startswith(BB)}
    if heads_ok and names_same:
        verdict = "STRICT_LOADABLE (same backbone names; current agent loads it as-is)"
    elif heads_ok and ck_bb and not unmatched_w:
        verdict = ("LOAD_WITHOUT_BACKBONE (heads match; every >=2-D backbone weight equals pretrained, keep timm weights"
                   + ("; review the unmatched 1-D tensors above)" if unmatched else ")"))
    elif heads_ok:
        verdict = "NEEDS_BACKBONE_REMAP (heads match; backbone weights differ from pretrained)"
    else:
        verdict = "INCOMPATIBLE_HEADS (see missing/unexpected/shape above)"
    print(f"\nVERDICT: {verdict}  time_horizon={horizon}")


if __name__ == "__main__":
    main()

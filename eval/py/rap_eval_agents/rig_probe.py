"""Exp-0 / T1 step 8: forward-only probes inside RAPModel (no weights change).

RAPModel.forward is split into ``encode`` (frozen DINOv3 + FPN + camera/level
embeddings) and ``decode`` (4 shared Traj_refiner rounds + scorer), so that

- 8a fixed candidates: the scorer re-ranks the raw-rig proposals under the new rig,
  either reading the new images only in the last round (``inject_last``) or in every
  round with the raw proposals teacher-forced (``force``);
- 8b / 8c feature translation: the decoder runs on the crop-control features moved
  by the shift (``translate``), i.e. what a perfectly translation-equivariant backbone
  would output, with the shifted calibration.

``decode(encode(x))`` must equal ``model(x, None, return_score=True)``; the runner
checks this on every token.

For the step-8 figures it also records, without changing any output, where the
spatial cross-attention samples (``record_sca`` / ``sca_stats``), how far the
trajectory queries and proposals drift per round (``query_cos`` / ``proposal_ade``)
and per-position feature similarity maps (``cos_map``).

``fill_as_outside`` is the one intervention on the decoder: SCA sampling locations that
land on black fill or on RAP's padding row are moved outside the image, so they add
exactly 0, as they did when the same 3D points projected outside the raw image.
"""
import contextlib
import math
import sys
from typing import Dict, List, Optional, Tuple

import torch

FEAT_H, FEAT_W = 28, 48  # RAP input 432 x 768 padded to 448 x 768, 16 px patches
CONTENT_H = 27  # rows 0..26 hold image content; row 27 is RAP's bottom padding


def encode(model, feats: Dict[str, torch.Tensor], want_vit: bool = False):
    """``model._backbone`` as RAPModel.forward calls it; optionally also the raw ViT patch tokens."""
    bb = model._backbone
    store = {}
    if want_vit:
        orig = bb.img_backbone.forward_features

        def forward_features(x):
            out = orig(x)
            store["vit"] = out
            return out

        bb.img_backbone.forward_features = forward_features  # instance attribute shadows the method
    try:
        image_feature = bb(feats["camera_feature"], img_metas=feats)
    finally:
        if want_vit:
            del bb.img_backbone.forward_features
    vit = None
    if want_vit:
        x = store["vit"]  # (B*N, extra + 28*48, C), same slicing as ImgEncoder._tokens_to_map
        vit = x[:, x.shape[1] - FEAT_H * FEAT_W:].unsqueeze(2)  # (N, H*W, 1, C) like feat_flatten
    return image_feature, vit


def with_calibration(image_feature, feat_flatten: torch.Tensor, feats: Dict[str, torch.Tensor]):
    """Same spatial layout, other features and / or other img_metas (lidar2img)."""
    _, spatial_shapes, level_start_index, _ = image_feature
    return feat_flatten, spatial_shapes, level_start_index, {"img_metas": feats}


def strip_embeds(model, feat_flatten: torch.Tensor) -> torch.Tensor:
    """FPN output without the constant per-camera / level embeddings (for similarity stats)."""
    bb = model._backbone
    return feat_flatten - bb.cams_embeds[:, None, None, :] - bb.level_embeds[None, None, 0, :]


def decode(model, feats: Dict[str, torch.Tensor], image_feature,
           force: Optional[List[torch.Tensor]] = None,
           inject_last: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, object]:
    """RAPModel.forward after the backbone, return_score=True path.

    force: one proposal tensor per round; each Bev_refiner reads the images at these
        proposals instead of the ones decoded from its own queries.
    inject_last: (queries entering the last round, proposals of the last round), taken from
        another run; only the last Bev_refiner and the scorer run.
    """
    heads = model._trajectory_head
    if not all(getattr(h, "traj_bev", False) for h in heads):
        raise RuntimeError("probe assumes traj_bev=True (every round reads the images)")
    ego = feats["ego_status"][:, -1]
    if model.b2d:
        ego = ego.clone()
        ego[:, 1:3] = 0
    bev_in: List[torch.Tensor] = []
    if inject_last is not None:
        bev3, p3 = inject_last
        props = [p3]
        bev = heads[-1].Bev_refiner(p3, bev3, image_feature)
    else:
        bev = model.hist_encoding(ego)[:, None] + model.init_feature.weight[None]
        props = []
        for i, refine in enumerate(heads):
            bev_in.append(bev)
            if force is None:
                bev, props = refine(bev, props, image_feature)
            else:
                props.append(force[i])
                bev = refine.Bev_refiner(force[i], bev, image_feature)
    proposals = props[-1]
    pred_logit, pred_logit2 = model.scorer(proposals, bev)[:2]
    if pred_logit2 is not None:
        score = (torch.sigmoid(pred_logit) + torch.sigmoid(pred_logit2))[:, :, -1] / 2
    else:
        score = torch.sigmoid(pred_logit)[:, :, -1]
    return {"proposals": proposals, "score": score, "props": props, "bev_in": bev_in, "bev_out": bev}


def content_range(shift: float, n_content: int) -> Tuple[int, int]:
    """Rows (or columns) of the crop-control map whose 16 px patch is entirely content that stays
    in view after moving by ``shift`` patches (> 0: down / right)."""
    lo = math.ceil(max(0.0, -shift) - 1e-9)
    hi = math.floor(min(n_content - 1.0, n_content - 1.0 - shift) + 1e-9)
    return lo, hi


def translate(crop_ff: torch.Tensor, shifted_ff: torch.Tensor, dy: float, dx: float,
              n_rows: int = CONTENT_H, n_cols: int = FEAT_W) -> Tuple[torch.Tensor, torch.Tensor]:
    """out[r, c] = crop(r - dy, c - dx) (bilinear) wherever every tap is a full-content patch of the
    crop control, else shifted[r, c] (the black band, padding row and partial-content patches of the
    real shifted image). Layout (N, H*W, B, C) as feat_flatten. Returns (out, valid (H, W) bool)."""
    n, hw, b, ch = crop_ff.shape
    assert hw == FEAT_H * FEAT_W and shifted_ff.shape == crop_ff.shape
    dev = crop_ff.device
    r = torch.arange(FEAT_H, device=dev, dtype=torch.float64)[:, None].expand(FEAT_H, FEAT_W)
    c = torch.arange(FEAT_W, device=dev, dtype=torch.float64)[None, :].expand(FEAT_H, FEAT_W)
    sr, sc = r - dy, c - dx
    r0, c0 = torch.floor(sr), torch.floor(sc)
    fr, fc = sr - r0, sc - c0
    r1 = torch.where(fr > 1e-9, r0 + 1, r0)
    c1 = torch.where(fc > 1e-9, c0 + 1, c0)
    rlo, rhi = content_range(dy, n_rows)
    clo, chi = content_range(dx, n_cols)
    valid = (r0 >= rlo) & (r1 <= rhi) & (c0 >= clo) & (c1 <= chi) & (r < n_rows)

    m = crop_ff.reshape(n, FEAT_H, FEAT_W, b * ch)
    ri0, ri1 = r0.clamp(0, FEAT_H - 1).long(), r1.clamp(0, FEAT_H - 1).long()
    ci0, ci1 = c0.clamp(0, FEAT_W - 1).long(), c1.clamp(0, FEAT_W - 1).long()
    wr, wc = fr.to(m.dtype)[..., None], fc.to(m.dtype)[..., None]
    interp = ((1 - wr) * (1 - wc) * m[:, ri0, ci0] + (1 - wr) * wc * m[:, ri0, ci1]
              + wr * (1 - wc) * m[:, ri1, ci0] + wr * wc * m[:, ri1, ci1])
    out = torch.where(valid[None, ..., None], interp, shifted_ff.reshape(n, FEAT_H, FEAT_W, b * ch))
    return out.reshape(n, hw, b, ch), valid


def masked_cos(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity over the valid (H, W) positions, per camera -> (N,)."""
    n = a.shape[0]
    a = a.reshape(n, FEAT_H, FEAT_W, -1)[:, valid].float()
    b = b.reshape(n, FEAT_H, FEAT_W, -1)[:, valid].float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(1)


def masked_rel_l2(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """||a - b|| / ||b|| over the valid positions, per camera -> (N,)."""
    n = a.shape[0]
    a = a.reshape(n, FEAT_H, FEAT_W, -1)[:, valid].float()
    b = b.reshape(n, FEAT_H, FEAT_W, -1)[:, valid].float()
    return (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-12)



# ---------- step-8 figure data ----------

N_POSE_TRAJ = 8  # waypoints NAVSIM scores (4 s at 0.5 s); RAP predicts more
RAP_MEAN = (123.675, 116.28, 103.53)
RAP_STD = (58.395, 57.12, 57.375)
SCA_CATS = ("out", "band", "pad", "content")  # where an SCA sampling location falls


def cos_map(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor, slot: int) -> torch.Tensor:
    """(H, W) cosine similarity of camera ``slot``; NaN outside ``valid``."""
    a = a[slot].reshape(FEAT_H, FEAT_W, -1).float()
    b = b[slot].reshape(FEAT_H, FEAT_W, -1).float()
    m = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return torch.where(valid, m, torch.full_like(m, float("nan")))


def query_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine over the (proposal, waypoint) queries, (B, Q, C). Queries are indexed by
    candidate slot, not by image position, so two rigs compare directly."""
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean())


def proposal_ade(a: torch.Tensor, b: torch.Tensor) -> float:
    """Slot-matched mean xy distance over the first 8 waypoints, (B, 64, T, 3)."""
    return float((a[..., :N_POSE_TRAJ, :2] - b[..., :N_POSE_TRAJ, :2]).norm(dim=-1).mean())


def pool_cover(ref: torch.Tensor, new: torch.Tensor) -> float:
    """Mean over ``ref`` candidates of the ADE to the closest ``new`` candidate (slot-free)."""
    r, n = ref[0, :, :N_POSE_TRAJ, :2], new[0, :, :N_POSE_TRAJ, :2]
    d = (r[:, None] - n[None]).norm(dim=-1).mean(-1)  # (64, 64)
    return float(d.min(1).values.mean())


def _sca_module(model):
    for m in model.modules():
        if type(m).__name__ == "MSDeformableAttention3D":
            return sys.modules[type(m).__module__]
    raise RuntimeError("no MSDeformableAttention3D in the model")


@contextlib.contextmanager
def record_sca(model):
    """Collect, in call order, ("ps", reference_points_cam, bev_mask) from every point_sampling call
    and ("attn", sampling_locations, attention_weights) from every SCA deformable-attention call.
    One refiner round = one "ps" followed by one "attn" per encoder layer. Outputs are unchanged."""
    records: List[tuple] = []
    mod = _sca_module(model)
    orig_fp32, orig_py = mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch

    class _Rec:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            records.append(("attn", sampling_locations.detach(), attention_weights.detach()))
            return orig_fp32.apply(value, spatial_shapes, level_start_index, sampling_locations,
                                   attention_weights, im2col_step)

    def _py(value, spatial_shapes, sampling_locations, attention_weights):
        records.append(("attn", sampling_locations.detach(), attention_weights.detach()))
        return orig_py(value, spatial_shapes, sampling_locations, attention_weights)

    encoders = {id(h.Bev_refiner.bev_decoder): h.Bev_refiner.bev_decoder for h in model._trajectory_head}
    for enc in encoders.values():
        orig_ps = enc.point_sampling

        def ps(reference_points, img_metas, _orig=orig_ps):
            out = _orig(reference_points, img_metas)
            records.append(("ps", out[0].detach(), out[1].detach()))
            return out

        enc.point_sampling = ps
    mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch = _Rec, _py
    try:
        yield records
    finally:
        mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch = orig_fp32, orig_py
        for enc in encoders.values():
            del enc.point_sampling


def black_mask(feats: Dict[str, torch.Tensor], slot: int) -> torch.Tensor:
    """(448, 768) pixels of RAP's normalised input that are exactly black (rig fill / dropped camera)."""
    img = feats["camera_feature"][0, slot].float()  # (3, 448, 768), normalised RGB
    black = torch.tensor([-m / s for m, s in zip(RAP_MEAN, RAP_STD)], device=img.device)[:, None, None]
    return ((img - black).abs() < 1e-3).all(0)


def sca_stats(records: List[tuple], feats: Dict[str, torch.Tensor], n_poses: int, slot: int = 1) -> torch.Tensor:
    """Per round and waypoint, for camera ``slot``: [fraction of the 64 candidates whose query attends
    this camera, then attention mass (summed over heads and points, averaged over those queries and
    the encoder layers) falling outside the image / on black fill / on the padding row / on content].
    Returns (rounds, n_poses, 1 + len(SCA_CATS)); NaN mass where no query attends."""
    # point_sampling normalises by img_shape, which PadMultiViewImage sets to the padded 448 x 768,
    # so normalised (x, y) map onto the whole padded image; rows >= 432 are RAP's bottom padding.
    h_pad, w_pad = feats["camera_feature"].shape[-2:]
    h_img = CONTENT_H * 16
    black = black_mask(feats, slot)
    rounds, cur = [], None
    for rec in records:
        if rec[0] == "ps":
            cur = {"bev_mask": rec[2], "attn": []}
            rounds.append(cur)
        else:
            cur["attn"].append(rec[1:])
    out = torch.full((len(rounds), n_poses, 1 + len(SCA_CATS)), float("nan"))
    for r, rd in enumerate(rounds):
        mask = rd["bev_mask"][slot][0]  # (num_query, D), batch sample 0 as in SpatialCrossAttention
        idx = mask.sum(-1).nonzero().squeeze(-1)
        pose = idx % n_poses
        n_query = mask.shape[0]
        part = torch.bincount(pose, minlength=n_poses).float() / (n_query // n_poses)
        out[r, :, 0] = part.cpu()
        if len(idx) == 0:
            continue
        mass = torch.zeros(len(idx), len(SCA_CATS), device=mask.device)
        for loc, w in rd["attn"]:
            loc, w = loc[slot, :len(idx)].float(), w[slot, :len(idx)].float()  # (n, heads, levels, P, 2), (n, heads, levels, P)
            x, y = loc[..., 0], loc[..., 1]
            out_img = (x < 0) | (x > 1) | (y < 0) | (y > 1)
            py = (y * h_pad).clamp(0, h_pad - 1)
            px = (x * w_pad).clamp(0, w_pad - 1)
            pad = ~out_img & (py >= h_img)
            on_black = ~out_img & ~pad & black[py.long(), px.long()]
            content = ~out_img & ~pad & ~on_black
            heads = w.shape[1]
            for c, m in enumerate((out_img, on_black, pad, content)):  # order of SCA_CATS
                mass[:, c] += (w * m).flatten(1).sum(1) / heads
        mass /= max(len(rd["attn"]), 1)
        sums = torch.zeros(n_poses, len(SCA_CATS), device=mask.device).index_add_(0, pose, mass)
        cnt = torch.bincount(pose, minlength=n_poses).float()[:, None]
        out[r, :, 1:] = torch.where(cnt > 0, sums / cnt.clamp_min(1), torch.full_like(sums, float("nan"))).cpu()
    return out


def fill_mask(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """(num_cams, 448, 768): black fill of each camera, plus RAP's padding rows (>= 432)."""
    n = feats["camera_feature"].shape[1]
    m = torch.stack([black_mask(feats, i) for i in range(n)])
    m[:, CONTENT_H * 16:] = True
    return m


def push_outside(loc: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Sampling locations (bs * num_cams, Q, heads, levels, P, 2), normalised to the padded image,
    with every in-image location whose nearest pixel is in ``masks`` (num_cams, H, W) set to (-1, -1):
    outside the image, where MSDeformAttn (CUDA kernel and grid_sample fallback) samples 0."""
    n_cam, h, w = masks.shape
    x, y = loc[..., 0], loc[..., 1]
    inside = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    px = (x * w).clamp(0, w - 1).long()
    py = (y * h).clamp(0, h - 1).long()
    cam = (torch.arange(loc.shape[0], device=loc.device) % n_cam).view(-1, *([1] * (loc.dim() - 2)))
    hit = inside & masks[cam.expand_as(px), py, px]
    return torch.where(hit[..., None], torch.full_like(loc, -1.0), loc)


@contextlib.contextmanager
def fill_as_outside(model, feats: Dict[str, torch.Tensor]):
    """Inside the block, SCA ignores black fill and padding (see push_outside). Enter record_sca first
    to record the moved locations."""
    masks = fill_mask(feats)
    mod = _sca_module(model)
    orig_fp32, orig_py = mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch

    class _Oob:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return orig_fp32.apply(value, spatial_shapes, level_start_index,
                                   push_outside(sampling_locations, masks), attention_weights, im2col_step)

    def _py(value, spatial_shapes, sampling_locations, attention_weights):
        return orig_py(value, spatial_shapes, push_outside(sampling_locations, masks), attention_weights)

    mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch = _Oob, _py
    try:
        yield
    finally:
        mod.MultiScaleDeformableAttnFunction_fp32, mod.multi_scale_deformable_attn_pytorch = orig_fp32, orig_py

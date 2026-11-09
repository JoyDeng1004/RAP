"""T2 statistics on records from ``rig_probe.record_sca`` (forward-only, outputs unchanged).

Round 1, layer 0 of the trajectory refiner is rig-independent up to its reference points: its queries
come from ``init_feature`` + ego status only (no image read yet), so the sampling offsets and attention
weights of a query are identical under two rigs, and only the projected reference points differ.
``r1l0_flip`` pairs the sampling points of the same query under the reference (C1) and the changed rig
and measures the attention mass that moves between "outside / padding" (RAP's learned no-read sink)
and image content. ``anchor_cos`` compares image features at the projections of the same 3D pillar
points (round 1 reference points) under the two rigs. ``surface_cos`` compares them instead at pixels
that show the same surface point under the two rigs (``t2_render.surface_pairs``): the encoder's view
of one surface, free of which surface a fixed 3D point happens to land on.
"""
from typing import Dict, List, Tuple

import torch

from . import rig_probe as rp
from . import t2_render as tr

PAD_FROM = rp.CONTENT_H * 16  # rows >= 432 of the padded 448 x 768 input are RAP's bottom padding


def round1(records: List[tuple]) -> Tuple[tuple, tuple]:
    """(ps, first attn) of refiner round 1: ps = (ref_cam (N,B,Q,D,2), bev_mask (N,B,Q,D)),
    attn = (loc (B*N, L, heads, levels, P, 2), weight (B*N, L, heads, levels, P))."""
    ps = next((r for r in records if r[0] == "ps"), None)
    if ps is None:
        raise RuntimeError("no point_sampling record")
    i = records.index(ps)
    attn = next((r for r in records[i + 1:] if r[0] == "attn"), None)
    if attn is None:
        raise RuntimeError("no SCA record after round 1 point_sampling")
    return ps[1:], attn[1:]


def _classes(loc: torch.Tensor, h_pad: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sink, content) masks of normalised sampling locations: sink = outside the image or on RAP's
    bottom padding rows, content = inside the image rows."""
    x, y = loc[..., 0], loc[..., 1]
    out = (x < 0) | (x > 1) | (y < 0) | (y > 1)
    pad = ~out & (y * h_pad >= PAD_FROM)
    return out | pad, ~out & ~pad


def _query_mass(w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Attention mass of the points in ``m`` per query, summed over levels / points, averaged over heads."""
    heads = w.shape[1]
    return (w * m).flatten(1).sum(1) / heads


def r1l0_flip(rec_ref: List[tuple], rec_new: List[tuple], feats_new: Dict[str, torch.Tensor],
              n_poses: int, slot: int = 1, atol: float = 1e-5) -> Dict[str, object]:
    """Round 1 / layer 0 sampling of camera ``slot`` under the reference and the changed rig."""
    (_, m_ref), (loc_ref, w_ref) = round1(rec_ref)
    (_, m_new), (loc_new, w_new) = round1(rec_new)
    mask_ref, mask_new = m_ref[slot][0], m_new[slot][0]  # (Q, D), batch sample 0 as SCA uses
    q_ref = mask_ref.sum(-1).nonzero().squeeze(-1)
    q_new = mask_new.sum(-1).nonzero().squeeze(-1)
    n_query = mask_ref.shape[0]
    pos_ref = torch.full((n_query,), -1, dtype=torch.long, device=q_ref.device)
    pos_new = pos_ref.clone()
    pos_ref[q_ref] = torch.arange(len(q_ref), device=q_ref.device)
    pos_new[q_new] = torch.arange(len(q_new), device=q_new.device)
    common = ((pos_ref >= 0) & (pos_new >= 0)).nonzero().squeeze(-1)
    out: Dict[str, object] = {
        "n_query_ref": int(len(q_ref)), "n_query_new": int(len(q_new)), "n_query_common": int(len(common)),
        "lost_query_frac": float(((pos_ref >= 0) & (pos_new < 0)).sum()) / max(len(q_ref), 1),
        "gained_query_frac": float(((pos_new >= 0) & (pos_ref < 0)).sum()) / max(len(q_ref), 1),
    }
    nan_pose = torch.full((n_poses,), float("nan"))
    if len(common) == 0:
        out.update(flip_in=float("nan"), flip_in_obj=float("nan"), flip_out=float("nan"),
                   sink_ref=float("nan"), sink_new=float("nan"), _flip_in_pose=nan_pose.numpy())
        return out
    lr, wr = loc_ref[slot, pos_ref[common]].float(), w_ref[slot, pos_ref[common]].float()
    ln, wn = loc_new[slot, pos_new[common]].float(), w_new[slot, pos_new[common]].float()
    if not torch.allclose(wr, wn, rtol=0, atol=atol):
        raise AssertionError(f"round 1 / layer 0 attention weights differ across rigs "
                             f"(max {float((wr - wn).abs().max()):.2e}): the query is not rig-independent")
    h_pad = feats_new["camera_feature"].shape[-2]
    w_pad = feats_new["camera_feature"].shape[-1]
    sink_r, cont_r = _classes(lr, h_pad)
    sink_n, cont_n = _classes(ln, h_pad)
    black = rp.black_mask(feats_new, slot)  # raster background in the changed rig's input
    px = (ln[..., 0] * w_pad).clamp(0, w_pad - 1).long()
    py = (ln[..., 1] * h_pad).clamp(0, h_pad - 1).long()
    obj = cont_n & ~black[py, px]  # lands on a drawn raster primitive
    flip_in = _query_mass(wn, sink_r & cont_n)
    pose = common % n_poses
    cnt = torch.bincount(pose, minlength=n_poses).float()
    per_pose = torch.zeros(n_poses, device=flip_in.device).index_add_(0, pose, flip_in)
    out.update(
        flip_in=float(flip_in.mean()),
        flip_in_obj=float(_query_mass(wn, sink_r & obj).mean()),
        flip_out=float(_query_mass(wn, cont_r & sink_n).mean()),
        sink_ref=float(_query_mass(wr, sink_r).mean()),
        sink_new=float(_query_mass(wn, sink_n).mean()),
        _flip_in_pose=torch.where(cnt > 0, per_pose / cnt.clamp_min(1), torch.full_like(per_pose, float("nan"))).cpu().numpy(),
    )
    return out


def _sample(feat_flatten: torch.Tensor, slot: int, xy: torch.Tensor) -> torch.Tensor:
    """Bilinear features of camera ``slot`` at normalised (x, y) on the padded input, the way
    MSDeformAttn samples (grid_sample, align_corners=False, zero outside). xy (n, 2) -> (n, C)."""
    fm = feat_flatten[slot].reshape(rp.FEAT_H, rp.FEAT_W, -1).permute(2, 0, 1)[None].float()
    grid = (2 * xy.float() - 1).view(1, -1, 1, 2)
    s = torch.nn.functional.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return s[0, :, :, 0].T


def _on_drawn(feats: Dict[str, torch.Tensor], slot: int, xy: torch.Tensor) -> torch.Tensor:
    """True where normalised (x, y) on the padded input falls on a non-black raster pixel."""
    h_pad, w_pad = feats["camera_feature"].shape[-2:]
    black = rp.black_mask(feats, slot)
    px = (xy[..., 0] * w_pad).clamp(0, w_pad - 1).long()
    py = (xy[..., 1] * h_pad).clamp(0, h_pad - 1).long()
    return ~black[py, px]


def anchor_cos(model, rec_a: List[tuple], imf_a, rec_b: List[tuple], imf_b, slot: int = 1,
               feats_a=None, feats_b=None, n_poses: int = 0) -> Dict[str, float]:
    """Cosine of (embedding-free) image features at the projections of the same round-1 3D reference
    points under two inputs, over points that land on image content in both. With the raster inputs
    ``feats_a`` / ``feats_b`` also split into points on a drawn pixel in either input vs on black
    background in both. With ``n_poses`` also ``_pose_pt``: (4, n_poses) mean cosine per distinct
    reference point and waypoint (query q is waypoint q % n_poses), NaN where no pair."""
    (ref_a, m_a), _ = round1(rec_a)
    (ref_b, m_b), _ = round1(rec_b)
    ra, rb = ref_a[slot][0], ref_b[slot][0]  # (Q, D, 2)
    h_pad = rp.FEAT_H * 16
    in_a = (ra >= 0).all(-1) & (ra[..., 0] <= 1) & (ra[..., 1] * h_pad < PAD_FROM)
    in_b = (rb >= 0).all(-1) & (rb[..., 0] <= 1) & (rb[..., 1] * h_pad < PAD_FROM)
    both = m_a[slot][0] & m_b[slot][0] & in_a & in_b
    n = int(both.sum())
    if n == 0:
        out = {"anchor_cos": float("nan"), "anchor_n": 0}
        if feats_a is not None:
            out.update(anchor_cos_drawn=float("nan"), anchor_n_drawn=0, anchor_cos_bg=float("nan"))
        out.update({f"anchor_cos_p{i}": float("nan") for i in range(4)})
        if n_poses:
            out["_pose_pt"] = torch.full((4, n_poses), float("nan")).numpy()
        return out
    fa = _sample(rp.strip_embeds(model, imf_a[0]), slot, ra[both])
    fb = _sample(rp.strip_embeds(model, imf_b[0]), slot, rb[both])
    cos = torch.nn.functional.cosine_similarity(fa, fb, dim=-1)
    out = {"anchor_cos": float(cos.mean()), "anchor_n": n}
    if feats_a is not None:
        drawn = _on_drawn(feats_a, slot, ra[both]) | _on_drawn(feats_b, slot, rb[both])
        out["anchor_n_drawn"] = int(drawn.sum())
        out["anchor_cos_drawn"] = float(cos[drawn].mean()) if drawn.any() else float("nan")
        out["anchor_cos_bg"] = float(cos[~drawn].mean()) if (~drawn).any() else float("nan")
    # Official RAP's 16 reference points are 4 distinct 3D points, each repeated 4 times: anchor d is
    # corner d % 4 at height d % 4 (encoder.py:252-256), i.e. FL z=-2, FR z=0.67, RR z=3.33, RL z=6 m.
    k = torch.arange(ra.shape[1], device=ra.device).expand(ra.shape[0], -1)[both] % 4
    for i in range(4):
        sel = k == i
        out[f"anchor_cos_p{i}"] = float(cos[sel].mean()) if sel.any() else float("nan")
    if n_poses:
        pose = torch.arange(ra.shape[0], device=ra.device)[:, None].expand(-1, ra.shape[1])[both] % n_poses
        cell = k * n_poses + pose
        sums = torch.zeros(4 * n_poses, device=cos.device).index_add_(0, cell, cos.float())
        cnt = torch.bincount(cell, minlength=4 * n_poses).float()
        out["_pose_pt"] = torch.where(cnt > 0, sums / cnt.clamp_min(1), torch.full_like(sums, float("nan"))) \
            .view(4, n_poses).cpu().numpy()
    return out


def canvas_to_norm(uv) -> torch.Tensor:
    """Continuous raster canvas coordinates (n, 2) -> normalised (x, y) on RAP's padded input: the
    1080-row crop is resized to 432 x 768 and padded to 448 rows (x scale and y scale are both 0.4)."""
    uv = torch.as_tensor(uv, dtype=torch.float64)
    x = uv[:, 0] / tr.CANVAS_W
    y = (uv[:, 1] - tr.CROP) / (tr.CANVAS_H - 2 * tr.CROP) * PAD_FROM / (rp.FEAT_H * 16)
    return torch.stack([x, y], -1).float()


DEPTH_BINS = (10.0, 25.0)  # m: near / mid / far surface points (depth in the reference camera)


def surface_cos(model, imf_a, imf_b, pairs: Dict[str, object], slot: int = 1) -> Dict[str, float]:
    """Cosine of (embedding-free) features of camera ``slot`` at same-surface correspondences
    (a at uv_a, b at uv_b), sampled like ``anchor_cos``; plus the same pairs with b read at uv_a
    (surf_samepix_cos: what ignoring the correspondence would give), the mean correspondence shift in
    feature cells, and the cosine per surface kind and depth bin."""
    n = len(pairs["kind"])
    names = ["surf_cos", "surf_samepix_cos", "surf_shift_cells", "surf_cos_box", "surf_cos_map", "surf_cos_light"]
    names += [f"surf_cos_z{i}" for i in range(len(DEPTH_BINS) + 1)]
    out: Dict[str, float] = {"surf_n": n}
    if n == 0:
        out.update({k: float("nan") for k in names})
        out.update(surf_n_box=0, surf_n_map=0, surf_n_light=0)
        return out
    fa_map = rp.strip_embeds(model, imf_a[0])
    fb_map = rp.strip_embeds(model, imf_b[0])
    xa = canvas_to_norm(pairs["uv_a"]).to(fa_map.device)
    xb = canvas_to_norm(pairs["uv_b"]).to(fa_map.device)
    fa = _sample(fa_map, slot, xa)
    cos = torch.nn.functional.cosine_similarity(fa, _sample(fb_map, slot, xb), dim=-1).cpu()
    same = torch.nn.functional.cosine_similarity(fa, _sample(fb_map, slot, xa), dim=-1).cpu()
    shift = ((xb - xa) * torch.tensor([rp.FEAT_W, rp.FEAT_H], device=xa.device)).norm(dim=-1)
    out.update(surf_cos=float(cos.mean()), surf_samepix_cos=float(same.mean()), surf_shift_cells=float(shift.mean()))
    kind = torch.as_tensor(pairs["kind"])
    for name, k in (("box", tr.KIND_BOX), ("map", tr.KIND_MAP), ("light", tr.KIND_LIGHT)):
        sel = kind == k
        out[f"surf_n_{name}"] = int(sel.sum())
        out[f"surf_cos_{name}"] = float(cos[sel].mean()) if sel.any() else float("nan")
    zbin = torch.bucketize(torch.as_tensor(pairs["depth"]).float(), torch.tensor(DEPTH_BINS))
    for i in range(len(DEPTH_BINS) + 1):
        sel = zbin == i
        out[f"surf_cos_z{i}"] = float(cos[sel].mean()) if sel.any() else float("nan")
    return out


CELL_PX = 40  # one 16 px feature cell of RAP's 0.4x-resized input, in raster canvas pixels


def warp_swap(imf_new, imf_ref, pairs: Dict[int, Dict[str, object]], fill_rest: bool = False):
    """Copy of the backbone output ``imf_new`` in which, per camera slot, every content cell whose
    centre shows a surface also shown in the reference view takes the reference features at that
    surface (bilinear on ``imf_ref``): what a perfectly rig-invariant encoder of the same surface would
    give, with everything else (calibration, decoder) unchanged. ``pairs[slot]`` =
    ``t2_render.surface_pairs(surf_new, surf_ref, stride=CELL_PX)``, so uv_a are the new cell centres.
    ``fill_rest``: the remaining content cells (no surface, or none matched) take the reference cell at
    the same position. Embeddings are per slot, so both maps carry the same ones. Returns
    (image_feature tuple, {slot: fraction of the CONTENT_H x FEAT_W cells swapped by correspondence})."""
    ff_ref = imf_ref[0]
    ff = imf_new[0].clone()
    frac: Dict[int, float] = {}
    n_cells = rp.CONTENT_H * rp.FEAT_W
    for slot, pr in pairs.items():
        uv_a = torch.as_tensor(pr["uv_a"], dtype=torch.float64)
        uv_b = torch.as_tensor(pr["uv_b"], dtype=torch.float64)
        j = torch.floor(uv_a[:, 0] / CELL_PX).long()
        i = torch.floor((uv_a[:, 1] - tr.CROP) / CELL_PX).long()
        centre = torch.stack([(j + 0.5) * CELL_PX, tr.CROP + (i + 0.5) * CELL_PX], -1).double()
        if len(uv_a) and ((uv_a - centre).abs().max() > 1.0 or (i >= rp.CONTENT_H).any()):
            raise ValueError("warp_swap needs surface_pairs on the cell-centre grid (stride=CELL_PX)")
        # sample pixel = cell centre + half a pixel: read the reference at the same sub-pixel offset
        vals = _sample(ff_ref, slot, canvas_to_norm(uv_b + (centre - uv_a)).to(ff.device))
        cells = (i * rp.FEAT_W + j).to(ff.device)
        if fill_rest:
            content = torch.arange(n_cells, device=ff.device)
            ff[slot, content] = ff_ref[slot, content]
        ff[slot, cells, 0] = vals.to(ff.dtype)
        frac[slot] = len(cells) / n_cells
    return (ff,) + tuple(imf_new[1:]), frac

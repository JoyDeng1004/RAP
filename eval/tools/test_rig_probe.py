"""Self-test for rap_eval_agents.rig_probe.translate / content_range (torch on CPU, no data, no GPU).

    PYTHONPATH=eval/py python eval/tools/test_rig_probe.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
from rap_eval_agents import rig_probe as rp  # noqa: E402

H, W, CH = rp.FEAT_H, rp.FEAT_W, 5


def to_ff(m):  # (N, H, W, C) -> feat_flatten layout (N, H*W, 1, C)
    return m.reshape(m.shape[0], H * W, 1, m.shape[-1])


def test_content_range():
    assert rp.content_range(-7, 27) == (7, 26)
    assert rp.content_range(3, 27) == (0, 23)
    assert rp.content_range(-6.81, 27) == (7, 26)
    assert rp.content_range(3.38, 27) == (0, 22)
    assert rp.content_range(2.4, 48) == (0, 44)
    assert rp.content_range(0, 48) == (0, 47)


def test_integer_shift_recovers_exact_translation():
    """Equivariant backbone: shifted = crop moved by k rows. translate must reproduce it on every
    full-content position and keep the shifted map elsewhere (band + padding row)."""
    g = torch.Generator().manual_seed(0)
    for dy, dx in ((-7, 0), (-3, 0), (3, 0), (1, 0), (0, 2), (-1, -2)):
        crop = torch.randn(4, H, W, CH, generator=g)
        shifted = torch.randn(4, H, W, CH, generator=g)  # band / padding content: arbitrary
        ideal = shifted.clone()
        for r in range(rp.CONTENT_H):
            for c in range(W):
                sr, sc = r - dy, c - dx
                rlo, rhi = rp.content_range(dy, rp.CONTENT_H)
                clo, chi = rp.content_range(dx, W)
                if rlo <= sr <= rhi and clo <= sc <= chi:
                    ideal[:, r, c] = crop[:, sr, sc]
        out, valid = rp.translate(to_ff(crop), to_ff(shifted), dy, dx)
        assert torch.equal(out, to_ff(ideal)), (dy, dx)
        n_rows = (rp.content_range(dy, 27)[1] - rp.content_range(dy, 27)[0] + 1)
        n_cols = (rp.content_range(dx, W)[1] - rp.content_range(dx, W)[0] + 1)
        assert int(valid.sum()) == n_rows * n_cols, (dy, dx, int(valid.sum()))
        assert not valid[rp.CONTENT_H:].any()  # padding row never comes from the crop map
        assert rp.masked_cos(out, to_ff(ideal), valid).min() > 0.9999


def test_fractional_shift_is_bilinear():
    """Features linear in position are moved exactly by bilinear interpolation."""
    rr, cc = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    lin = torch.stack([rr, cc, 2 * rr - cc, torch.ones_like(rr), rr + 3 * cc], -1)[None].repeat(4, 1, 1, 1)
    for dy, dx in ((-6.81, 0.0), (3.38, 0.0), (0.0, 2.4), (-3.38, -0.6)):
        out, valid = rp.translate(to_ff(lin), to_ff(torch.zeros_like(lin)), dy, dx)
        out = out.reshape(4, H, W, -1)
        exp = torch.stack([rr - dy, cc - dx, 2 * (rr - dy) - (cc - dx), torch.ones_like(rr),
                           (rr - dy) + 3 * (cc - dx)], -1)
        assert valid.any() and torch.allclose(out[:, valid], exp[valid][None].expand(4, -1, -1), atol=1e-9), (dy, dx)
        assert (out[:, ~valid] == 0).all()
        # every tap of a valid position is full content of the crop map
        rlo, rhi = rp.content_range(dy, rp.CONTENT_H)
        src_r = rr[valid] - dy
        assert src_r.min() >= rlo - 1e-9 and torch.ceil(src_r - 1e-9).max() <= rhi + 1e-9, (dy, src_r.min(), src_r.max())


def test_sca_stats_categories():
    """One round, camera slot 1: query 0 (pose 0) puts all mass on content, query 3 (pose 1) splits it
    between out-of-image, black fill and the padding row. Other cameras must not leak in."""
    n_poses, Q, D = 2, 6, 3  # 3 candidates x 2 waypoints, query q -> pose q % 2
    img = torch.zeros(1, 4, 3, 448, 768)  # normalised gray
    black = torch.tensor([-m / s for m, s in zip(rp.RAP_MEAN, rp.RAP_STD)])[:, None, None]
    img[0, 1, :, :100, :] = black  # black band on the top 100 rows of F0
    feats = {"camera_feature": img}
    bev_mask = torch.zeros(4, 1, Q, D, dtype=torch.bool)
    bev_mask[1, 0, 0, 0] = bev_mask[1, 0, 3, 2] = True  # F0 sees queries 0 and 3
    bev_mask[0, 0, :, :] = True  # B0 sees everything (must not matter)
    heads, P = 2, 2
    loc = torch.full((4, 2, heads, 1, P, 2), 0.5)
    w = torch.full((4, 2, heads, 1, P), 0.5)
    # query 3 (row 1 of the F0 block): head 0 -> out / black, head 1 -> pad / content
    loc[1, 1, 0, 0, 0] = torch.tensor([1.2, 0.5])
    loc[1, 1, 0, 0, 1] = torch.tensor([0.5, 50 / 448])
    loc[1, 1, 1, 0, 0] = torch.tensor([0.5, 440 / 448])
    rec = [("ps", None, bev_mask), ("attn", loc, w), ("attn", loc, w)]
    out = rp.sca_stats(rec, feats, n_poses, slot=1)
    assert out.shape == (1, n_poses, 5)
    assert torch.allclose(out[0, :, 0], torch.tensor([1 / 3, 1 / 3]))
    assert torch.allclose(out[0, 0, 1:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(out[0, 1, 1:], torch.tensor([0.25, 0.25, 0.25, 0.25])), out[0, 1]
    bev_mask[1] = False
    assert torch.isnan(rp.sca_stats([("ps", None, bev_mask), ("attn", loc, w)], feats, n_poses)[0, :, 1:]).all()


def test_divergence_helpers():
    g = torch.Generator().manual_seed(1)
    p = torch.randn(1, 64, 10, 3, generator=g)
    assert rp.proposal_ade(p, p) == 0.0 and rp.pool_cover(p, p) == 0.0
    q = p.clone()
    q[..., :2] += torch.tensor([3.0, 4.0])
    assert abs(rp.proposal_ade(q, p) - 5.0) < 1e-5
    perm = p[:, torch.randperm(64, generator=g)]
    assert rp.pool_cover(p, perm) == 0.0 and rp.proposal_ade(p, perm) > 0  # coverage is slot-free
    x = torch.randn(1, 640, 16, generator=g)
    assert abs(rp.query_cos(x, x) - 1.0) < 1e-6 and abs(rp.query_cos(x, -x) + 1.0) < 1e-6
    m = rp.cos_map(to_ff(torch.randn(4, H, W, CH, generator=g)), to_ff(torch.randn(4, H, W, CH, generator=g)),
                   torch.zeros(H, W, dtype=torch.bool), 1)
    assert m.shape == (H, W) and torch.isnan(m).all()


def test_push_outside():
    """Locations on a camera's fill / padding move to (-1, -1); others, and other cameras, stay."""
    feats = {"camera_feature": torch.zeros(1, 4, 3, 448, 768)}
    black = torch.tensor([-m / s for m, s in zip(rp.RAP_MEAN, rp.RAP_STD)])[:, None, None]
    feats["camera_feature"][0, 1, :, 300:432, :] = black  # bottom band of F0
    masks = rp.fill_mask(feats)
    assert masks[1, 350].all() and masks[0, 440].all() and not masks[0, 350].any() and not masks[1, 100].any()
    loc = torch.full((4, 3, 2, 1, 2, 2), 0.5)
    loc[:, 0, 0, 0, 0] = torch.tensor([0.5, 350 / 448])  # band on F0 only
    loc[:, 1, 0, 0, 0] = torch.tensor([0.5, 440 / 448])  # padding on every camera
    loc[:, 2, 0, 0, 0] = torch.tensor([1.3, 350 / 448])  # already outside
    out = rp.push_outside(loc, masks)
    assert torch.equal(out[1, 0, 0, 0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(out[0, 0, 0, 0, 0], loc[0, 0, 0, 0, 0])  # B0 has no band there
    assert (out[:, 1, 0, 0, 0] == -1).all()
    assert torch.equal(out[:, 2], loc[:, 2])
    untouched = torch.ones_like(loc[..., 0], dtype=torch.bool)
    untouched[1, 0, 0, 0, 0] = untouched[:, 1, 0, 0, 0] = False
    assert torch.equal(out[untouched], loc[untouched])


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")

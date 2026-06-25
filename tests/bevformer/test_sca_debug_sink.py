"""Unit tests for the opt-in debug capture added to the SCA deformable attention.

These exercise the C6 hook (`debug_sink`) without any real data: a single CPU
forward through `MSDeformableAttention3D` uses the `multi_scale_deformable_attn_pytorch`
fallback, so no CUDA is required.
"""

import torch

from navsim.agents.rap_dino.bevformer.spatial_cross_attention import MSDeformableAttention3D


def _build_inputs():
    torch.manual_seed(0)
    embed_dims, num_heads, num_levels, num_points = 8, 2, 1, 2
    num_query, num_Z = 3, 1  # num_Z == num_Z_anchors (D)

    attn = MSDeformableAttention3D(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=True,
    )
    attn.eval()

    spatial_shapes = torch.tensor([[2, 3]], dtype=torch.long)  # H=2, W=3 -> 6 values
    num_value = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
    level_start_index = torch.tensor([0], dtype=torch.long)

    query = torch.randn(1, num_query, embed_dims)
    value = torch.randn(1, num_value, embed_dims)
    reference_points = torch.rand(1, num_query, num_Z, 2)  # normalized [0, 1]

    kwargs = dict(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    return attn, kwargs


def test_debug_sink_populates_sampling_tensors():
    attn, kwargs = _build_inputs()
    sink = {}
    attn.debug_sink = sink
    with torch.no_grad():
        attn(**kwargs)

    assert "sampling_locations" in sink and "attention_weights" in sink
    sampling_locations = sink["sampling_locations"]
    attention_weights = sink["attention_weights"]
    # (bs, num_query, num_heads, num_levels, num_all_points, 2)
    assert sampling_locations.shape[-1] == 2
    assert sampling_locations.shape[:3] == (1, 3, 2)
    # attention weights normalized over the points dimension
    assert attention_weights.shape[:3] == (1, 3, 2)
    assert torch.isfinite(sampling_locations).all()
    assert torch.isfinite(attention_weights).all()


def test_debug_sink_none_is_noop():
    attn, kwargs = _build_inputs()

    attn.debug_sink = None
    with torch.no_grad():
        out_without = attn(**kwargs)

    attn.debug_sink = {}
    with torch.no_grad():
        out_with = attn(**kwargs)

    # Capture must not change the forward result.
    torch.testing.assert_close(out_without, out_with)

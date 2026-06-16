"""Compatibility helpers for dependency version mismatches."""

from __future__ import annotations


def patch_torch_pytree() -> None:
    """Expose the public pytree API expected by newer transformers releases."""
    try:
        import torch
    except ImportError:
        return

    pytree = getattr(torch.utils, "_pytree", None)
    if pytree is None:
        return

    if hasattr(pytree, "register_pytree_node") or not hasattr(
        pytree, "_register_pytree_node"
    ):
        return

    def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return pytree._register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = _register_pytree_node_compat

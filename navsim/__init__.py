"""NAVSIM package initialization and runtime compatibility helpers."""


def _patch_torch_pytree() -> None:
    """Bridge the PyTorch 2.1 pytree API expected by recent transformers."""
    try:
        import torch
    except ImportError:
        return

    pytree = getattr(torch.utils, "_pytree", None)
    if pytree is None:
        return
    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return pytree._register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs)

    pytree.register_pytree_node = register_pytree_node_compat


_patch_torch_pytree()

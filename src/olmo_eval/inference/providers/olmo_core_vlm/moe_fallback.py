"""Torch stand-ins for the TransformerEngine MoE permutation ops.

OLMo-core's OLMoDDP MoE blocks route tokens through TransformerEngine's
``moe_permute`` / ``moe_unpermute`` even without expert parallelism, and leave
them as ``None`` when TransformerEngine is not installed (it is not in the eval
images). These reproduce the "index" map semantics the no-EP forward uses:
token copies grouped by expert in ascending expert order, then merged back
weighted by the routing probabilities, accumulating in float32.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def moe_permute(
    inp: Any, routing_map: Any, num_out_tokens: int | None = -1, map_type: str = "index", **_: Any
) -> tuple[Any, Any]:
    """Group each token's ``top_k`` copies by expert.

    :param inp: ``(tokens, hidden)`` inputs.
    :param routing_map: ``(tokens, top_k)`` expert index per routing slot.
    :param num_out_tokens: Number of permuted rows to keep (``-1``/``None`` keeps all).
    :returns: ``(permuted, row_id_map)``: the ``(rows, hidden)`` permuted copies and,
        for each permuted row, the flat ``token * top_k + slot`` index it came from.
    """
    import torch

    if map_type != "index":
        raise NotImplementedError(f"moe_permute fallback supports map_type='index', not {map_type}")
    top_k = routing_map.shape[1]
    order = torch.argsort(routing_map.reshape(-1).long(), stable=True)
    if num_out_tokens is not None and num_out_tokens >= 0:
        order = order[:num_out_tokens]
    return inp.index_select(0, order // top_k), order


def moe_unpermute(
    inp: Any,
    row_id_map: Any,
    restore_shape: Any = None,
    map_type: str = "index",
    merging_probs: Any = None,
    **_: Any,
) -> Any:
    """Merge permuted rows back into their tokens, weighted by ``merging_probs``.

    :param inp: ``(rows, hidden)`` expert outputs in :func:`moe_permute` order.
    :param row_id_map: The ``row_id_map`` returned by :func:`moe_permute`.
    :param restore_shape: ``(tokens, hidden)`` shape of the original inputs.
    :param merging_probs: ``(tokens, top_k)`` routing weights; ``None`` sums the copies.
    """
    import torch

    if map_type != "index":
        raise NotImplementedError(
            f"moe_unpermute fallback supports map_type='index', not {map_type}"
        )
    num_tokens = restore_shape[0]
    top_k = row_id_map.numel() // num_tokens if merging_probs is None else merging_probs.shape[1]
    # Dropped slots (num_out_tokens below tokens * top_k) contribute zeros.
    rows = inp.new_zeros((num_tokens * top_k, inp.shape[-1]), dtype=torch.float32)
    rows[row_id_map.long()] = inp.float()
    rows = rows.view(num_tokens, top_k, -1)
    if merging_probs is not None:
        rows = rows * merging_probs.float().unsqueeze(-1)
    return rows.sum(dim=1).to(inp.dtype)


def install_torch_moe_permutation() -> bool:
    """Point OLMo-core's MoE permutation hooks at the torch fallbacks when TE is absent.

    :returns: Whether the fallbacks were installed (``False`` when TransformerEngine
        provides the ops, or the installed OLMo-core has no such hooks).
    """
    import importlib

    try:
        moe_utils = importlib.import_module("olmo_core.nn.moe.utils")
    except ImportError:
        return False
    if not hasattr(moe_utils, "moe_permute") or moe_utils.moe_permute is not None:
        return False
    setattr(moe_utils, "moe_permute", moe_permute)  # noqa: B010
    setattr(moe_utils, "moe_unpermute", moe_unpermute)  # noqa: B010
    logger.warning(
        "TransformerEngine is not installed; MoE token permutation uses the torch fallback "
        "(same routing, float32 merge; not bitwise identical to TE's kernels)"
    )
    return True


def grouped_mm_loop(mat_a: Any, mat_b: Any, *, offs: Any = None, **kwargs: Any) -> Any:
    """Per-group matmul with the semantics OLMo-core's routed experts use from ``F.grouped_mm``.

    ``mat_a`` is ``(rows, K)`` with rows grouped by expert, ``mat_b`` is ``(groups, K, N)``
    and ``offs[i]`` is the end row of group ``i``. Rows past ``offs[-1]`` are zero.
    """

    if offs is None or mat_a.dim() != 2 or mat_b.dim() != 3 or kwargs:
        raise NotImplementedError("grouped_mm fallback supports 2D x 3D with offs only")
    out = mat_a.new_zeros((mat_a.shape[0], mat_b.shape[-1]))
    start = 0
    for group, end in enumerate(offs.tolist()):
        if end > start:
            out[start:end] = mat_a[start:end] @ mat_b[group]
        start = end
    return out


def install_grouped_mm_fallback(device: Any) -> bool:
    """Replace ``F.grouped_mm`` with :func:`grouped_mm_loop` on Blackwell with torch < 2.13.

    torch 2.10's grouped-mm CUTLASS kernels are not built for Blackwell (sm_10x) and fail
    with an unspecified launch failure on B200/B300; the training stack runs torch 2.13.

    :returns: Whether the fallback was installed.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not hasattr(F, "grouped_mm"):
        return False
    major, _ = torch.cuda.get_device_capability(device)
    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if major < 10 or version >= (2, 13):
        return False
    setattr(F, "grouped_mm", grouped_mm_loop)  # noqa: B010
    logger.warning(
        "torch %s grouped_mm has no Blackwell kernels; MoE experts use a per-expert matmul loop",
        torch.__version__,
    )
    return True

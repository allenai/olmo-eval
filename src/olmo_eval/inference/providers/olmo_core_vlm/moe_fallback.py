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


# Largest element offset the routed-expert SwiGLU Triton kernel addresses without int32
# overflow: it builds ``row * stride + col`` from an int32 row index and an i32 stride.
_INT32_MAX = 2**31 - 1
# Smallest row count a routed-expert SwiGLU launch is specialized for. OLMo-core's kernel
# caps its grid at 1024 programs of 16 rows, so launches of at least this many rows differ
# only in the ``rows`` constexpr.
_SWIGLU_MIN_BUCKET_ROWS = 16384


def _floor_pow2(n: int) -> int:
    return 1 << (max(1, n).bit_length() - 1)


def swiglu_window_rows(
    rows: int,
    width: int,
    *,
    min_rows: int = _SWIGLU_MIN_BUCKET_ROWS,
    max_elements: int = _INT32_MAX,
) -> int:
    """Row count of every routed-expert SwiGLU kernel launch for a ``(rows, width)`` input.

    A power of two between ``min_rows`` and the largest power of two whose launch stays
    int32-addressable (``window * width <= max_elements``; 2**19 rows for 2H = 2048), so the
    kernel compiles at most ``log2(cap / min_rows) + 1`` variants however long decoding runs.

    :param rows: Routed rows in the input.
    :param width: ``2H``, the up/gate width.
    :param min_rows: Smallest window; inputs with fewer rows are padded up to it.
    :param max_elements: Largest element count one launch may address.
    :returns: ``min_rows`` (capped) for small inputs, otherwise the largest power of two
        ``<= rows``, capped at the int32 limit.
    """
    cap = _floor_pow2(max_elements // width)
    smallest = min(_floor_pow2(min_rows), cap)
    if rows <= smallest:
        return smallest
    return min(cap, _floor_pow2(rows))


def swiglu_valid_prefix_bucketed(
    swiglu_valid_prefix: Any,
    x: Any,
    num_elements: Any,
    *,
    start: Any = None,
    out: Any = None,
    min_rows: int = _SWIGLU_MIN_BUCKET_ROWS,
    max_elements: int = _INT32_MAX,
    **kwargs: Any,
) -> Any:
    """Call OLMo-core's ``swiglu_valid_prefix`` only on fixed power-of-two row windows.

    The kernel takes the input's row count as a ``tl.constexpr``, so uncached decoding, whose
    routed row count (active rows x span x top-k) changes on every step, compiled a new kernel
    for every forward and grew the Triton cache by ~170 KB per step. It also builds element
    offsets in int32, which overflows past ``2**31 - 1`` elements (over 65,536 tokens per
    forward for OLMo 3.5's top-16 routing and 2H = 2048).

    Every launch here sees exactly :func:`swiglu_window_rows` rows: inputs at least that long
    are covered by contiguous windows (views, no copy), the last one slid back to end at the
    final row and restricted to rows the previous windows did not cover; shorter inputs are
    copied into a padded buffer whose pad rows lie outside the valid range and are never read.
    Each row is computed once, by the same kernel, from its own inputs only (the op is
    row-wise), so the result is bitwise identical to a single call on the whole input.

    :param swiglu_valid_prefix: The original OLMo-core function.
    :param x: ``(rows, 2H)`` contiguous up/gate rows.
    :param num_elements: Device scalar, number of valid rows from ``start``.
    :param start: First valid row (``None``, an int or a device scalar).
    :param out: Optional ``(rows, H)`` output; rows outside the valid range stay untouched.
    :param min_rows: Smallest launch, see :func:`swiglu_window_rows`.
    :param max_elements: Largest element count one launch may address.
    :returns: The ``(rows, H)`` output.
    """
    import torch

    if x.ndim != 2:
        return swiglu_valid_prefix(x, num_elements, start=start, out=out, **kwargs)
    rows, width = x.shape
    window = swiglu_window_rows(rows, width, min_rows=min_rows, max_elements=max_elements)
    count = num_elements.to(torch.int64)
    zero = count.new_zeros(())
    # Like the kernel's own ``row < rows`` mask, rows past the input are never touched.
    begin = (zero if start is None else zero + start).clamp(0, rows)
    end = (begin + count).clamp(max=rows)

    if rows < window:
        padded_x = x.new_empty((window, width))
        padded_x[:rows].copy_(x)
        padded_out = x.new_empty((window, width // 2))
        if out is not None:
            padded_out[:rows].copy_(out)
        swiglu_valid_prefix(
            padded_x, (end - begin).clamp(min=0), start=begin, out=padded_out, **kwargs
        )
        if out is None:
            return padded_out[:rows]
        out.copy_(padded_out[:rows])
        return out

    if out is None:
        out = x.new_empty((rows, width // 2))
    for lo in range(0, rows, window):
        hi = min(lo + window, rows)
        first = min(lo, rows - window)  # the last window slides back to stay full
        chunk_begin = begin.clamp(lo, hi)
        chunk_end = end.clamp(lo, hi)
        swiglu_valid_prefix(
            x[first : first + window],
            (chunk_end - chunk_begin).clamp(min=0),
            start=chunk_begin - first,
            out=out[first : first + window],
            **kwargs,
        )
    return out


def install_swiglu_row_buckets() -> bool:
    """Route OLMo-core's routed-expert SwiGLU kernel through :func:`swiglu_valid_prefix_bucketed`.

    Only no-grad forwards (eval) use this kernel.

    :returns: Whether the wrapper was installed (``False`` when the installed OLMo-core has no
        such kernel, or the wrapper is already in place).
    """
    import functools
    import importlib

    try:
        routed_experts = importlib.import_module("olmo_core.nn.moe.v2.routed_experts")
    except ImportError:
        return False
    original = getattr(routed_experts, "swiglu_valid_prefix", None)
    if original is None or (
        isinstance(original, functools.partial) and original.func is swiglu_valid_prefix_bucketed
    ):
        return False
    bucketed = functools.partial(swiglu_valid_prefix_bucketed, original)
    setattr(routed_experts, "swiglu_valid_prefix", bucketed)  # noqa: B010
    logger.info(
        "Routed-expert SwiGLU kernel calls run on power-of-two, int32-addressable row windows"
    )
    return True


def _ceil_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _autotune_key_only(kernel: Any, name: str) -> bool:
    """Whether ``name`` is an autotune key of ``kernel`` that its body never reads."""
    import ast

    keys: Any = None
    node = kernel
    for _ in range(8):
        if keys is None and isinstance(getattr(node, "keys", None), list):
            keys = node.keys
        src = getattr(node, "src", None)
        if isinstance(src, str):
            if keys is None or name not in keys:
                return False
            tree = ast.parse(src)
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
            return not any(
                isinstance(n, ast.Name) and n.id == name for stmt in fn.body for n in ast.walk(stmt)
            )
        node = getattr(node, "fn", None)
        if node is None:
            return False
    return False


class _PowerOfTwoKeyLaunch:
    """Launch proxy for an autotuned Triton kernel that rounds one key argument up to a power
    of two before the autotuner sees it."""

    def __init__(self, kernel: Any, key: str) -> None:
        self.kernel = kernel
        self.key = key

    def __getitem__(self, grid: Any) -> Any:
        launch = self.kernel[grid]

        def run(*args: Any, **kwargs: Any) -> Any:
            value = kwargs.get(self.key)
            if isinstance(value, int) and not isinstance(value, bool):
                kwargs[self.key] = _ceil_pow2(value)
            return launch(*args, **kwargs)

        return run

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)


def install_conv1d_autotune_buckets() -> bool:
    """Round fla's short-convolution autotune key ``NB`` up to a power of two.

    fla's ``causal_conv1d_fwd`` passes ``NB = cdiv(B * T, 1024)`` to its Triton kernel as a
    constexpr autotune key that the kernel body never reads. Uncached decoding grows
    ``B * T`` every step, so it re-autotuned (32 compiles, ~10 s, ~7 MB of Triton cache)
    every 1024 / B steps. With power-of-two buckets it re-autotunes only when ``B * T``
    doubles. The convolution's arithmetic does not depend on ``NB`` or the chosen tile, so
    outputs are unchanged.

    fla's ``l2norm_fwd_kernel`` and ``layer_norm_gated_fwd_kernel`` also key on a token-count
    ``NB`` (``cdiv(T, 65536)``), but they re-autotune only once per 65,536 rows, and their
    configs change ``num_warps`` around a row reduction, so a different pick could change
    bits; they are left as they are.

    :returns: Whether the proxy was installed (``False`` when fla is not installed, its kernel
        reads ``NB`` or has no such key, or the proxy is already in place).
    """
    import importlib

    try:
        ops = importlib.import_module("fla.modules.conv.triton.ops")
    except ImportError:
        return False
    kernel = getattr(ops, "causal_conv1d_fwd_kernel", None)
    if kernel is None or isinstance(kernel, _PowerOfTwoKeyLaunch):
        return False
    if not _autotune_key_only(kernel, "NB"):
        logger.warning("fla's causal_conv1d_fwd_kernel reads NB; leaving its autotune key as is")
        return False
    setattr(ops, "causal_conv1d_fwd_kernel", _PowerOfTwoKeyLaunch(kernel, "NB"))  # noqa: B010
    logger.info("fla short-convolution autotune key NB is rounded up to a power of two")
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

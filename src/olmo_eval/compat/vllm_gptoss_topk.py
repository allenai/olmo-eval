"""Compatibility for non-power-of-two GPT-OSS routing on vLLM 0.19.1.

The Triton top-k kernel bundled with vLLM 0.19.1 constructs
``tl.arange(0, k)``, which only compiles when ``k`` is a power of two. Keep
the fast Triton path for supported values and use the bundled torch fallback
for other values. This patch is intentionally scoped to the GPT-OSS MoE
module rather than replacing ``triton_kernels.topk`` process-wide.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def make_non_power_of_two_dispatch(
    triton_topk: Callable[..., Any],
    torch_topk: Callable[..., Any],
) -> Callable[..., Any]:
    """Return a top-k dispatcher that falls back only for unsupported K."""

    @wraps(triton_topk)
    def dispatch(
        x: Any,
        k: int,
        apply_softmax: bool = True,
        dim: int = 1,
        y_indx: Any = None,
        n_rows: int | None = None,
        all_gather: bool = False,
    ) -> Any:
        if _is_power_of_two(k) or all_gather:
            return triton_topk(
                x,
                k,
                apply_softmax,
                dim,
                y_indx,
                n_rows,
                all_gather,
            )
        return torch_topk(x, k, apply_softmax, dim, y_indx, n_rows)

    dispatch._olmo_eval_nonpow2_dispatch = True  # type: ignore[attr-defined]
    return dispatch


def reference_scaled_topk(
    logits: Any,
    *,
    top_k: int,
    reference_k: int,
) -> tuple[Any, Any]:
    """Select top-K experts while retaining the native reference-K scale."""
    import torch

    if not 1 <= reference_k <= logits.shape[-1]:
        raise ValueError("reference_k must be within the expert dimension")
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k must be within the expert dimension")

    selected_count = max(top_k, reference_k)
    ranked_ids = torch.argsort(-logits, dim=1, stable=True)[:, :selected_count]
    ranked_logits = torch.take_along_dim(logits, ranked_ids, dim=1)
    reference_logsumexp = torch.logsumexp(
        ranked_logits[:, :reference_k].float(),
        dim=-1,
        keepdim=True,
    )
    selected_ids = ranked_ids[:, :top_k]
    selected_weights = torch.exp(
        ranked_logits[:, :top_k].float() - reference_logsumexp
    ).to(logits.dtype)
    return selected_weights, selected_ids


def _routing_from_selected(
    *,
    topk_weights: Any,
    topk_ids: Any,
    num_experts: int,
    routing_module: Any,
) -> Any:
    """Build vLLM routing metadata from selected weights and expert IDs."""
    import torch

    topk_ids, sort_indices = torch.sort(topk_ids, dim=1)
    topk_weights = torch.gather(topk_weights, 1, sort_indices)
    topk_ids = topk_ids.to(torch.int32)

    flat_ids = topk_ids.flatten()
    num_assignments = flat_ids.numel()
    combine_indx = torch.argsort(flat_ids, stable=True).to(torch.int32)
    dispatch_indx = torch.empty_like(combine_indx)
    dispatch_indx[combine_indx.to(torch.int64)] = torch.arange(
        num_assignments,
        device=topk_ids.device,
        dtype=torch.int32,
    )
    expert_counts = torch.zeros(
        num_experts,
        device=topk_ids.device,
        dtype=torch.int32,
    )
    expert_counts.scatter_add_(
        0,
        flat_ids.to(torch.int64),
        torch.ones_like(flat_ids),
    )
    ragged_batch_metadata = routing_module.make_ragged_tensor_metadata(
        expert_counts,
        num_assignments,
    )
    gate_scal = topk_weights.flatten()[combine_indx.to(torch.int64)]
    routing_data = routing_module.RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        num_experts,
        topk_ids.shape[1],
        ragged_batch_metadata,
    )
    gather_idx = routing_module.GatherIndx(
        combine_indx,
        dispatch_indx,
    )
    scatter_idx = routing_module.ScatterIndx(
        dispatch_indx,
        combine_indx,
    )
    return routing_data, gather_idx, scatter_idx


def apply_gptoss_non_power_of_two_topk_patch() -> bool:
    """Patch vLLM's GPT-OSS module; return whether a patch was installed."""

    import torch
    from vllm.model_executor.layers.fused_moe import gpt_oss_triton_kernels_moe

    current_routing = gpt_oss_triton_kernels_moe.legacy_routing
    if getattr(current_routing, "_olmo_eval_nonpow2_dispatch", False):
        return False

    def routing(
        logits: Any,
        n_expts_act: int,
        sm_first: bool = False,
    ) -> Any:
        if _is_power_of_two(n_expts_act):
            return current_routing(logits, n_expts_act, sm_first=sm_first)

        scores = torch.softmax(logits, dim=-1) if sm_first else logits
        topk_ids = torch.argsort(-scores, dim=1, stable=True)[:, :n_expts_act]
        topk_weights = torch.take_along_dim(scores, topk_ids, dim=1)
        if not sm_first:
            topk_weights = torch.softmax(topk_weights.float(), dim=-1).to(logits.dtype)

        return _routing_from_selected(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=logits.shape[-1],
            routing_module=gpt_oss_triton_kernels_moe,
        )

    routing._olmo_eval_nonpow2_dispatch = True  # type: ignore[attr-defined]
    gpt_oss_triton_kernels_moe.legacy_routing = routing
    return True


def apply_gptoss_reference_scale_patch(*, reference_k: int = 4) -> bool:
    """Patch GPT-OSS routing to preserve the checkpoint-default K scale."""
    from vllm.model_executor.layers.fused_moe import gpt_oss_triton_kernels_moe

    current_routing = gpt_oss_triton_kernels_moe.legacy_routing
    if getattr(current_routing, "_olmo_eval_reference_scale", False):
        return False

    def routing(
        logits: Any,
        n_expts_act: int,
        sm_first: bool = False,
    ) -> Any:
        if sm_first:
            raise ValueError(
                "GPT-OSS reference scaling requires selected-logit softmax routing"
            )
        topk_weights, topk_ids = reference_scaled_topk(
            logits,
            top_k=n_expts_act,
            reference_k=reference_k,
        )
        return _routing_from_selected(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=logits.shape[-1],
            routing_module=gpt_oss_triton_kernels_moe,
        )

    routing._olmo_eval_nonpow2_dispatch = True  # type: ignore[attr-defined]
    routing._olmo_eval_reference_scale = True  # type: ignore[attr-defined]
    routing._olmo_eval_reference_k = reference_k  # type: ignore[attr-defined]
    routing._olmo_eval_wrapped_routing = current_routing  # type: ignore[attr-defined]
    gpt_oss_triton_kernels_moe.legacy_routing = routing
    return True

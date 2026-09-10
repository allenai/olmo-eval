"""Opt-in Qwen selected-expert weight perturbations for vLLM 0.19.1.

The selected expert IDs are left untouched. Only their already-normalized
weights are transformed before the fused expert kernel.
"""

from __future__ import annotations

import itertools
import sys
from functools import wraps
from typing import Any

import torch

from olmo_eval.compat.qwen_expert_weight_policies import (
    QwenExpertWeightPolicy,
    transform_topk_weights,
)
from olmo_eval.compat.qwen_realized_k_telemetry import record_qwen_realized_k

QWEN_NUM_EXPERTS = 128


def apply_qwen_expert_weight_patch(
    *,
    policy: QwenExpertWeightPolicy,
    record_realized_k: bool = False,
    num_experts: int = QWEN_NUM_EXPERTS,
) -> bool:
    """Patch vLLM's standard router; return whether a patch was installed."""
    if policy.mode == "normal":
        raise ValueError("normal mode must not install the vLLM patch")

    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    current_compute = FusedTopKRouter._compute_routing
    if getattr(current_compute, "_olmo_eval_qwen_expert_weight_patch", False):
        return False

    current_init = FusedTopKRouter.__init__
    layer_indices = itertools.count()
    reported_match = False

    @wraps(current_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal reported_match
        current_init(self, *args, **kwargs)
        if self.top_k == policy.router_k and self.global_num_experts == num_experts:
            self._olmo_eval_qwen_layer_index = next(layer_indices)
            if not reported_match:
                print(
                    "olmo-eval: Qwen expert-weight policy matched a router layer "
                    f"(top_k={self.top_k}, global_num_experts={self.global_num_experts})",
                    file=sys.stderr,
                )
                reported_match = True

    @wraps(current_compute)
    def patched_compute(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = current_compute(
            self,
            hidden_states,
            router_logits,
            indices_type,
        )
        layer_index = getattr(self, "_olmo_eval_qwen_layer_index", None)
        if layer_index is None:
            return topk_weights, topk_ids
        transformed = transform_topk_weights(
            topk_weights,
            topk_ids,
            policy=policy,
            layer_index=layer_index,
        )
        if record_realized_k:
            record_qwen_realized_k(
                layer_index=layer_index,
                weights=transformed,
                policy_description=policy.describe(),
            )
        return transformed, topk_ids

    patched_compute._olmo_eval_qwen_expert_weight_patch = True  # type: ignore[attr-defined]
    patched_compute._olmo_eval_qwen_expert_weight_policy = policy  # type: ignore[attr-defined]
    patched_compute._olmo_eval_qwen_record_realized_k = record_realized_k  # type: ignore[attr-defined]
    patched_compute._olmo_eval_qwen_num_experts = num_experts  # type: ignore[attr-defined]
    FusedTopKRouter.__init__ = patched_init
    FusedTopKRouter._compute_routing = patched_compute
    return True

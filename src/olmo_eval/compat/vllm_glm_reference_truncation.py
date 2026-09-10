"""GLM native-reference expert truncation for vLLM 0.26.

GLM-5.2 uses sigmoid routing, selects eight experts, normalizes the selected
weights over those eight experts, and applies a model-level routed scaling
factor.  Merely changing ``num_experts_per_tok`` to four changes the
normalization denominator.  This opt-in compatibility patch keeps a K=4 MoE
kernel while deriving its four routing weights from the native K=8
denominator.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch

GLM52_NUM_EXPERTS = 256
GLM52_KEEP_K = 4
GLM52_REFERENCE_K = 8

GroupedTopKImplementation = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def native_reference_truncated_grouped_topk(
    *,
    grouped_topk_impl: GroupedTopKImplementation,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    keep_k: int,
    reference_k: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select native top-K experts and scale them with a reference-K sum.

    The reference and retained selections are computed separately because
    vLLM may request unsorted top-k results.  Slicing a native top-8 result is
    therefore not guaranteed to retain the four highest-ranked experts.
    """
    if not 1 <= keep_k < reference_k:
        raise ValueError("expected 1 <= keep_k < reference_k")

    common_kwargs = {
        "hidden_states": hidden_states,
        "gating_output": router_logits,
        "renormalize": False,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "scoring_func": scoring_func,
        # Apply the scale once, after using the unscaled reference weights as
        # the denominator. In GLM-5.2's CUDA path vLLM applies the model's 2.5
        # scale to the routed output, so this router-side value is 1.0.
        "routed_scaling_factor": 1.0,
        "e_score_correction_bias": e_score_correction_bias,
    }
    reference_weights, _ = grouped_topk_impl(
        topk=reference_k,
        **common_kwargs,
    )
    retained_weights, retained_ids = grouped_topk_impl(
        topk=keep_k,
        **common_kwargs,
    )

    if reference_weights.shape[-1] != reference_k:
        raise RuntimeError(
            "GLM reference router returned "
            f"K={reference_weights.shape[-1]}, expected K={reference_k}"
        )
    if retained_weights.shape != retained_ids.shape:
        raise RuntimeError(
            "GLM retained router returned mismatched weight/id shapes: "
            f"{retained_weights.shape} != {retained_ids.shape}"
        )
    if retained_weights.shape[-1] != keep_k:
        raise RuntimeError(
            f"GLM retained router returned K={retained_weights.shape[-1]}, expected K={keep_k}"
        )

    denominator = reference_weights.to(torch.float32).sum(dim=-1, keepdim=True)
    weights = retained_weights.to(torch.float32) / denominator
    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor
    return weights, retained_ids


def apply_glm_reference_truncation_patch(
    *,
    keep_k: int = GLM52_KEEP_K,
    reference_k: int = GLM52_REFERENCE_K,
    num_experts: int = GLM52_NUM_EXPERTS,
) -> bool:
    """Patch matching GLM grouped routers; return whether it was installed."""
    if not 1 <= keep_k < reference_k <= num_experts:
        raise ValueError(
            "GLM reference truncation requires 1 <= keep_k < reference_k <= num_experts"
        )

    from vllm.model_executor.layers.fused_moe.router import grouped_topk_router

    router_class = grouped_topk_router.GroupedTopKRouter
    current_compute = router_class._compute_routing
    if getattr(current_compute, "_olmo_eval_glm_reference_truncation_patch", False):
        return False

    current_init = router_class.__init__
    reported_match = False

    @wraps(current_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal reported_match
        current_init(self, *args, **kwargs)
        if self.top_k != keep_k or self.global_num_experts != num_experts:
            return

        expected = {
            "renormalize": True,
            "scoring_func": "sigmoid",
            "num_expert_group": 1,
            "topk_group": 1,
        }
        mismatches = {
            name: (getattr(self, name), value)
            for name, value in expected.items()
            if getattr(self, name) != value
        }
        if self.e_score_correction_bias is None:
            mismatches["e_score_correction_bias"] = (None, "a tensor")
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {wanted!r})"
                for name, (actual, wanted) in mismatches.items()
            )
            raise RuntimeError(
                "GLM reference truncation found a candidate router with "
                f"unexpected native semantics: {details}"
            )

        self._olmo_eval_glm_reference_truncation = True
        if not reported_match:
            print(
                "olmo-eval: GLM native-reference truncation matched a router "
                f"(kernel_k={keep_k}, reference_k={reference_k}, "
                f"global_num_experts={num_experts})",
                file=sys.stderr,
            )
            reported_match = True

    @wraps(current_compute)
    def patched_compute(
        self: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self, "_olmo_eval_glm_reference_truncation", False):
            return current_compute(
                self,
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
        if router_logits.shape[-1] != num_experts:
            raise RuntimeError(
                "GLM reference truncation received "
                f"{router_logits.shape[-1]} router logits, expected {num_experts}"
            )

        return native_reference_truncated_grouped_topk(
            grouped_topk_impl=grouped_topk_router.grouped_topk,
            hidden_states=hidden_states,
            router_logits=router_logits,
            keep_k=keep_k,
            reference_k=reference_k,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

    patched_compute._olmo_eval_glm_reference_truncation_patch = True  # type: ignore[attr-defined]
    patched_compute._olmo_eval_glm_keep_k = keep_k  # type: ignore[attr-defined]
    patched_compute._olmo_eval_glm_reference_k = reference_k  # type: ignore[attr-defined]
    patched_compute._olmo_eval_glm_num_experts = num_experts  # type: ignore[attr-defined]
    router_class.__init__ = patched_init
    router_class._compute_routing = patched_compute
    return True

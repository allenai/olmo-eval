from collections.abc import Callable
from typing import Any

import pytest
import torch

from olmo_eval.compat.vllm_glm_reference_truncation import (
    native_reference_truncated_grouped_topk,
)


def make_grouped_topk(
    calls: list[dict[str, Any]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    # Intentionally return the reference-K row out of rank order. The helper
    # must make a separate keep-K selection rather than slice this row.
    raw_reference_weights = torch.tensor([[0.10, 0.20, 0.05, 0.25, 0.15, 0.08, 0.12, 0.05]])
    raw_reference_ids = torch.tensor([[7, 2, 31, 1, 9, 18, 4, 22]], dtype=torch.int32)
    raw_retained_weights = torch.tensor([[0.25, 0.20, 0.15, 0.12]])
    raw_retained_ids = torch.tensor([[1, 2, 9, 4]], dtype=torch.int32)

    def grouped_topk(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        if kwargs["topk"] == 8:
            return raw_reference_weights, raw_reference_ids
        if kwargs["topk"] == 4:
            return raw_retained_weights, raw_retained_ids
        raise AssertionError(f"unexpected topk={kwargs['topk']}")

    return grouped_topk


def test_native_reference_truncation_uses_top8_denominator_and_top4_ids() -> None:
    calls: list[dict[str, Any]] = []
    bias = torch.zeros(256)

    weights, ids = native_reference_truncated_grouped_topk(
        grouped_topk_impl=make_grouped_topk(calls),
        hidden_states=torch.zeros(1, 16),
        router_logits=torch.zeros(1, 256),
        keep_k=4,
        reference_k=8,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        e_score_correction_bias=bias,
    )

    expected = torch.tensor([[0.25, 0.20, 0.15, 0.12]])
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(ids, torch.tensor([[1, 2, 9, 4]], dtype=torch.int32))
    assert weights.sum().item() == pytest.approx(0.72)
    assert [call["topk"] for call in calls] == [8, 4]
    assert all(call["renormalize"] is False for call in calls)
    assert all(call["routed_scaling_factor"] == 1.0 for call in calls)
    assert all(call["scoring_func"] == "sigmoid" for call in calls)
    assert all(call["e_score_correction_bias"] is bias for call in calls)


def test_native_reference_truncation_applies_router_side_scale_once() -> None:
    calls: list[dict[str, Any]] = []

    weights, _ = native_reference_truncated_grouped_topk(
        grouped_topk_impl=make_grouped_topk(calls),
        hidden_states=torch.zeros(1, 16),
        router_logits=torch.zeros(1, 256),
        keep_k=4,
        reference_k=8,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=torch.zeros(256),
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([[0.625, 0.50, 0.375, 0.30]]),
    )
    assert all(call["routed_scaling_factor"] == 1.0 for call in calls)


@pytest.mark.parametrize(
    ("keep_k", "reference_k"),
    [(0, 8), (8, 8), (9, 8)],
)
def test_native_reference_truncation_rejects_invalid_k(
    keep_k: int,
    reference_k: int,
) -> None:
    with pytest.raises(ValueError, match="keep_k < reference_k"):
        native_reference_truncated_grouped_topk(
            grouped_topk_impl=lambda **kwargs: (torch.empty(0), torch.empty(0)),
            hidden_states=torch.zeros(1, 16),
            router_logits=torch.zeros(1, 256),
            keep_k=keep_k,
            reference_k=reference_k,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            e_score_correction_bias=torch.zeros(256),
        )

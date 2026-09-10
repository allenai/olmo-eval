from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "adaptive_experts"
    / "nemotron_router_contribution_profile.py"
)
SPEC = importlib.util.spec_from_file_location("nemotron_profile", SCRIPT)
assert SPEC and SPEC.loader
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_router_metrics_use_selected_and_all_sigmoid_denominators() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6] + [0.1] * 18 + [0.01] * 2])
    selected_ids = torch.arange(22)[None, :]
    selected_weights = scores[:, :22] / scores[:, :22].sum(dim=-1, keepdim=True) * 5

    k_values, scalars, order = PROFILE.router_metric_tensors(
        scores=scores,
        selected_ids=selected_ids,
        selected_weights=selected_weights,
    )

    assert order.shape == (1, 22)
    assert torch.allclose(k_values[0, -1, 0], torch.tensor(1.0))
    expected_all_share = scores[:, :22].sum() / scores.sum()
    assert torch.allclose(k_values[0, -1, 1], expected_all_share)
    assert torch.allclose(k_values[0, -1, 2], torch.tensor(1.0))
    assert torch.allclose(scalars[0, -1], torch.tensor(5.0))


def test_contribution_metrics_separate_shared_and_routed_outputs() -> None:
    torch.manual_seed(7)
    batch, hidden = 3, 5
    raw_weights = torch.linspace(22, 1, 22).repeat(batch, 1)
    weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True) * 5
    unweighted = torch.randn(batch, 22, hidden)
    contributions = unweighted * weights[..., None]
    routed_projection_bias = torch.randn(batch, hidden)
    routed = contributions.sum(dim=1) + routed_projection_bias
    shared = torch.randn(batch, hidden)
    total = routed + shared
    order = torch.arange(22).repeat(batch, 1)

    rank, k_values, scalars, validation = PROFILE.contribution_metric_tensors(
        selected_weights=weights,
        projected_contributions=contributions,
        routed_projection_bias=routed_projection_bias,
        routed_output=routed,
        shared_output=shared,
        total_output=total,
        order=order,
    )

    assert rank.shape == (batch, 22, len(PROFILE.RANK_METRICS))
    assert k_values.shape == (
        batch,
        len(PROFILE.K_VALUES),
        len(PROFILE.CONTRIBUTION_K_METRICS),
    )
    assert torch.allclose(k_values[:, -1, 0], torch.ones(batch))
    assert torch.allclose(k_values[:, -1, 2], torch.ones(batch), atol=1e-6)
    assert torch.allclose(k_values[:, -1, 5], torch.ones(batch), atol=1e-6)
    assert torch.allclose(k_values[:, -1, 8], torch.ones(batch), atol=1e-6)
    assert torch.allclose(k_values[:, -1, 10], torch.ones(batch), atol=1e-6)
    assert torch.all(validation < 1e-5)
    assert scalars.shape == (batch, len(PROFILE.CONTRIBUTION_SCALAR_METRICS))


def test_reference_and_renormalized_counterfactuals_differ_below_native_k() -> None:
    weights = torch.ones(1, 22) * (5 / 22)
    contributions = torch.zeros(1, 22, 2)
    contributions[..., 0] = weights
    routed = contributions.sum(dim=1)
    shared = torch.tensor([[0.0, 1.0]])
    total = routed + shared
    order = torch.arange(22)[None, :]

    _, k_values, _, _ = PROFILE.contribution_metric_tensors(
        selected_weights=weights,
        projected_contributions=contributions,
        routed_projection_bias=torch.zeros_like(routed),
        routed_output=routed,
        shared_output=shared,
        total_output=total,
        order=order,
    )

    # At K=11 the reference-preserving routed norm is half the baseline,
    # while renormalization restores this deliberately collinear example.
    k11_index = PROFILE.K_VALUES.index(11)
    assert torch.allclose(k_values[0, k11_index, 4], torch.tensor(0.5))
    assert torch.allclose(k_values[0, k11_index, 7], torch.tensor(1.0))


def test_mamba_rmsnorm_fallback_matches_reference() -> None:
    for name in (
        "mamba_ssm.ops.triton.layernorm_gated",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops",
        "mamba_ssm",
    ):
        sys.modules.pop(name, None)
    assert PROFILE.install_mamba_rmsnorm_fallback()
    assert importlib.util.find_spec("mamba_ssm") is not None
    fallback = sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    z = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0])
    gated = x * F.silu(z)
    grouped = gated.reshape(1, 2, 2)
    expected = (
        grouped
        * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-5)
    ).reshape_as(x) * weight
    actual = fallback(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=1e-5,
        group_size=2,
        norm_before_gate=False,
    )
    assert torch.allclose(actual, expected)


def test_round_robin_shards_preserve_task_balance() -> None:
    examples = [
        {"task": task, "id": number}
        for task in ("gpqa", "ifeval", "math")
        for number in range(20)
    ]
    for shard_index in range(4):
        selected = PROFILE.select_examples(
            examples,
            shard_index=shard_index,
            num_shards=4,
            offset=0,
            limit=None,
        )
        assert len(selected) == 15
        assert {task: sum(row["task"] == task for row in selected) for task in ("gpqa", "ifeval", "math")} == {
            "gpqa": 5,
            "ifeval": 5,
            "math": 5,
        }

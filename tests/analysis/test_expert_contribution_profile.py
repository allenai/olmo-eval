import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "adaptive_experts"
    / "expert_contribution_profile.py"
)
SPEC = importlib.util.spec_from_file_location("expert_contribution_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


def metric_index(names, name):
    return names.index(name)


def test_identical_expert_outputs_are_locally_redundant():
    weights = torch.tensor([[0.40, 0.20, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02]])
    expert_output = torch.zeros(1, 8, 4)
    expert_output[..., 0] = 1.0
    contributions = expert_output * weights[..., None]
    output = contributions.sum(dim=1)
    moments = profile.compute_moments(
        gate_weights=weights,
        contributions=contributions,
        moe_output=output,
        residual_norm=torch.ones(1),
    )

    k_cosine = moments.k_sum[:, metric_index(profile.K_METRICS, "counterfactual_cosine")]
    k_relative_l2 = moments.k_sum[
        :, metric_index(profile.K_METRICS, "counterfactual_relative_l2")
    ]
    leave_one_out_l2 = moments.rank_sum[
        :, metric_index(profile.RANK_METRICS, "leave_one_out_relative_l2")
    ]
    assert torch.allclose(k_cosine, torch.ones(8), atol=1e-6)
    assert torch.allclose(k_relative_l2, torch.zeros(8), atol=1e-6)
    assert torch.allclose(leave_one_out_l2, torch.zeros(8), atol=1e-6)


def test_large_low_weight_activations_reverse_contribution_ranking():
    weights = torch.tensor([[0.40, 0.20, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02]])
    expert_output = torch.eye(8)[None, :, :]
    expert_output[:, 4:] *= 10.0
    contributions = expert_output * weights[..., None]
    output = contributions.sum(dim=1)
    moments = profile.compute_moments(
        gate_weights=weights,
        contributions=contributions,
        moe_output=output,
        residual_norm=torch.ones(1),
    )

    scalar = moments.scalar_sum
    assert scalar[
        metric_index(profile.SCALAR_METRICS, "bottom4_any_exceeds_top4")
    ].item() == 1.0
    assert scalar[
        metric_index(profile.SCALAR_METRICS, "bottom4_top4_inversion_fraction")
    ].item() > 0.5
    weighted_norm = moments.rank_sum[
        :, metric_index(profile.RANK_METRICS, "weighted_contribution_norm")
    ]
    assert weighted_norm[4] > weighted_norm[0]


def test_counterfactual_k8_reconstructs_full_output():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 8, generator=generator)
    weights = torch.softmax(logits, dim=-1)
    expert_output = torch.randn(3, 8, 16, generator=generator)
    contributions = expert_output * weights[..., None]
    output = contributions.sum(dim=1)
    moments = profile.compute_moments(
        gate_weights=weights,
        contributions=contributions,
        moe_output=output,
        residual_norm=torch.ones(3),
    )

    assert moments.validation.max().item() < 1e-6
    k8_relative_l2_sum = moments.k_sum[
        7, metric_index(profile.K_METRICS, "counterfactual_relative_l2")
    ]
    assert k8_relative_l2_sum.item() < 1e-6


def test_coherence_detects_canceling_expert_outputs():
    weights = torch.full((1, 8), 1 / 8)
    expert_output = torch.zeros(1, 8, 4)
    expert_output[:, :4, 0] = 1.0
    expert_output[:, 4:, 0] = -0.9
    contributions = expert_output * weights[..., None]
    moments = profile.compute_moments(
        gate_weights=weights,
        contributions=contributions,
        moe_output=contributions.sum(dim=1),
        residual_norm=torch.ones(1),
    )

    coherence = moments.scalar_sum[
        metric_index(profile.SCALAR_METRICS, "coherence_ratio")
    ]
    assert coherence.item() < 0.1

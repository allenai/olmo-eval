import pytest
import torch

from olmo_eval.compat.qwen_expert_weight_policies import (
    QwenExpertWeightPolicy,
    qwen_expert_weight_policy_from_env,
    transform_topk_weights,
)


@pytest.fixture
def routing() -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.tensor(
        [
            [0.30, 0.20, 0.15, 0.12, 0.09, 0.06, 0.05, 0.03],
            [0.25, 0.19, 0.16, 0.13, 0.10, 0.08, 0.06, 0.03],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor(
        [
            [2, 11, 19, 31, 47, 61, 89, 120],
            [4, 17, 28, 39, 55, 72, 94, 126],
        ],
        dtype=torch.int32,
    )
    return weights, expert_ids


def apply(
    routing: tuple[torch.Tensor, torch.Tensor],
    policy: QwenExpertWeightPolicy,
    *,
    layer_index: int = 0,
) -> torch.Tensor:
    weights, expert_ids = routing
    return transform_topk_weights(
        weights,
        expert_ids,
        policy=policy,
        layer_index=layer_index,
    )


def test_normal_mode_is_exact_identity(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing

    transformed = apply(routing, QwenExpertWeightPolicy(mode="normal"))

    assert transformed is weights


def test_uniform_mode_preserves_sum(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing

    transformed = apply(routing, QwenExpertWeightPolicy(mode="uniform"))

    torch.testing.assert_close(transformed, torch.full_like(weights, 1 / 8))
    torch.testing.assert_close(transformed.sum(-1), weights.sum(-1))


def test_shuffle_is_deterministic_and_preserves_each_weight_multiset(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing
    policy = QwenExpertWeightPolicy(mode="shuffle", seed=101)

    first = apply(routing, policy, layer_index=7)
    repeat = apply(routing, policy, layer_index=7)
    other_layer = apply(routing, policy, layer_index=8)

    torch.testing.assert_close(first, repeat)
    torch.testing.assert_close(first.sort(-1).values, weights.sort(-1).values)
    torch.testing.assert_close(first.sum(-1), weights.sum(-1))
    assert not torch.equal(first, weights)
    assert not torch.equal(first, other_layer)


def test_shuffle_changes_with_seed(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    first = apply(
        routing,
        QwenExpertWeightPolicy(mode="shuffle", seed=101),
        layer_index=7,
    )
    second = apply(
        routing,
        QwenExpertWeightPolicy(mode="shuffle", seed=202),
        layer_index=7,
    )

    assert not torch.equal(first, second)


@pytest.mark.parametrize("exponent", [0.5, 0.75, 1.5, 2.0])
def test_temperature_preserves_order_and_sum(
    routing: tuple[torch.Tensor, torch.Tensor], exponent: float
) -> None:
    weights, _ = routing

    transformed = apply(
        routing,
        QwenExpertWeightPolicy(mode="temperature", exponent=exponent),
    )

    assert torch.all(transformed[..., :-1] >= transformed[..., 1:])
    torch.testing.assert_close(transformed.sum(-1), weights.sum(-1))
    original_ratio = weights[:, 0] / weights[:, -1]
    transformed_ratio = transformed[:, 0] / transformed[:, -1]
    if exponent < 1:
        assert torch.all(transformed_ratio < original_ratio)
    else:
        assert torch.all(transformed_ratio > original_ratio)


def test_temperature_endpoints_are_identity_and_uniform(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing

    identity = apply(
        routing,
        QwenExpertWeightPolicy(mode="temperature", exponent=1),
    )
    uniform = apply(
        routing,
        QwenExpertWeightPolicy(mode="temperature", exponent=0),
    )

    assert identity is weights
    torch.testing.assert_close(uniform, torch.full_like(weights, 1 / 8))


def test_fixed_rank_profile_removes_token_specific_weight_values(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing
    profile = (0.32, 0.20, 0.14, 0.10, 0.08, 0.07, 0.05, 0.04)

    transformed = apply(
        routing,
        QwenExpertWeightPolicy(mode="rank_profile", profile=profile),
    )

    expected = torch.tensor(profile).expand_as(weights)
    torch.testing.assert_close(transformed, expected)
    torch.testing.assert_close(transformed[0], transformed[1])
    torch.testing.assert_close(transformed.sum(-1), weights.sum(-1))


def test_adaptive_mass_uses_smallest_prefix_and_renormalizes(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing

    transformed = apply(
        routing,
        QwenExpertWeightPolicy(mode="adaptive_mass", mass_threshold=0.5, renormalize=True),
    )

    # Row 0 reaches 0.50 at K=2; row 1 reaches 0.60 at K=3.
    assert torch.count_nonzero(transformed[0]).item() == 2
    assert torch.count_nonzero(transformed[1]).item() == 3
    torch.testing.assert_close(transformed.sum(-1), weights.sum(-1))
    assert torch.all(transformed[:, 3:] == 0)


def test_adaptive_mass_can_preserve_selected_mass(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, _ = routing

    transformed = apply(
        routing,
        QwenExpertWeightPolicy(mode="adaptive_mass", mass_threshold=0.5, renormalize=False),
    )

    torch.testing.assert_close(transformed[0, :2], weights[0, :2])
    torch.testing.assert_close(transformed[1, :3], weights[1, :3])
    torch.testing.assert_close(transformed.sum(-1), torch.tensor([0.50, 0.60]))


def test_adaptive_mass_honors_min_and_max_k(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    minimum = apply(
        routing,
        QwenExpertWeightPolicy(
            mode="adaptive_mass",
            mass_threshold=0.1,
            min_k=4,
            max_k=8,
            renormalize=False,
        ),
    )
    maximum = apply(
        routing,
        QwenExpertWeightPolicy(
            mode="adaptive_mass",
            mass_threshold=1.0,
            min_k=1,
            max_k=4,
            renormalize=False,
        ),
    )

    assert torch.all(torch.count_nonzero(minimum, dim=-1) == 4)
    assert torch.all(torch.count_nonzero(maximum, dim=-1) == 4)


@pytest.mark.parametrize("renormalize", [False, True])
def test_truncate_keeps_fixed_prefix(
    routing: tuple[torch.Tensor, torch.Tensor], renormalize: bool
) -> None:
    weights, _ = routing

    transformed = apply(
        routing,
        QwenExpertWeightPolicy(mode="truncate", keep_k=4, renormalize=renormalize),
    )

    assert torch.all(transformed[:, 4:] == 0)
    if renormalize:
        torch.testing.assert_close(transformed.sum(-1), weights.sum(-1))
        torch.testing.assert_close(
            transformed[:, 0] / transformed[:, 1],
            weights[:, 0] / weights[:, 1],
        )
    else:
        torch.testing.assert_close(transformed[:, :4], weights[:, :4])


def test_k12_without_renormalization_preserves_k8_scale() -> None:
    weights = torch.tensor(
        [[0.20, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.04, 0.025, 0.02, 0.015]]
    )
    expert_ids = torch.arange(12).unsqueeze(0)
    unnormalized = transform_topk_weights(
        weights,
        expert_ids,
        policy=QwenExpertWeightPolicy(
            mode="truncate",
            router_k=12,
            keep_k=12,
            reference_k=8,
            renormalize=False,
        ),
    )
    native = transform_topk_weights(
        weights,
        expert_ids,
        policy=QwenExpertWeightPolicy(
            mode="truncate",
            router_k=12,
            keep_k=12,
            reference_k=8,
            renormalize=True,
        ),
    )

    torch.testing.assert_close(unnormalized[:, :8].sum(-1), weights.sum(-1))
    assert torch.all(unnormalized.sum(-1) > weights.sum(-1))
    torch.testing.assert_close(native, weights)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        (
            {
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE": "temperature",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_EXPONENT": "0.75",
            },
            QwenExpertWeightPolicy(mode="temperature", exponent=0.75),
        ),
        (
            {
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE": "rank_profile",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_RANK_PROFILE": "8,7,6,5,4,3,2,1",
            },
            QwenExpertWeightPolicy(mode="rank_profile", profile=(8, 7, 6, 5, 4, 3, 2, 1)),
        ),
        (
            {
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE": "adaptive_mass",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_MASS_THRESHOLD": "0.7",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_MIN_K": "2",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_MAX_K": "6",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE": "false",
            },
            QwenExpertWeightPolicy(
                mode="adaptive_mass",
                mass_threshold=0.7,
                min_k=2,
                max_k=6,
                renormalize=False,
            ),
        ),
        (
            {
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE": "truncate",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K": "12",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K": "12",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K": "8",
                "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE": "false",
            },
            QwenExpertWeightPolicy(
                mode="truncate",
                router_k=12,
                keep_k=12,
                reference_k=8,
                renormalize=False,
            ),
        ),
    ],
)
def test_policy_from_environment(environ: dict[str, str], expected: QwenExpertWeightPolicy) -> None:
    assert qwen_expert_weight_policy_from_env(environ) == expected


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(lambda: QwenExpertWeightPolicy(mode="reverse"), id="unknown-mode"),
        pytest.param(
            lambda: QwenExpertWeightPolicy(mode="temperature", exponent=-1),
            id="negative-exponent",
        ),
        pytest.param(
            lambda: QwenExpertWeightPolicy(mode="rank_profile", profile=(1,) * 7),
            id="short-profile",
        ),
        pytest.param(
            lambda: QwenExpertWeightPolicy(mode="adaptive_mass", mass_threshold=0),
            id="zero-threshold",
        ),
        pytest.param(
            lambda: QwenExpertWeightPolicy(
                mode="adaptive_mass", mass_threshold=0.5, min_k=5, max_k=4
            ),
            id="invalid-bounds",
        ),
        pytest.param(
            lambda: QwenExpertWeightPolicy(mode="truncate", keep_k=9),
            id="invalid-keep-k",
        ),
        pytest.param(
            lambda: QwenExpertWeightPolicy(mode="truncate", router_k=12, keep_k=12, reference_k=13),
            id="invalid-reference-k",
        ),
    ],
)
def test_invalid_policy_is_rejected(policy: object) -> None:
    with pytest.raises(ValueError):
        policy()  # type: ignore[operator]


def test_invalid_shape_and_top_k_are_rejected(
    routing: tuple[torch.Tensor, torch.Tensor],
) -> None:
    weights, expert_ids = routing
    policy = QwenExpertWeightPolicy(mode="shuffle")

    with pytest.raises(ValueError, match="shape mismatch"):
        transform_topk_weights(weights, expert_ids[:, :-1], policy=policy)
    with pytest.raises(ValueError, match="expected top-8"):
        transform_topk_weights(weights[:, :-1], expert_ids[:, :-1], policy=policy)

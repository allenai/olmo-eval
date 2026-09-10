"""Pure weight policies for Qwen's already-selected, normalized top-8 experts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

QWEN_TOP_K = 8
VALID_EXPERT_WEIGHT_MODES = frozenset(
    {
        "normal",
        "shuffle",
        "uniform",
        "temperature",
        "rank_profile",
        "adaptive_mass",
        "truncate",
    }
)


@dataclass(frozen=True)
class QwenExpertWeightPolicy:
    """Configuration for one mutually exclusive top-8 weight intervention."""

    mode: str
    seed: int = 0
    exponent: float | None = None
    profile: tuple[float, ...] | None = None
    mass_threshold: float | None = None
    min_k: int = 1
    max_k: int = QWEN_TOP_K
    keep_k: int | None = None
    router_k: int = QWEN_TOP_K
    reference_k: int = QWEN_TOP_K
    renormalize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in VALID_EXPERT_WEIGHT_MODES:
            raise ValueError(
                f"unsupported expert-weight mode {self.mode!r}; "
                f"expected one of {sorted(VALID_EXPERT_WEIGHT_MODES)}"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.mode == "temperature":
            if self.exponent is None or not math.isfinite(self.exponent):
                raise ValueError("temperature mode requires a finite exponent")
            if self.exponent < 0:
                raise ValueError("temperature exponent must be non-negative")
        if self.mode == "rank_profile":
            if self.profile is None or len(self.profile) != QWEN_TOP_K:
                raise ValueError("rank_profile mode requires exactly eight weights")
            if any(not math.isfinite(value) or value < 0 for value in self.profile):
                raise ValueError("rank-profile weights must be finite and non-negative")
            if sum(self.profile) <= 0:
                raise ValueError("rank-profile weights must have a positive sum")
        if self.mode == "adaptive_mass":
            if (
                self.mass_threshold is None
                or not math.isfinite(self.mass_threshold)
                or not 0 < self.mass_threshold <= 1
            ):
                raise ValueError("adaptive-mass threshold must be in (0, 1]")
            if not 1 <= self.min_k <= self.max_k <= QWEN_TOP_K:
                raise ValueError("adaptive-mass bounds must satisfy 1 <= min_k <= max_k <= 8")
        if self.mode != "truncate" and self.router_k != QWEN_TOP_K:
            raise ValueError("only truncate mode supports router_k other than 8")
        if self.mode == "truncate":
            if self.keep_k is None or not 1 <= self.keep_k <= self.router_k:
                raise ValueError("truncate mode requires 1 <= keep_k <= router_k")
            if not 1 <= self.reference_k <= self.router_k:
                raise ValueError("truncate mode requires 1 <= reference_k <= router_k")

    def describe(self) -> str:
        if self.mode == "shuffle":
            return f"shuffle(seed={self.seed})"
        if self.mode == "temperature":
            return f"temperature(exponent={self.exponent:g})"
        if self.mode == "rank_profile":
            values = ",".join(f"{value:g}" for value in self.profile or ())
            return f"rank_profile([{values}])"
        if self.mode == "adaptive_mass":
            return (
                f"adaptive_mass(threshold={self.mass_threshold:g},"
                f"min_k={self.min_k},max_k={self.max_k},"
                f"renormalize={self.renormalize})"
            )
        if self.mode == "truncate":
            return (
                f"truncate(router_k={self.router_k},keep_k={self.keep_k},"
                f"reference_k={self.reference_k},renormalize={self.renormalize})"
            )
        return self.mode


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _parse_profile(value: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())


def qwen_expert_weight_policy_from_env(
    environ: Mapping[str, str],
) -> QwenExpertWeightPolicy:
    """Build and validate one policy from the vLLM process environment."""
    mode = environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE", "").strip().lower()
    if not mode:
        raise ValueError("Qwen expert-weight mode is not set")

    kwargs: dict[str, object] = {
        "mode": mode,
        "seed": int(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED", "0")),
    }
    if mode == "temperature":
        kwargs["exponent"] = float(environ["OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_EXPONENT"])
    elif mode == "rank_profile":
        kwargs["profile"] = _parse_profile(environ["OLMO_EVAL_VLLM_QWEN_EXPERT_RANK_PROFILE"])
    elif mode == "adaptive_mass":
        kwargs.update(
            mass_threshold=float(environ["OLMO_EVAL_VLLM_QWEN_EXPERT_MASS_THRESHOLD"]),
            min_k=int(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_MIN_K", "1")),
            max_k=int(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_MAX_K", "8")),
            renormalize=_parse_bool(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE", "true")),
        )
    elif mode == "truncate":
        kwargs.update(
            keep_k=int(environ["OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K"]),
            router_k=int(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K", "8")),
            reference_k=int(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K", "8")),
            renormalize=_parse_bool(environ.get("OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE", "true")),
        )
    return QwenExpertWeightPolicy(**kwargs)  # type: ignore[arg-type]


def _renormalize_to_reference_sum(
    values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    work = values.to(torch.float32)
    target_sum = reference.to(torch.float32).sum(dim=-1, keepdim=True)
    denominator = work.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return (work / denominator * target_sum).to(reference.dtype)


def temperature_weights(weights: torch.Tensor, *, exponent: float) -> torch.Tensor:
    """Flatten (p < 1) or sharpen (p > 1) weights without changing rank."""
    if exponent == 1:
        return weights
    powered = weights.to(torch.float32).pow(exponent)
    return _renormalize_to_reference_sum(powered, weights)


def fixed_rank_profile_weights(
    weights: torch.Tensor,
    *,
    profile: Sequence[float],
) -> torch.Tensor:
    """Replace token-specific values with one fixed profile assigned by rank."""
    # Avoid weights.new_tensor(profile): its host-to-device copy is forbidden
    # when vLLM invokes the router while capturing a CUDA graph.
    profile_tensor = torch.stack(
        [torch.full_like(weights[..., 0], value) for value in profile],
        dim=-1,
    )
    return _renormalize_to_reference_sum(profile_tensor, weights)


def adaptive_mass_weights(
    weights: torch.Tensor,
    *,
    threshold: float,
    min_k: int = 1,
    max_k: int = QWEN_TOP_K,
    renormalize: bool = True,
) -> torch.Tensor:
    """Keep the smallest rank prefix reaching a cumulative top-8 mass threshold."""
    work = weights.to(torch.float32)
    normalized = work / work.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    cumulative_before = normalized.cumsum(dim=-1) - normalized
    ranks = torch.arange(weights.shape[-1], device=weights.device)
    keep = (ranks < min_k) | ((ranks < max_k) & (cumulative_before < threshold))
    selected = weights * keep.to(weights.dtype)
    if renormalize:
        return _renormalize_to_reference_sum(selected, weights)
    return selected


def truncated_weights(
    weights: torch.Tensor,
    *,
    keep_k: int,
    reference_k: int,
    renormalize: bool,
) -> torch.Tensor:
    """Keep a rank prefix, optionally preserving a reference-prefix scale."""
    ranks = torch.arange(weights.shape[-1], device=weights.device)
    selected = weights
    if renormalize:
        selected = selected * (ranks < keep_k).to(weights.dtype)
        return _renormalize_to_reference_sum(selected, weights)

    # vLLM has normalized the full router-K row. Rescaling by the reference
    # prefix makes that prefix sum to the original row sum. For router_k=12,
    # reference_k=8 this preserves native K=8 scale while ranks 9--12 add mass.
    work = weights.to(torch.float32)
    target_sum = work.sum(dim=-1, keepdim=True)
    reference_sum = (
        work[..., :reference_k].sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    )
    reference_scaled = (work * target_sum / reference_sum).to(weights.dtype)
    return reference_scaled * (ranks < keep_k).to(weights.dtype)


def _hash_expert_ids(
    topk_ids: torch.Tensor,
    *,
    seed: int,
    layer_index: int,
) -> torch.Tensor:
    """Return deterministic pseudo-random keys without using global RNG state."""
    mask = 0x7FFFFFFF
    salt = ((seed & mask) ^ (((layer_index + 1) * 0x1E3779B1) & mask)) & mask
    values = (topk_ids.to(torch.int64) + salt) & mask
    values = ((values ^ (values >> 16)) * 0x045D9F3B) & mask
    values = ((values ^ (values >> 16)) * 0x045D9F3B) & mask
    return (values ^ (values >> 16)) & mask


def transform_topk_weights(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    policy: QwenExpertWeightPolicy,
    layer_index: int = 0,
) -> torch.Tensor:
    """Apply one policy while leaving the selected expert IDs untouched."""
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(f"weight/id shape mismatch: {topk_weights.shape} != {topk_ids.shape}")
    if topk_weights.shape[-1] != policy.router_k:
        raise ValueError(f"expected top-{policy.router_k} weights, got {topk_weights.shape[-1]}")

    if policy.mode == "normal":
        return topk_weights
    if policy.mode == "uniform":
        return fixed_rank_profile_weights(topk_weights, profile=(1.0,) * QWEN_TOP_K)
    if policy.mode == "shuffle":
        keys = _hash_expert_ids(
            topk_ids,
            seed=policy.seed,
            layer_index=layer_index,
        )
        permutation = torch.argsort(keys, dim=-1, stable=True)
        return torch.gather(topk_weights, dim=-1, index=permutation)
    if policy.mode == "temperature":
        assert policy.exponent is not None
        return temperature_weights(topk_weights, exponent=policy.exponent)
    if policy.mode == "rank_profile":
        assert policy.profile is not None
        return fixed_rank_profile_weights(topk_weights, profile=policy.profile)
    if policy.mode == "adaptive_mass":
        assert policy.mass_threshold is not None
        return adaptive_mass_weights(
            topk_weights,
            threshold=policy.mass_threshold,
            min_k=policy.min_k,
            max_k=policy.max_k,
            renormalize=policy.renormalize,
        )
    assert policy.mode == "truncate" and policy.keep_k is not None
    return truncated_weights(
        topk_weights,
        keep_k=policy.keep_k,
        reference_k=policy.reference_k,
        renormalize=policy.renormalize,
    )

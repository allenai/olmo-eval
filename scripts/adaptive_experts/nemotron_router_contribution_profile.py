#!/usr/bin/env python3
"""Profile BF16 Nemotron-3-Super routing and expert contributions at native K=22.

The script records the trajectory during autoregressive generation. During the
prompt prefill and token-by-token decode it records:

* sigmoid router-score concentration, both against all 512 routed experts and
  within the selected 22;
* individual routed-expert contributions after the LatentMoE output projection;
* the always-on shared-expert output separately from the routed mixture; and
* normalized and reference-preserving smaller-K counterfactual mixtures.

The Hugging Face reference MoE calls every unused expert on a zero tensor so that
training systems regard all parameters as used.  This script replaces only that
inference-time loop with an algebraically equivalent selected-expert loop.  The
smoke test validates the resulting routed and total MoE reconstructions.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import platform
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PHASES = ("prompt", "reasoning", "final")
K_VALUES = (1, 2, 4, 8, 11, 13, 15, 17, 19, 21, 22)
NATIVE_K = 22

ROUTER_K_METRICS = (
    "selected_score_share",
    "all_sigmoid_score_share",
    "renormalization_multiplier",
)
ROUTER_SCALAR_METRICS = (
    "selected_entropy",
    "selected_effective_experts",
    "selected_score_sum",
    "all_sigmoid_score_sum",
    "raw_top22_selection_overlap",
    "actual_weight_sum",
)
RANK_METRICS = (
    "gate_share",
    "actual_gate_weight",
    "expert_output_norm",
    "weighted_contribution_norm",
    "contribution_norm_share",
    "projection_share_of_routed_output",
)
CONTRIBUTION_K_METRICS = (
    "cumulative_gate_share",
    "cumulative_contribution_norm_share",
    "reference_routed_cosine",
    "reference_routed_relative_l2",
    "reference_routed_norm_ratio",
    "renormalized_routed_cosine",
    "renormalized_routed_relative_l2",
    "renormalized_routed_norm_ratio",
    "reference_total_cosine",
    "reference_total_relative_l2",
    "renormalized_total_cosine",
    "renormalized_total_relative_l2",
)
CONTRIBUTION_SCALAR_METRICS = (
    "routed_output_norm",
    "shared_output_norm",
    "total_moe_output_norm",
    "routed_to_shared_norm_ratio",
    "contribution_coherence_ratio",
    "routed_reconstruction_relative_l2",
    "total_reconstruction_relative_l2",
)


def install_mamba_rmsnorm_fallback() -> bool:
    """Provide the one Mamba symbol required by Nemotron's torch-only path.

    Nemotron's remote modeling module imports ``rmsnorm_fn`` unconditionally,
    even when ``config.use_mamba_kernels`` is false. The optimized
    ``mamba-ssm`` package is not required by the subsequent pure-PyTorch state
    space implementation, so avoid compiling it merely to satisfy this import.
    """
    try:
        importlib.import_module("mamba_ssm.ops.triton.layernorm_gated")
        return False
    except ImportError:
        pass

    def rmsnorm_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None = None,
        eps: float = 1e-6,
        group_size: int | None = None,
        norm_before_gate: bool = True,
    ) -> torch.Tensor:
        dtype = x.dtype
        value = x.float()
        gate = z.float() if z is not None else None
        if gate is not None and not norm_before_gate:
            value = value * F.silu(gate)
        size = group_size or value.shape[-1]
        grouped = value.reshape(*value.shape[:-1], -1, size)
        grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        output = grouped.reshape_as(value) * weight.float()
        if bias is not None:
            output = output + bias.float()
        if gate is not None and norm_before_gate:
            output = output * F.silu(gate)
        return output.to(dtype)

    module_names = (
        "mamba_ssm",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm_gated",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for name in module_names[:-1]:
        modules[name].__path__ = []  # type: ignore[attr-defined]
    for name, module in modules.items():
        module.__spec__ = importlib.machinery.ModuleSpec(
            name,
            loader=None,
            is_package=name != module_names[-1],
        )
    modules[module_names[-1]].rmsnorm_fn = rmsnorm_fn  # type: ignore[attr-defined]
    modules["mamba_ssm"].ops = modules["mamba_ssm.ops"]  # type: ignore[attr-defined]
    modules["mamba_ssm.ops"].triton = modules["mamba_ssm.ops.triton"]  # type: ignore[attr-defined]
    modules["mamba_ssm.ops.triton"].layernorm_gated = modules[module_names[-1]]  # type: ignore[attr-defined]
    sys.modules.update(modules)
    # The stub intentionally supplies only the gated RMSNorm needed by the
    # torch path. Prevent Transformers from mistaking it for the complete
    # optimized mamba-ssm distribution.
    try:
        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils.is_mamba_2_ssm_available = lambda: False
    except ImportError:
        pass
    return True


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return (left * right).sum(dim=-1) / denominator.clamp_min(1e-12)


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(candidate - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(1e-12)


def router_metric_tensors(
    *,
    scores: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K metrics, scalar metrics, and descending selected-weight order."""
    scores = scores.float()
    selected_weights = selected_weights.float()
    order = torch.argsort(selected_weights, dim=-1, descending=True)
    sorted_ids = selected_ids.gather(-1, order)
    sorted_weights = selected_weights.gather(-1, order)
    sorted_scores = scores.gather(-1, sorted_ids)

    selected_score_sum = sorted_scores.sum(dim=-1)
    all_score_sum = scores.sum(dim=-1)
    selected_cumulative = sorted_scores.cumsum(dim=-1)
    positions = torch.tensor([k - 1 for k in K_VALUES], device=scores.device, dtype=torch.long)
    selected_share = selected_cumulative.index_select(-1, positions) / selected_score_sum[
        ..., None
    ].clamp_min(1e-12)
    all_share = selected_cumulative.index_select(-1, positions) / all_score_sum[
        ..., None
    ].clamp_min(1e-12)
    k_values = torch.stack(
        (selected_share, all_share, 1.0 / selected_share.clamp_min(1e-12)), dim=-1
    )

    selected_distribution = sorted_scores / selected_score_sum[..., None].clamp_min(1e-12)
    entropy = -(selected_distribution * selected_distribution.clamp_min(1e-30).log()).sum(dim=-1)
    raw_top_ids = torch.topk(scores, NATIVE_K, dim=-1).indices
    overlap = (
        (sorted_ids[..., :, None] == raw_top_ids[..., None, :]).any(dim=-1).float().mean(dim=-1)
    )
    scalars = torch.stack(
        (
            entropy,
            entropy.exp(),
            selected_score_sum,
            all_score_sum,
            overlap,
            sorted_weights.sum(dim=-1),
        ),
        dim=-1,
    )
    return k_values, scalars, order


def contribution_metric_tensors(
    *,
    selected_weights: torch.Tensor,
    projected_contributions: torch.Tensor,
    routed_projection_bias: torch.Tensor,
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    total_output: torch.Tensor,
    order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rank, K, scalar, and validation tensors for each token."""
    weights = selected_weights.float().gather(-1, order)
    gather_index = order[..., None].expand(*order.shape, projected_contributions.shape[-1])
    contribution = projected_contributions.float().gather(-2, gather_index)
    routed_bias = routed_projection_bias.float()
    routed = routed_output.float()
    shared = shared_output.float()
    total = total_output.float()

    weight_sum = weights.sum(dim=-1)
    gate_share = weights / weight_sum[..., None].clamp_min(1e-12)
    contribution_norm = torch.linalg.vector_norm(contribution, dim=-1)
    expert_output_norm = contribution_norm / weights.clamp_min(1e-12)
    contribution_norm_sum = contribution_norm.sum(dim=-1)
    contribution_norm_share = contribution_norm / contribution_norm_sum[..., None].clamp_min(1e-12)
    routed_norm = torch.linalg.vector_norm(routed, dim=-1)
    projection_share = (contribution * routed[..., None, :]).sum(dim=-1) / routed_norm[
        ..., None
    ].square().clamp_min(1e-12)
    rank_values = torch.stack(
        (
            gate_share,
            weights,
            expert_output_norm,
            contribution_norm,
            contribution_norm_share,
            projection_share,
        ),
        dim=-1,
    )

    cumulative = contribution.cumsum(dim=-2)
    cumulative_gate = gate_share.cumsum(dim=-1)
    positions = torch.tensor([k - 1 for k in K_VALUES], device=weights.device, dtype=torch.long)
    reference_routed = cumulative.index_select(-2, positions) + routed_bias[..., None, :]
    retained_gate = cumulative_gate.index_select(-1, positions)
    renormalized_routed = (
        cumulative.index_select(-2, positions)
        / retained_gate[..., None].clamp_min(1e-12)
        + routed_bias[..., None, :]
    )
    reference_total = reference_routed + shared[..., None, :]
    renormalized_total = renormalized_routed + shared[..., None, :]
    cumulative_contribution_share = contribution_norm.cumsum(dim=-1).index_select(
        -1, positions
    ) / contribution_norm_sum[..., None].clamp_min(1e-12)
    reference_norm = torch.linalg.vector_norm(reference_routed, dim=-1)
    renormalized_norm = torch.linalg.vector_norm(renormalized_routed, dim=-1)
    k_values = torch.stack(
        (
            retained_gate,
            cumulative_contribution_share,
            safe_cosine(reference_routed, routed[..., None, :]),
            relative_l2(reference_routed, routed[..., None, :]),
            reference_norm / routed_norm[..., None].clamp_min(1e-12),
            safe_cosine(renormalized_routed, routed[..., None, :]),
            relative_l2(renormalized_routed, routed[..., None, :]),
            renormalized_norm / routed_norm[..., None].clamp_min(1e-12),
            safe_cosine(reference_total, total[..., None, :]),
            relative_l2(reference_total, total[..., None, :]),
            safe_cosine(renormalized_total, total[..., None, :]),
            relative_l2(renormalized_total, total[..., None, :]),
        ),
        dim=-1,
    )

    reconstructed_routed = contribution.sum(dim=-2) + routed_bias
    reconstructed_total = reconstructed_routed + shared
    routed_reconstruction = relative_l2(reconstructed_routed, routed)
    total_reconstruction = relative_l2(reconstructed_total, total)
    shared_norm = torch.linalg.vector_norm(shared, dim=-1)
    total_norm = torch.linalg.vector_norm(total, dim=-1)
    scalars = torch.stack(
        (
            routed_norm,
            shared_norm,
            total_norm,
            routed_norm / shared_norm.clamp_min(1e-12),
            routed_norm / contribution_norm_sum.clamp_min(1e-12),
            routed_reconstruction,
            total_reconstruction,
        ),
        dim=-1,
    )
    validation = torch.stack(
        (
            (weight_sum - 5.0).abs(),
            routed_reconstruction,
            total_reconstruction,
            (reference_routed[..., -1, :] - routed).abs().amax(dim=-1),
            (reference_total[..., -1, :] - total).abs().amax(dim=-1),
        ),
        dim=-1,
    )
    return rank_values, k_values, scalars, validation


class ProfileStats:
    """Online sufficient statistics indexed by phase and MoE-layer position."""

    def __init__(self, num_moe_layers: int):
        shape = (len(PHASES), num_moe_layers)
        self.count = np.zeros(shape, dtype=np.int64)
        self.router_k_sum = np.zeros(
            (*shape, len(K_VALUES), len(ROUTER_K_METRICS)), dtype=np.float64
        )
        self.router_k_sumsq = np.zeros_like(self.router_k_sum)
        self.router_scalar_sum = np.zeros((*shape, len(ROUTER_SCALAR_METRICS)), dtype=np.float64)
        self.router_scalar_sumsq = np.zeros_like(self.router_scalar_sum)
        self.rank_sum = np.zeros((*shape, NATIVE_K, len(RANK_METRICS)), dtype=np.float64)
        self.rank_sumsq = np.zeros_like(self.rank_sum)
        self.contribution_k_sum = np.zeros(
            (*shape, len(K_VALUES), len(CONTRIBUTION_K_METRICS)), dtype=np.float64
        )
        self.contribution_k_sumsq = np.zeros_like(self.contribution_k_sum)
        self.contribution_scalar_sum = np.zeros(
            (*shape, len(CONTRIBUTION_SCALAR_METRICS)), dtype=np.float64
        )
        self.contribution_scalar_sumsq = np.zeros_like(self.contribution_scalar_sum)

    def add(
        self,
        *,
        phase: str,
        layer: int,
        router_k: torch.Tensor,
        router_scalar: torch.Tensor,
        rank: torch.Tensor,
        contribution_k: torch.Tensor,
        contribution_scalar: torch.Tensor,
    ) -> None:
        phase_id = PHASES.index(phase)
        values = (
            ("router_k", router_k),
            ("router_scalar", router_scalar),
            ("rank", rank),
            ("contribution_k", contribution_k),
            ("contribution_scalar", contribution_scalar),
        )
        token_count = int(router_k.shape[0])
        self.count[phase_id, layer] += token_count
        for name, tensor in values:
            tensor = tensor.detach()
            summed = tensor.sum(dim=0).double().cpu().numpy()
            squared = tensor.square().sum(dim=0).double().cpu().numpy()
            getattr(self, f"{name}_sum")[phase_id, layer] += summed
            getattr(self, f"{name}_sumsq")[phase_id, layer] += squared

    def merge(self, other: ProfileStats) -> None:
        for name in (
            "count",
            "router_k_sum",
            "router_k_sumsq",
            "router_scalar_sum",
            "router_scalar_sumsq",
            "rank_sum",
            "rank_sumsq",
            "contribution_k_sum",
            "contribution_k_sumsq",
            "contribution_scalar_sum",
            "contribution_scalar_sumsq",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            **{
                name: getattr(self, name)
                for name in (
                    "count",
                    "router_k_sum",
                    "router_k_sumsq",
                    "router_scalar_sum",
                    "router_scalar_sumsq",
                    "rank_sum",
                    "rank_sumsq",
                    "contribution_k_sum",
                    "contribution_k_sumsq",
                    "contribution_scalar_sum",
                    "contribution_scalar_sumsq",
                )
            },
        )

    def summary(self) -> dict[str, Any]:
        denominator = np.maximum(self.count, 1)

        def moments(name: str, extra_dims: int) -> dict[str, Any]:
            den = denominator[(...,) + (None,) * extra_dims]
            mean = getattr(self, f"{name}_sum") / den
            variance = np.maximum(getattr(self, f"{name}_sumsq") / den - mean**2, 0)
            return {"mean": mean.tolist(), "variance": variance.tolist()}

        return {
            "phases": PHASES,
            "k_values": K_VALUES,
            "router_k_metrics": ROUTER_K_METRICS,
            "router_scalar_metrics": ROUTER_SCALAR_METRICS,
            "rank_metrics": RANK_METRICS,
            "contribution_k_metrics": CONTRIBUTION_K_METRICS,
            "contribution_scalar_metrics": CONTRIBUTION_SCALAR_METRICS,
            "count": self.count.tolist(),
            "router_k": moments("router_k", 2),
            "router_scalar": moments("router_scalar", 1),
            "rank": moments("rank", 2),
            "contribution_k": moments("contribution_k", 2),
            "contribution_scalar": moments("contribution_scalar", 1),
        }


@dataclass
class PendingLayer:
    router_k: torch.Tensor
    router_scalar: torch.Tensor
    selected_weights: torch.Tensor
    order: torch.Tensor
    latent_contributions: torch.Tensor | None = None
    routed_output: torch.Tensor | None = None
    shared_output: torch.Tensor | None = None


class Recorder:
    def __init__(self, *, num_moe_layers: int, buffer_tokens: int = 64):
        self.num_moe_layers = num_moe_layers
        self.buffer_tokens = buffer_tokens
        self.phase: str | None = None
        self.targets: tuple[ProfileStats, ...] = ()
        self.pending: dict[int, PendingLayer] = {}
        self.buffers: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}
        self.validation = {
            "calls": 0,
            "max_gate_sum_abs_error": 0.0,
            "max_routed_reconstruction_relative_l2": 0.0,
            "max_total_reconstruction_relative_l2": 0.0,
            "max_native_k_routed_abs_error": 0.0,
            "max_native_k_total_abs_error": 0.0,
        }

    @property
    def active(self) -> bool:
        return self.phase is not None

    def begin(self, phase: str, *targets: ProfileStats) -> None:
        if self.pending:
            raise RuntimeError("recorder has unflushed layer state")
        if self.buffers:
            raise RuntimeError("recorder has unflushed metric buffers")
        self.phase = phase
        self.targets = targets

    def set_phase(self, phase: str) -> None:
        if self.pending:
            raise RuntimeError("cannot change phase during a model layer")
        self.phase = phase

    def finish(self) -> None:
        if len(self.pending) != 0:
            raise RuntimeError(f"unflushed Nemotron layers: {sorted(self.pending)}")
        for key in list(self.buffers):
            self._flush(key)
        self.phase = None
        self.targets = ()

    def _flush(self, key: tuple[str, int]) -> None:
        buffer = self.buffers.pop(key)
        phase, layer = key
        values = buffer["values"]
        concatenated = [
            torch.cat([entry[index] for entry in values], dim=0)
            for index in range(6)
        ]
        for target in buffer["targets"]:
            target.add(
                phase=phase,
                layer=layer,
                router_k=concatenated[0],
                router_scalar=concatenated[1],
                rank=concatenated[2],
                contribution_k=concatenated[3],
                contribution_scalar=concatenated[4],
            )
        maxima = concatenated[5].amax(dim=0).detach().float().cpu().tolist()
        names = (
            "max_gate_sum_abs_error",
            "max_routed_reconstruction_relative_l2",
            "max_total_reconstruction_relative_l2",
            "max_native_k_routed_abs_error",
            "max_native_k_total_abs_error",
        )
        for name, value in zip(names, maxima, strict=True):
            self.validation[name] = max(self.validation[name], float(value))

    def record_router(
        self,
        *,
        layer: int,
        scores: torch.Tensor,
        selected_ids: torch.Tensor,
        selected_weights: torch.Tensor,
    ) -> None:
        if not self.active:
            return
        router_k, router_scalar, order = router_metric_tensors(
            scores=scores,
            selected_ids=selected_ids,
            selected_weights=selected_weights,
        )
        self.pending[layer] = PendingLayer(
            router_k=router_k,
            router_scalar=router_scalar,
            selected_weights=selected_weights,
            order=order,
        )

    def record_latent(self, layer: int, contributions: torch.Tensor) -> None:
        if self.active:
            self.pending[layer].latent_contributions = contributions

    def record_routed(self, layer: int, output: torch.Tensor) -> None:
        if self.active:
            self.pending[layer].routed_output = output

    def record_shared(self, layer: int, output: torch.Tensor) -> None:
        if self.active:
            self.pending[layer].shared_output = output

    def complete_layer(
        self, *, layer: int, module: torch.nn.Module, total_output: torch.Tensor
    ) -> None:
        if not self.active:
            return
        pending = self.pending.pop(layer)
        if (
            pending.latent_contributions is None
            or pending.routed_output is None
            or pending.shared_output is None
        ):
            raise RuntimeError(f"missing contribution state for MoE layer {layer}")
        latent = pending.latent_contributions
        projection = module.fc2_latent_proj
        projected = F.linear(
            latent.reshape(-1, latent.shape[-1]),
            projection.weight,
            bias=None,
        ).reshape(
            *latent.shape[:-1],
            total_output.shape[-1],
        )
        projection_bias = (
            projection.bias
            if projection.bias is not None
            else torch.zeros(
                total_output.shape[-1],
                dtype=projected.dtype,
                device=projected.device,
            )
        )
        routed_projection_bias = projection_bias.expand(projected.shape[0], -1)
        shared = pending.shared_output.reshape(-1, total_output.shape[-1])
        total = total_output.reshape(-1, total_output.shape[-1])
        routed = pending.routed_output.reshape(-1, total_output.shape[-1])
        rank_values, k_values, scalars, validation = contribution_metric_tensors(
            selected_weights=pending.selected_weights,
            projected_contributions=projected,
            routed_projection_bias=routed_projection_bias,
            routed_output=routed,
            shared_output=shared,
            total_output=total,
            order=pending.order,
        )
        key = (self.phase or "prompt", layer)
        buffer = self.buffers.setdefault(
            key,
            {"targets": self.targets, "values": [], "tokens": 0},
        )
        buffer["values"].append(
            tuple(
                tensor.detach()
                for tensor in (
                    pending.router_k,
                    pending.router_scalar,
                    rank_values,
                    k_values,
                    scalars,
                    validation,
                )
            )
        )
        buffer["tokens"] += int(pending.router_k.shape[0])
        if buffer["tokens"] >= self.buffer_tokens:
            self._flush(key)
        self.validation["calls"] += 1


def install_instrumentation(
    model: torch.nn.Module, recorder: Recorder
) -> tuple[list[Any], list[int]]:
    """Install hooks plus the selected-expert-only inference loop."""
    handles = []
    moe_layer_indices = [
        index for index, layer in enumerate(model.model.layers) if layer.block_type == "moe"
    ]
    if len(moe_layer_indices) != recorder.num_moe_layers:
        raise ValueError("MoE layer count changed during instrumentation")

    for moe_position, model_layer_index in enumerate(moe_layer_indices):
        module = model.model.layers[model_layer_index].mixer
        original_moe = module.moe

        def selected_only_moe(
            this: torch.nn.Module,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
            *,
            position: int = moe_position,
        ) -> torch.Tensor:
            final = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
            contributions = (
                torch.zeros(
                    hidden_states.shape[0],
                    topk_indices.shape[-1],
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if recorder.active
                else None
            )
            # One device synchronization for the active set, rather than one
            # ``.item()`` synchronization per selected expert.
            active_experts = torch.unique(topk_indices).detach().cpu().tolist()
            for expert_idx in active_experts:
                token_indices, rank_indices = torch.where(topk_indices == expert_idx)
                expert_output = this.experts[expert_idx](hidden_states[token_indices])
                weighted = expert_output * topk_weights[token_indices, rank_indices, None]
                final.index_add_(0, token_indices, weighted.to(final.dtype))
                if contributions is not None:
                    contributions[token_indices, rank_indices] = weighted.to(contributions.dtype)
            if contributions is not None:
                recorder.record_latent(position, contributions)
            return final.type(hidden_states.dtype)

        module.moe = types.MethodType(selected_only_moe, module)
        module._adaptive_profile_original_moe = original_moe

        def gate_hook(
            gate: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor],
            *,
            position: int = moe_position,
        ) -> None:
            if not recorder.active:
                return
            hidden = inputs[0].reshape(-1, gate.config.hidden_size)
            logits = F.linear(hidden.float(), gate.weight.float())
            scores = logits.sigmoid()
            selected_ids, selected_weights = output
            recorder.record_router(
                layer=position,
                scores=scores,
                selected_ids=selected_ids,
                selected_weights=selected_weights,
            )

        handles.append(module.gate.register_forward_hook(gate_hook))

        def routed_hook(
            _projection: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            position: int = moe_position,
        ) -> None:
            recorder.record_routed(position, output)

        handles.append(module.fc2_latent_proj.register_forward_hook(routed_hook))

        def shared_hook(
            _shared: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            position: int = moe_position,
        ) -> None:
            recorder.record_shared(position, output)

        handles.append(module.shared_experts.register_forward_hook(shared_hook))

        def moe_hook(
            this: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            position: int = moe_position,
        ) -> None:
            recorder.complete_layer(layer=position, module=this, total_output=output)

        handles.append(module.register_forward_hook(moe_hook))
    return handles, moe_layer_indices


def read_manifest(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_examples(
    examples: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
    offset: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    selected = examples[shard_index::num_shards]
    if offset:
        selected = selected[offset:]
    return selected if limit is None else selected[:limit]


def find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start + len(pattern)
    return None


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def make_sharded_cache(model: torch.nn.Module, batch_size: int = 1) -> Any:
    """Create the custom hybrid cache with each layer's Mamba state on its device."""
    modeling_module = sys.modules[model.__class__.__module__]
    cache_class = modeling_module.NemotronHHybridDynamicCache
    input_device = module_device(model.model.embeddings)
    cache = cache_class(
        model.config,
        batch_size,
        dtype=model.dtype,
        device=input_device,
    )
    for index, layer in enumerate(model.model.layers):
        device = module_device(layer)
        cache.conv_states[index] = cache.conv_states[index].to(device)
        cache.ssm_states[index] = cache.ssm_states[index].to(device)
        cache.key_cache[index] = cache.key_cache[index].to(device)
        cache.value_cache[index] = cache.value_cache[index].to(device)
    return cache


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids


class GenerationPhaseController:
    """Assign prompt/reasoning/final phases to calls made inside ``generate``."""

    def __init__(self, recorder: Recorder, end_think_ids: list[int]):
        self.recorder = recorder
        self.end_think_ids = end_think_ids
        self.call_index = 0
        self.current_decode_phase = "reasoning"
        self.recent_ids: list[int] = []
        self.input_ids: list[int] = []

    def pre_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        input_ids = kwargs.get("input_ids")
        self.input_ids = [] if input_ids is None else input_ids.detach().reshape(-1).cpu().tolist()
        self.recorder.set_phase("prompt" if self.call_index == 0 else self.current_decode_phase)

    def post_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        if self.call_index > 0 and self.current_decode_phase == "reasoning":
            for token_id in self.input_ids:
                self.recent_ids.append(token_id)
                if len(self.recent_ids) > len(self.end_think_ids):
                    self.recent_ids.pop(0)
                if self.recent_ids == self.end_think_ids:
                    self.current_decode_phase = "final"
        self.call_index += 1

    def install(self, model: torch.nn.Module) -> tuple[Any, Any]:
        return (
            model.model.register_forward_pre_hook(self.pre_hook, with_kwargs=True),
            model.model.register_forward_hook(self.post_hook, with_kwargs=True),
        )


def load_model(args: argparse.Namespace) -> tuple[Any, Any, float]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    install_mamba_rmsnorm_fallback()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.hf_cache, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(args.model, cache_dir=args.hf_cache, trust_remote_code=True)
    config.use_mamba_kernels = False
    started = time.monotonic()
    max_memory = {index: args.max_memory_per_gpu for index in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        cache_dir=args.hf_cache,
        trust_remote_code=True,
        dtype="auto",
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    return tokenizer, model, time.monotonic() - started


def profile(args: argparse.Namespace) -> None:
    import transformers

    args.output.mkdir(parents=True, exist_ok=True)
    examples = select_examples(
        read_manifest(args.manifest),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        offset=args.offset,
        limit=args.limit,
    )
    tokenizer, model, load_seconds = load_model(args)
    if int(model.config.num_experts_per_tok) != NATIVE_K:
        raise ValueError(f"expected native K={NATIVE_K}, got {model.config.num_experts_per_tok}")
    if not bool(model.config.norm_topk_prob):
        raise ValueError("profile requires native normalized routing")
    if float(model.config.routed_scaling_factor) != 5.0:
        raise ValueError("profile expects routed_scaling_factor=5")

    moe_layer_count = sum(layer.block_type == "moe" for layer in model.model.layers)
    recorder = Recorder(num_moe_layers=moe_layer_count, buffer_tokens=args.chunk_size)
    handles, moe_layer_indices = install_instrumentation(model, recorder)
    aggregate = ProfileStats(moe_layer_count)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    input_device = module_device(model.model.embeddings)
    rows_path = args.output / "examples.jsonl.gz"

    with gzip.open(rows_path, "wt") as output_file:
        for number, example in enumerate(examples, start=1):
            started = time.monotonic()
            prompt_ids = render_prompt(tokenizer, example["messages"])
            context_limit = int(getattr(model.config, "max_position_embeddings", 262144))
            max_new_tokens = min(args.max_new_tokens, max(context_limit - prompt_ids.shape[-1], 0))
            generated: list[int] = []
            if max_new_tokens:
                torch.manual_seed(int(example.get("seed", args.seed + number)))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(example.get("seed", args.seed + number)))
                generation_cache = make_sharded_cache(model)
                controller = GenerationPhaseController(recorder, end_think_ids)
                phase_handles = controller.install(model)
                recorder.begin("prompt", aggregate)
                try:
                    with torch.inference_mode():
                        sequence = model.generate(
                            input_ids=prompt_ids.to(input_device),
                            past_key_values=generation_cache,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            use_cache=True,
                        )
                    generated = sequence[0, prompt_ids.shape[-1] :].detach().cpu().tolist()
                    # ``generate`` stops after sampling its final token, before
                    # that token is passed back through the model. Profile it
                    # once against the already-mutated generation cache.
                    if generated:
                        with torch.inference_mode():
                            model.model(
                                input_ids=torch.tensor(
                                    [[generated[-1]]], dtype=torch.long, device=input_device
                                ),
                                past_key_values=generation_cache,
                                use_cache=True,
                            )
                finally:
                    for handle in phase_handles:
                        handle.remove()
                    recorder.finish()
            row = {
                **{
                    key: example.get(key)
                    for key in ("task", "doc_id", "native_id", "baseline_score", "seed")
                },
                "prompt_tokens": int(prompt_ids.shape[-1]),
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                "elapsed_seconds": time.monotonic() - started,
            }
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            aggregate.save(args.output / "summary.partial.npz")
            (args.output / "summary.partial.json").write_text(json.dumps(aggregate.summary()))
            print(
                json.dumps(
                    {
                        "example": number,
                        "total": len(examples),
                        "key": f"{example.get('task')}:{example.get('doc_id')}",
                        "prompt_tokens": int(prompt_ids.shape[-1]),
                        "generated_tokens": len(generated),
                        "elapsed_seconds": time.monotonic() - started,
                        "validation": recorder.validation,
                    }
                ),
                flush=True,
            )

    for handle in handles:
        handle.remove()
    aggregate.save(args.output / "summary.npz")
    (args.output / "summary.json").write_text(json.dumps(aggregate.summary()))
    metadata = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "examples": len(examples),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "load_seconds": load_seconds,
        "cuda_devices": torch.cuda.device_count(),
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "moe_model_layer_indices": moe_layer_indices,
        "native_k": NATIVE_K,
        "routed_scaling_factor": float(model.config.routed_scaling_factor),
        "shared_expert": True,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": True,
        },
        "validation": recorder.validation,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if (
        recorder.validation["max_routed_reconstruction_relative_l2"] > args.max_reconstruction_error
        or recorder.validation["max_total_reconstruction_relative_l2"]
        > args.max_reconstruction_error
    ):
        raise RuntimeError(f"contribution reconstruction failed: {recorder.validation}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path("/weka/oe-eval-default/oyvindt/hf-cache"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-memory-per-gpu", default="76GiB")
    parser.add_argument("--max-reconstruction-error", type=float, default=0.03)
    return parser.parse_args()


if __name__ == "__main__":
    profile(parse_args())

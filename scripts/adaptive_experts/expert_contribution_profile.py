#!/usr/bin/env python3
"""Profile Qwen3 selected-expert activations on fixed K=8 token trajectories."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PHASES = ("prompt", "reasoning", "final")
NUM_SELECTED = 8
RANK_METRICS = (
    "gate_weight",
    "expert_output_norm",
    "weighted_contribution_norm",
    "contribution_norm_share",
    "projection_share",
    "leave_one_out_relative_l2",
    "leave_one_out_cosine",
    "contribution_rank_position",
)
K_METRICS = (
    "cumulative_gate_share",
    "cumulative_contribution_norm_share",
    "counterfactual_cosine",
    "counterfactual_relative_l2",
    "counterfactual_norm_ratio",
)
SCALAR_METRICS = (
    "moe_output_norm",
    "residual_norm",
    "moe_to_residual_norm_ratio",
    "coherence_ratio",
    "bottom4_contribution_norm_share",
    "bottom4_projection_share",
    "bottom4_top4_inversion_fraction",
    "bottom4_any_exceeds_top4",
    "gate_contribution_top4_overlap",
)


@dataclass
class PendingMoments:
    count: int
    rank_sum: torch.Tensor
    rank_sumsq: torch.Tensor
    k_sum: torch.Tensor
    k_sumsq: torch.Tensor
    scalar_sum: torch.Tensor
    scalar_sumsq: torch.Tensor
    validation: torch.Tensor


class ContributionStats:
    def __init__(self, *, num_layers: int):
        shape = (len(PHASES), num_layers)
        self.count = np.zeros(shape, dtype=np.int64)
        self.rank_sum = np.zeros((*shape, NUM_SELECTED, len(RANK_METRICS)), dtype=np.float64)
        self.rank_sumsq = np.zeros_like(self.rank_sum)
        self.k_sum = np.zeros((*shape, NUM_SELECTED, len(K_METRICS)), dtype=np.float64)
        self.k_sumsq = np.zeros_like(self.k_sum)
        self.scalar_sum = np.zeros((*shape, len(SCALAR_METRICS)), dtype=np.float64)
        self.scalar_sumsq = np.zeros_like(self.scalar_sum)

    @property
    def num_layers(self) -> int:
        return self.count.shape[1]

    def add(
        self,
        *,
        phase_id: int,
        layer: int,
        count: int,
        rank_sum: np.ndarray,
        rank_sumsq: np.ndarray,
        k_sum: np.ndarray,
        k_sumsq: np.ndarray,
        scalar_sum: np.ndarray,
        scalar_sumsq: np.ndarray,
    ) -> None:
        self.count[phase_id, layer] += count
        self.rank_sum[phase_id, layer] += rank_sum
        self.rank_sumsq[phase_id, layer] += rank_sumsq
        self.k_sum[phase_id, layer] += k_sum
        self.k_sumsq[phase_id, layer] += k_sumsq
        self.scalar_sum[phase_id, layer] += scalar_sum
        self.scalar_sumsq[phase_id, layer] += scalar_sumsq

    def merge(self, other: ContributionStats) -> None:
        for name in (
            "count",
            "rank_sum",
            "rank_sumsq",
            "k_sum",
            "k_sumsq",
            "scalar_sum",
            "scalar_sumsq",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def means(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.maximum(self.count, 1)
        return (
            self.rank_sum / denominator[..., None, None],
            self.k_sum / denominator[..., None, None],
            self.scalar_sum / denominator[..., None],
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            count=self.count,
            rank_sum=self.rank_sum,
            rank_sumsq=self.rank_sumsq,
            k_sum=self.k_sum,
            k_sumsq=self.k_sumsq,
            scalar_sum=self.scalar_sum,
            scalar_sumsq=self.scalar_sumsq,
        )

    @classmethod
    def load(cls, path: Path) -> ContributionStats:
        data = np.load(path)
        stats = cls(num_layers=data["count"].shape[1])
        for name in (
            "count",
            "rank_sum",
            "rank_sumsq",
            "k_sum",
            "k_sumsq",
            "scalar_sum",
            "scalar_sumsq",
        ):
            getattr(stats, name)[:] = data[name]
        return stats


class ContributionRecorder:
    def __init__(self, *, num_layers: int):
        self.num_layers = num_layers
        self.phase_id: int | None = None
        self.targets: tuple[ContributionStats, ...] = ()
        self.pending: dict[int, PendingMoments] = {}
        self.residual_norm: dict[int, torch.Tensor] = {}
        self.validation = {
            "calls": 0,
            "max_gate_sum_abs_error": 0.0,
            "max_output_reconstruction_abs_error": 0.0,
            "max_output_reconstruction_relative_l2": 0.0,
            "max_k8_counterfactual_abs_error": 0.0,
            "max_k8_counterfactual_relative_l2": 0.0,
        }

    @property
    def active(self) -> bool:
        return self.phase_id is not None

    def begin(self, phase: str, *targets: ContributionStats) -> None:
        if self.pending or self.residual_norm:
            raise RuntimeError("recorder has unflushed state")
        self.phase_id = PHASES.index(phase)
        self.targets = targets

    def capture_residual(self, layer: int, hidden_states: torch.Tensor) -> None:
        if self.active:
            self.residual_norm[layer] = torch.linalg.vector_norm(
                hidden_states.reshape(-1, hidden_states.shape[-1]).float(), dim=-1
            )

    def record(
        self,
        *,
        layer: int,
        gate_weights: torch.Tensor,
        contributions: torch.Tensor,
        moe_output: torch.Tensor,
    ) -> None:
        if not self.active:
            return
        if layer in self.pending:
            raise RuntimeError(f"layer {layer} recorded twice before flush")
        residual_norm = self.residual_norm.pop(layer)
        pending = compute_moments(
            gate_weights=gate_weights,
            contributions=contributions,
            moe_output=moe_output,
            residual_norm=residual_norm,
        )
        self.pending[layer] = pending

    def flush(self) -> None:
        if self.phase_id is None:
            raise RuntimeError("recorder is inactive")
        expected = set(range(self.num_layers))
        if set(self.pending) != expected:
            missing = sorted(expected - set(self.pending))
            raise RuntimeError(f"missing recorded layers: {missing}")

        layers = [self.pending[layer] for layer in range(self.num_layers)]
        rank_sum = torch.stack([row.rank_sum for row in layers]).detach().cpu().numpy()
        rank_sumsq = torch.stack([row.rank_sumsq for row in layers]).detach().cpu().numpy()
        k_sum = torch.stack([row.k_sum for row in layers]).detach().cpu().numpy()
        k_sumsq = torch.stack([row.k_sumsq for row in layers]).detach().cpu().numpy()
        scalar_sum = torch.stack([row.scalar_sum for row in layers]).detach().cpu().numpy()
        scalar_sumsq = torch.stack([row.scalar_sumsq for row in layers]).detach().cpu().numpy()
        validation = torch.stack([row.validation for row in layers]).detach().cpu().numpy()

        for layer, row in enumerate(layers):
            for target in self.targets:
                target.add(
                    phase_id=self.phase_id,
                    layer=layer,
                    count=row.count,
                    rank_sum=rank_sum[layer],
                    rank_sumsq=rank_sumsq[layer],
                    k_sum=k_sum[layer],
                    k_sumsq=k_sumsq[layer],
                    scalar_sum=scalar_sum[layer],
                    scalar_sumsq=scalar_sumsq[layer],
                )

        self.validation["calls"] += self.num_layers
        self.validation["max_gate_sum_abs_error"] = max(
            self.validation["max_gate_sum_abs_error"], float(validation[:, 0].max())
        )
        self.validation["max_output_reconstruction_abs_error"] = max(
            self.validation["max_output_reconstruction_abs_error"],
            float(validation[:, 1].max()),
        )
        self.validation["max_k8_counterfactual_abs_error"] = max(
            self.validation["max_k8_counterfactual_abs_error"],
            float(validation[:, 3].max()),
        )
        self.validation["max_output_reconstruction_relative_l2"] = max(
            self.validation["max_output_reconstruction_relative_l2"],
            float(validation[:, 2].max()),
        )
        self.validation["max_k8_counterfactual_relative_l2"] = max(
            self.validation["max_k8_counterfactual_relative_l2"],
            float(validation[:, 4].max()),
        )
        self.pending.clear()
        self.residual_norm.clear()
        self.phase_id = None
        self.targets = ()


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return (left * right).sum(dim=-1) / denominator.clamp_min(1e-12)


def compute_moments(
    *,
    gate_weights: torch.Tensor,
    contributions: torch.Tensor,
    moe_output: torch.Tensor,
    residual_norm: torch.Tensor,
) -> PendingMoments:
    weights = gate_weights.float()
    contribution = contributions.float()
    output = moe_output.float()
    contribution_norm = torch.linalg.vector_norm(contribution, dim=-1)
    expert_output_norm = contribution_norm / weights.clamp_min(1e-12)
    contribution_norm_sum = contribution_norm.sum(dim=-1)
    contribution_norm_share = contribution_norm / contribution_norm_sum[:, None].clamp_min(1e-12)
    output_norm = torch.linalg.vector_norm(output, dim=-1)
    output_norm_squared = output_norm[:, None].square().clamp_min(1e-12)
    projection_share = (contribution * output[:, None, :]).sum(dim=-1) / output_norm_squared

    contribution_order = torch.argsort(contribution_norm, dim=-1, descending=True)
    contribution_rank = torch.empty_like(contribution_order)
    positions = torch.arange(1, NUM_SELECTED + 1, device=weights.device).expand_as(
        contribution_order
    )
    contribution_rank.scatter_(1, contribution_order, positions)

    weight_sum = weights.sum(dim=-1, keepdim=True)
    leave_one_out = (output[:, None, :] - contribution) / (
        weight_sum - weights
    )[..., None].clamp_min(1e-12)
    leave_one_out_relative_l2 = torch.linalg.vector_norm(
        leave_one_out - output[:, None, :], dim=-1
    ) / output_norm[:, None].clamp_min(1e-12)
    leave_one_out_cosine = safe_cosine(leave_one_out, output[:, None, :])

    rank_values = torch.stack(
        (
            weights,
            expert_output_norm,
            contribution_norm,
            contribution_norm_share,
            projection_share,
            leave_one_out_relative_l2,
            leave_one_out_cosine,
            contribution_rank.float(),
        ),
        dim=-1,
    )

    cumulative_weight = weights.cumsum(dim=-1)
    cumulative_gate_share = cumulative_weight / weight_sum.clamp_min(1e-12)
    cumulative_contribution = contribution.cumsum(dim=1)
    counterfactual = cumulative_contribution / cumulative_weight[..., None].clamp_min(1e-12)
    cumulative_contribution_norm_share = (
        contribution_norm.cumsum(dim=-1) / contribution_norm_sum[:, None].clamp_min(1e-12)
    )
    counterfactual_norm = torch.linalg.vector_norm(counterfactual, dim=-1)
    counterfactual_cosine = safe_cosine(counterfactual, output[:, None, :])
    counterfactual_relative_l2 = torch.linalg.vector_norm(
        counterfactual - output[:, None, :], dim=-1
    ) / output_norm[:, None].clamp_min(1e-12)
    counterfactual_norm_ratio = counterfactual_norm / output_norm[:, None].clamp_min(1e-12)
    k_values = torch.stack(
        (
            cumulative_gate_share,
            cumulative_contribution_norm_share,
            counterfactual_cosine,
            counterfactual_relative_l2,
            counterfactual_norm_ratio,
        ),
        dim=-1,
    )

    bottom_top_comparisons = contribution_norm[:, 4:, None] > contribution_norm[:, None, :4]
    bottom4_top4_inversion_fraction = bottom_top_comparisons.float().mean(dim=(1, 2))
    bottom4_any_exceeds_top4 = bottom_top_comparisons.any(dim=(1, 2)).float()
    top4_overlap = (contribution_order[:, :4] < 4).float().mean(dim=-1)
    coherence_ratio = output_norm / contribution_norm_sum.clamp_min(1e-12)
    scalar_values = torch.stack(
        (
            output_norm,
            residual_norm,
            output_norm / residual_norm.clamp_min(1e-12),
            coherence_ratio,
            contribution_norm_share[:, 4:].sum(dim=-1),
            projection_share[:, 4:].sum(dim=-1),
            bottom4_top4_inversion_fraction,
            bottom4_any_exceeds_top4,
            top4_overlap,
        ),
        dim=-1,
    )

    reconstructed = contribution.sum(dim=1)
    reconstruction_delta = reconstructed - output
    k8_delta = counterfactual[:, -1] - output
    validation = torch.stack(
        (
            (weights.sum(dim=-1) - 1.0).abs().max(),
            reconstruction_delta.abs().max(),
            (
                torch.linalg.vector_norm(reconstruction_delta, dim=-1)
                / output_norm.clamp_min(1e-12)
            ).max(),
            k8_delta.abs().max(),
            (
                torch.linalg.vector_norm(k8_delta, dim=-1)
                / output_norm.clamp_min(1e-12)
            ).max(),
        )
    )
    return PendingMoments(
        count=weights.shape[0],
        rank_sum=rank_values.sum(dim=0),
        rank_sumsq=rank_values.square().sum(dim=0),
        k_sum=k_values.sum(dim=0),
        k_sumsq=k_values.square().sum(dim=0),
        scalar_sum=scalar_values.sum(dim=0),
        scalar_sumsq=scalar_values.square().sum(dim=0),
        validation=validation,
    )


def recompute_contributions(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    contributions = torch.zeros(
        (*hidden_states.shape[:-1], NUM_SELECTED, hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_tensor in expert_hit:
        expert_idx = expert_idx_tensor[0]
        if expert_idx == module.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        unweighted = module.act_fn(gate) * up
        unweighted = F.linear(unweighted, module.down_proj[expert_idx])
        weighted = unweighted * top_k_weights[token_idx, top_k_pos, None]
        contributions[token_idx, top_k_pos] = weighted.to(contributions.dtype)
    return contributions


def install_instrumentation(
    model: torch.nn.Module, recorder: ContributionRecorder
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts

        def experts_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            if not recorder.active:
                return
            hidden_states, top_k_index, top_k_weights = inputs
            contributions = recompute_contributions(
                module,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
            recorder.record(
                layer=index,
                gate_weights=top_k_weights,
                contributions=contributions,
                moe_output=output,
            )

        handles.append(experts.register_forward_hook(experts_hook))

        def residual_hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            index: int = layer_index,
        ) -> None:
            recorder.capture_residual(index, inputs[0])

        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(residual_hook))
    return handles


def read_manifest(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    for start in range(len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start + len(pattern)
    return None


def stats_payload(stats: ContributionStats) -> dict[str, Any]:
    rank_mean, k_mean, scalar_mean = stats.means()
    return {
        "phases": PHASES,
        "rank_metrics": RANK_METRICS,
        "k_metrics": K_METRICS,
        "scalar_metrics": SCALAR_METRICS,
        "count": stats.count.tolist(),
        "rank_means": rank_mean.tolist(),
        "k_means": k_mean.tolist(),
        "scalar_means": scalar_mean.tolist(),
    }


def summary_payload(stats: ContributionStats) -> dict[str, Any]:
    payload = stats_payload(stats)
    denominator = np.maximum(stats.count, 1)
    rank_mean, k_mean, scalar_mean = stats.means()
    payload.update(
        {
            "rank_variances": np.maximum(
                stats.rank_sumsq / denominator[..., None, None] - rank_mean**2, 0
            ).tolist(),
            "k_variances": np.maximum(
                stats.k_sumsq / denominator[..., None, None] - k_mean**2, 0
            ).tolist(),
            "scalar_variances": np.maximum(
                stats.scalar_sumsq / denominator[..., None] - scalar_mean**2, 0
            ).tolist(),
        }
    )
    return payload


def profile(args: argparse.Namespace) -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group(
            "gloo", timeout=timedelta(hours=args.distributed_timeout_hours)
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    all_examples = read_manifest(args.manifest)
    if args.offset:
        all_examples = all_examples[args.offset :]
    if args.limit:
        all_examples = all_examples[: args.limit]
    examples = all_examples[rank::world_size]
    output_dir = args.output / "workers" / f"rank{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.hf_cache,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    load_seconds = time.monotonic() - load_started
    num_layers = int(model.config.num_hidden_layers)
    if int(model.config.num_experts_per_tok) != NUM_SELECTED:
        raise ValueError(f"expected K={NUM_SELECTED}, got {model.config.num_experts_per_tok}")
    if getattr(model.config, "shared_expert_intermediate_size", None):
        raise ValueError("unexpected shared-expert branch in audited checkpoint")

    rendered_validation = tokenizer.apply_chat_template(
        all_examples[0]["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    validation_ids = tokenizer(
        rendered_validation, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, : args.validation_tokens].to(device)
    with torch.inference_mode():
        ordinary_logits = model(
            input_ids=validation_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()
        ordinary_repeat_logits = model(
            input_ids=validation_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()

    recorder = ContributionRecorder(num_layers=num_layers)
    hook_handles = install_instrumentation(model, recorder)
    validation_stats = ContributionStats(num_layers=num_layers)
    with torch.inference_mode():
        recorder.begin("prompt", validation_stats)
        instrumented_logits = model(
            input_ids=validation_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits.float()
        recorder.flush()
    baseline_error = (ordinary_logits - ordinary_repeat_logits).abs()
    instrumented_error = (ordinary_logits - instrumented_logits).abs()
    logit_validation = {
        "tokens": int(validation_ids.shape[-1]),
        "ordinary_repeat_max_abs_error": float(baseline_error.max().item()),
        "ordinary_repeat_mean_abs_error": float(baseline_error.mean().item()),
        "instrumented_max_abs_error": float(instrumented_error.max().item()),
        "instrumented_mean_abs_error": float(instrumented_error.mean().item()),
        "top1_match": bool(
            torch.equal(ordinary_logits.argmax(dim=-1), instrumented_logits.argmax(dim=-1))
        ),
    }
    if (
        not logit_validation["top1_match"]
        or logit_validation["instrumented_max_abs_error"] > args.max_logit_error
    ):
        raise RuntimeError(f"instrumentation validation failed: {logit_validation}")
    if (
        recorder.validation["max_output_reconstruction_relative_l2"]
        > args.max_reconstruction_error
    ):
        raise RuntimeError(f"contribution reconstruction failed: {recorder.validation}")

    aggregate = ContributionStats(num_layers=num_layers)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    examples_path = output_dir / "examples.jsonl.gz"
    with gzip.open(examples_path, "wt") as examples_file, torch.inference_mode():
        for example_number, example in enumerate(examples, start=1):
            started = time.monotonic()
            rendered = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            if prompt_ids.shape[-1] != int(example["prompt_tokens_recorded"]):
                raise ValueError(
                    f"prompt token mismatch for {example['task']}:{example['doc_id']}: "
                    f"{prompt_ids.shape[-1]} != {example['prompt_tokens_recorded']}"
                )
            generated = list(example["generated_token_ids"])
            if args.max_replay_tokens is not None:
                generated = generated[: args.max_replay_tokens]
            think_end = find_subsequence(generated, end_think_ids)
            segments = [("reasoning", generated[:think_end])]
            if think_end is not None:
                segments.append(("final", generated[think_end:]))

            per_example = ContributionStats(num_layers=num_layers)
            recorder.begin("prompt", aggregate, per_example)
            attention_mask = torch.ones_like(prompt_ids)
            outputs = model(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
            recorder.flush()
            past = outputs.past_key_values
            del outputs

            for phase, token_ids in segments:
                for start in range(0, len(token_ids), args.chunk_size):
                    chunk_ids = torch.tensor(
                        token_ids[start : start + args.chunk_size],
                        dtype=torch.long,
                        device=device,
                    )[None, :]
                    cache_start = past.get_seq_length()
                    cache_position = torch.arange(
                        cache_start,
                        cache_start + chunk_ids.shape[-1],
                        dtype=torch.long,
                        device=device,
                    )
                    recorder.begin(phase, aggregate, per_example)
                    outputs = model(
                        input_ids=chunk_ids,
                        past_key_values=past,
                        cache_position=cache_position,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    recorder.flush()
                    past = outputs.past_key_values
                    del outputs

            identity = {
                key: example[key]
                for key in ("task", "doc_id", "native_id", "baseline_score", "seed")
            }
            examples_file.write(
                json.dumps(
                    {
                        **identity,
                        "prompt_tokens": int(prompt_ids.shape[-1]),
                        "replayed_tokens": len(generated),
                        "recorded_generated_tokens": int(example["generated_tokens_recorded"]),
                        "finish_reason": example["finish_reason"],
                        "elapsed_seconds": time.monotonic() - started,
                        **stats_payload(per_example),
                    }
                )
                + "\n"
            )
            examples_file.flush()
            del past, prompt_ids, per_example
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "example": example_number,
                        "total": len(examples),
                        "key": f"{example['task']}:{example['doc_id']}",
                        "replayed_tokens": len(generated),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )

    for handle in hook_handles:
        handle.remove()
    aggregate.save(output_dir / "summary.npz")
    (output_dir / "summary.json").write_text(json.dumps(summary_payload(aggregate)))
    metadata = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "rank": rank,
        "world_size": world_size,
        "examples": len(examples),
        "load_seconds": load_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "logit_validation": logit_validation,
        "online_validation": recorder.validation,
        "chunk_size": args.chunk_size,
        "max_replay_tokens": args.max_replay_tokens,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged = ContributionStats.load(args.output / "workers" / "rank0" / "summary.npz")
        for other_rank in range(1, world_size):
            merged.merge(
                ContributionStats.load(
                    args.output / "workers" / f"rank{other_rank}" / "summary.npz"
                )
            )
        merged.save(args.output / "summary.npz")
        (args.output / "summary.json").write_text(json.dumps(summary_payload(merged)))
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "world_size": world_size,
                    "manifest_examples": len(all_examples),
                    "active_experts": NUM_SELECTED,
                    "fixed_trajectory_replay": True,
                },
                indent=2,
            )
        )
    if world_size > 1:
        torch.distributed.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path("/weka/oe-eval-default/oyvindt/hf-cache"),
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-replay-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--validation-tokens", type=int, default=32)
    parser.add_argument("--max-logit-error", type=float, default=0.02)
    parser.add_argument("--max-reconstruction-error", type=float, default=0.02)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--distributed-timeout-hours", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    profile(parse_args())

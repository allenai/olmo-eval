#!/usr/bin/env python3
"""Profile Qwen3.5 K=8 router concentration and shared-expert contribution.

The model is run autoregressively so prompt, hidden reasoning, and final-answer
tokens are all represented.  Read-only hooks collect online moments for every
MoE layer without retaining full router-logit or hidden-state trajectories.
"""

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


PHASES = ("prompt", "reasoning", "final")
NATIVE_K = 8
ROUTER_RANK_METRICS = ("raw_probability", "selected_weight")
ROUTER_K_METRICS = (
    "raw_cumulative_mass",
    "selected_cumulative_share",
    "retained_weight_renormalization",
)
ROUTER_SCALAR_METRICS = (
    "raw_top8_mass",
    "full_entropy",
    "full_effective_experts",
    "selected_entropy",
    "selected_effective_experts",
    "rank8_rank9_probability_gap",
    "rank8_rank9_logit_gap",
    "selected_weight_sum",
)
BRANCH_SCALAR_METRICS = (
    "shared_gate",
    "routed_output_norm",
    "shared_unweighted_norm",
    "shared_output_norm",
    "total_moe_output_norm",
    "routed_to_total_norm_ratio",
    "shared_to_total_norm_ratio",
    "shared_branch_norm_fraction",
    "routed_projection_share",
    "shared_projection_share",
    "routed_shared_cosine",
    "branch_coherence_ratio",
    "total_reconstruction_relative_l2",
)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return (left * right).sum(dim=-1) / denominator.clamp_min(1e-12)


@dataclass
class LayerMoments:
    count: int
    router_rank_sum: torch.Tensor
    router_rank_sumsq: torch.Tensor
    router_k_sum: torch.Tensor
    router_k_sumsq: torch.Tensor
    router_scalar_sum: torch.Tensor
    router_scalar_sumsq: torch.Tensor
    branch_scalar_sum: torch.Tensor
    branch_scalar_sumsq: torch.Tensor
    validation: torch.Tensor


class ProfileStats:
    def __init__(self, *, num_layers: int):
        shape = (len(PHASES), num_layers)
        self.count = np.zeros(shape, dtype=np.int64)
        self.router_rank_sum = np.zeros(
            (*shape, NATIVE_K, len(ROUTER_RANK_METRICS)), dtype=np.float64
        )
        self.router_rank_sumsq = np.zeros_like(self.router_rank_sum)
        self.router_k_sum = np.zeros(
            (*shape, NATIVE_K, len(ROUTER_K_METRICS)), dtype=np.float64
        )
        self.router_k_sumsq = np.zeros_like(self.router_k_sum)
        self.router_scalar_sum = np.zeros(
            (*shape, len(ROUTER_SCALAR_METRICS)), dtype=np.float64
        )
        self.router_scalar_sumsq = np.zeros_like(self.router_scalar_sum)
        self.branch_scalar_sum = np.zeros(
            (*shape, len(BRANCH_SCALAR_METRICS)), dtype=np.float64
        )
        self.branch_scalar_sumsq = np.zeros_like(self.branch_scalar_sum)

    def add_call(self, phase_id: int, rows: list[LayerMoments]) -> None:
        self.count[phase_id] += np.asarray([row.count for row in rows], dtype=np.int64)
        for name in (
            "router_rank_sum",
            "router_rank_sumsq",
            "router_k_sum",
            "router_k_sumsq",
            "router_scalar_sum",
            "router_scalar_sumsq",
            "branch_scalar_sum",
            "branch_scalar_sumsq",
        ):
            value = torch.stack([getattr(row, name) for row in rows]).detach().cpu().numpy()
            getattr(self, name)[phase_id] += value

    def merge(self, other: "ProfileStats") -> None:
        for name in (
            "count",
            "router_rank_sum",
            "router_rank_sumsq",
            "router_k_sum",
            "router_k_sumsq",
            "router_scalar_sum",
            "router_scalar_sumsq",
            "branch_scalar_sum",
            "branch_scalar_sumsq",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def means(self) -> dict[str, np.ndarray]:
        denominator = np.maximum(self.count, 1)
        return {
            "router_rank": self.router_rank_sum / denominator[..., None, None],
            "router_k": self.router_k_sum / denominator[..., None, None],
            "router_scalar": self.router_scalar_sum / denominator[..., None],
            "branch_scalar": self.branch_scalar_sum / denominator[..., None],
        }

    def payload(self, *, include_variance: bool) -> dict[str, Any]:
        means = self.means()
        result: dict[str, Any] = {
            "phases": PHASES,
            "router_rank_metrics": ROUTER_RANK_METRICS,
            "router_k_metrics": ROUTER_K_METRICS,
            "router_scalar_metrics": ROUTER_SCALAR_METRICS,
            "branch_scalar_metrics": BRANCH_SCALAR_METRICS,
            "count": self.count.tolist(),
            "router_rank_means": means["router_rank"].tolist(),
            "router_k_means": means["router_k"].tolist(),
            "router_scalar_means": means["router_scalar"].tolist(),
            "branch_scalar_means": means["branch_scalar"].tolist(),
        }
        if include_variance:
            denominator = np.maximum(self.count, 1)
            result.update(
                {
                    "router_rank_variances": np.maximum(
                        self.router_rank_sumsq / denominator[..., None, None]
                        - means["router_rank"] ** 2,
                        0,
                    ).tolist(),
                    "router_k_variances": np.maximum(
                        self.router_k_sumsq / denominator[..., None, None]
                        - means["router_k"] ** 2,
                        0,
                    ).tolist(),
                    "router_scalar_variances": np.maximum(
                        self.router_scalar_sumsq / denominator[..., None]
                        - means["router_scalar"] ** 2,
                        0,
                    ).tolist(),
                    "branch_scalar_variances": np.maximum(
                        self.branch_scalar_sumsq / denominator[..., None]
                        - means["branch_scalar"] ** 2,
                        0,
                    ).tolist(),
                }
            )
        return result

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            count=self.count,
            router_rank_sum=self.router_rank_sum,
            router_rank_sumsq=self.router_rank_sumsq,
            router_k_sum=self.router_k_sum,
            router_k_sumsq=self.router_k_sumsq,
            router_scalar_sum=self.router_scalar_sum,
            router_scalar_sumsq=self.router_scalar_sumsq,
            branch_scalar_sum=self.branch_scalar_sum,
            branch_scalar_sumsq=self.branch_scalar_sumsq,
        )

    @classmethod
    def load(cls, path: Path) -> "ProfileStats":
        data = np.load(path)
        stats = cls(num_layers=data["count"].shape[1])
        for name in (
            "count",
            "router_rank_sum",
            "router_rank_sumsq",
            "router_k_sum",
            "router_k_sumsq",
            "router_scalar_sum",
            "router_scalar_sumsq",
            "branch_scalar_sum",
            "branch_scalar_sumsq",
        ):
            getattr(stats, name)[:] = data[name]
        return stats


class Recorder:
    def __init__(self, *, num_layers: int):
        self.num_layers = num_layers
        self.phase_id: int | None = None
        self.targets: tuple[ProfileStats, ...] = ()
        self.pending: dict[int, dict[str, torch.Tensor]] = {}
        self.completed: dict[int, LayerMoments] = {}
        self.validation = {
            "calls": 0,
            "route_events": 0,
            "route_id_mismatches": 0,
            "max_selected_weight_sum_abs_error": 0.0,
            "max_selected_weight_abs_error": 0.0,
            "max_total_reconstruction_relative_l2": 0.0,
        }

    @property
    def active(self) -> bool:
        return self.phase_id is not None

    def begin(self, phase: str, *targets: ProfileStats) -> None:
        if self.pending or self.completed or self.active:
            raise RuntimeError("recorder has unflushed state")
        self.phase_id = PHASES.index(phase)
        self.targets = targets

    def put(self, layer: int, name: str, value: torch.Tensor) -> None:
        if not self.active:
            return
        row = self.pending.setdefault(layer, {})
        if name in row:
            raise RuntimeError(f"duplicate {name} capture at layer {layer}")
        row[name] = value

    def complete(self, layer: int, total_output: torch.Tensor) -> None:
        if not self.active:
            return
        row = self.pending.pop(layer)
        required = {"router_logits", "selected_weights", "selected_ids", "routed", "shared_raw", "shared_gate"}
        if set(row) != required:
            raise RuntimeError(f"layer {layer} captures {sorted(row)}, expected {sorted(required)}")
        self.completed[layer] = compute_layer_moments(
            router_logits=row["router_logits"],
            selected_weights=row["selected_weights"],
            selected_ids=row["selected_ids"],
            routed_output=row["routed"],
            shared_unweighted=row["shared_raw"],
            shared_gate_logits=row["shared_gate"],
            total_output=total_output.reshape(-1, total_output.shape[-1]),
        )

    def end(self) -> None:
        if self.phase_id is None:
            raise RuntimeError("recorder is inactive")
        if self.pending:
            raise RuntimeError(f"incomplete captures at layers {sorted(self.pending)}")
        expected = set(range(self.num_layers))
        if set(self.completed) != expected:
            raise RuntimeError(f"missing completed layers {sorted(expected - set(self.completed))}")
        rows = [self.completed[index] for index in range(self.num_layers)]
        for target in self.targets:
            target.add_call(self.phase_id, rows)
        validation = torch.stack([row.validation for row in rows]).detach().cpu().numpy()
        self.validation["calls"] += self.num_layers
        self.validation["route_events"] += int(validation[:, 0].sum())
        self.validation["route_id_mismatches"] += int(validation[:, 1].sum())
        self.validation["max_selected_weight_sum_abs_error"] = max(
            self.validation["max_selected_weight_sum_abs_error"], float(validation[:, 2].max())
        )
        self.validation["max_selected_weight_abs_error"] = max(
            self.validation["max_selected_weight_abs_error"], float(validation[:, 3].max())
        )
        self.validation["max_total_reconstruction_relative_l2"] = max(
            self.validation["max_total_reconstruction_relative_l2"], float(validation[:, 4].max())
        )
        self.completed.clear()
        self.phase_id = None
        self.targets = ()


def compute_layer_moments(
    *,
    router_logits: torch.Tensor,
    selected_weights: torch.Tensor,
    selected_ids: torch.Tensor,
    routed_output: torch.Tensor,
    shared_unweighted: torch.Tensor,
    shared_gate_logits: torch.Tensor,
    total_output: torch.Tensor,
) -> LayerMoments:
    logits = router_logits.reshape(-1, router_logits.shape[-1]).float()
    actual_weights = selected_weights.reshape(-1, selected_weights.shape[-1]).float()
    actual_ids = selected_ids.reshape(-1, selected_ids.shape[-1])
    order = torch.argsort(actual_weights, dim=-1, descending=True)
    ids = actual_ids.gather(-1, order)
    weights = actual_weights.gather(-1, order)

    probabilities = torch.softmax(logits, dim=-1)
    raw_selected = probabilities.gather(-1, ids)
    # Replay the router's exact top-k call for validation.  In BF16, logits can
    # tie at the selection boundary; CUDA is allowed to break those ties
    # differently for topk(k=8) and topk(k=9), so the latter is only suitable
    # for measuring the rank-8/rank-9 gap.
    _, expected_top8_ids = torch.topk(probabilities, NATIVE_K, dim=-1)
    raw_top_values, raw_top9_ids = torch.topk(probabilities, NATIVE_K + 1, dim=-1)
    top8_mass = raw_selected.sum(dim=-1)
    expected_weights = raw_selected / top8_mass[:, None].clamp_min(1e-12)
    rank_values = torch.stack((raw_selected, weights), dim=-1)
    selected_cumulative = weights.cumsum(dim=-1)
    k_values = torch.stack(
        (
            raw_selected.cumsum(dim=-1),
            selected_cumulative,
            1.0 / selected_cumulative.clamp_min(1e-12),
        ),
        dim=-1,
    )
    full_entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    selected_entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1)
    router_scalars = torch.stack(
        (
            top8_mass,
            full_entropy,
            full_entropy.exp(),
            selected_entropy,
            selected_entropy.exp(),
            raw_top_values[:, 7] - raw_top_values[:, 8],
            logits.gather(-1, raw_top9_ids[:, 7:8]).squeeze(-1)
            - logits.gather(-1, raw_top9_ids[:, 8:9]).squeeze(-1),
            weights.sum(dim=-1),
        ),
        dim=-1,
    )

    routed = routed_output.reshape(-1, routed_output.shape[-1]).float()
    shared_raw = shared_unweighted.reshape(-1, shared_unweighted.shape[-1]).float()
    shared_gate = torch.sigmoid(shared_gate_logits.reshape(-1).float())
    shared = shared_raw * shared_gate[:, None]
    total = total_output.float()
    reconstructed = routed + shared
    routed_norm = torch.linalg.vector_norm(routed, dim=-1)
    shared_raw_norm = torch.linalg.vector_norm(shared_raw, dim=-1)
    shared_norm = torch.linalg.vector_norm(shared, dim=-1)
    total_norm = torch.linalg.vector_norm(total, dim=-1)
    total_norm_squared = total_norm.square().clamp_min(1e-12)
    branch_scalars = torch.stack(
        (
            shared_gate,
            routed_norm,
            shared_raw_norm,
            shared_norm,
            total_norm,
            routed_norm / total_norm.clamp_min(1e-12),
            shared_norm / total_norm.clamp_min(1e-12),
            shared_norm / (routed_norm + shared_norm).clamp_min(1e-12),
            (routed * total).sum(dim=-1) / total_norm_squared,
            (shared * total).sum(dim=-1) / total_norm_squared,
            safe_cosine(routed, shared),
            total_norm / (routed_norm + shared_norm).clamp_min(1e-12),
            torch.linalg.vector_norm(reconstructed - total, dim=-1)
            / total_norm.clamp_min(1e-12),
        ),
        dim=-1,
    )

    sorted_actual_ids = torch.sort(actual_ids, dim=-1).values
    sorted_expected_ids = torch.sort(expected_top8_ids, dim=-1).values
    mismatch_rows = (~torch.eq(sorted_actual_ids, sorted_expected_ids)).any(dim=-1)
    validation = torch.stack(
        (
            torch.tensor(float(actual_ids.numel()), device=logits.device),
            mismatch_rows.float().sum(),
            (weights.sum(dim=-1) - 1.0).abs().max(),
            (weights - expected_weights).abs().max(),
            branch_scalars[:, -1].max(),
        )
    )
    return LayerMoments(
        count=logits.shape[0],
        router_rank_sum=rank_values.sum(dim=0),
        router_rank_sumsq=rank_values.square().sum(dim=0),
        router_k_sum=k_values.sum(dim=0),
        router_k_sumsq=k_values.square().sum(dim=0),
        router_scalar_sum=router_scalars.sum(dim=0),
        router_scalar_sumsq=router_scalars.square().sum(dim=0),
        branch_scalar_sum=branch_scalars.sum(dim=0),
        branch_scalar_sumsq=branch_scalars.square().sum(dim=0),
        validation=validation,
    )


def install_hooks(model: torch.nn.Module, recorder: Recorder) -> list[Any]:
    handles = []
    layers = model.model.language_model.layers
    for layer_index, layer in enumerate(layers):
        moe = layer.mlp

        def gate_hook(_module, _inputs, output, *, index=layer_index):
            if recorder.active:
                logits, scores, ids = output
                recorder.put(index, "router_logits", logits)
                recorder.put(index, "selected_weights", scores)
                recorder.put(index, "selected_ids", ids)

        handles.append(moe.gate.register_forward_hook(gate_hook))

        def routed_hook(_module, _inputs, output, *, index=layer_index):
            recorder.put(index, "routed", output)

        handles.append(moe.experts.register_forward_hook(routed_hook))

        def shared_hook(_module, _inputs, output, *, index=layer_index):
            recorder.put(index, "shared_raw", output)

        handles.append(moe.shared_expert.register_forward_hook(shared_hook))

        def shared_gate_hook(_module, _inputs, output, *, index=layer_index):
            recorder.put(index, "shared_gate", output)

        handles.append(moe.shared_expert_gate.register_forward_hook(shared_gate_hook))

        def moe_hook(_module, _inputs, output, *, index=layer_index):
            recorder.complete(index, output)

        handles.append(moe.register_forward_hook(moe_hook))
    return handles


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = logits.float() / temperature
    if top_k > 0:
        cutoff = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(sorted_scores, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_examples(
    rows: list[dict[str, Any]], *, rank: int, world_size: int, offset: int, limit: int | None
) -> list[dict[str, Any]]:
    selected = rows[rank::world_size]
    if offset:
        selected = selected[offset:]
    return selected if limit is None else selected[:limit]


def profile(args: argparse.Namespace) -> None:
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

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
    examples = select_examples(
        all_examples,
        rank=rank,
        world_size=world_size,
        offset=args.offset,
        limit=args.limit,
    )
    worker_dir = args.output / "workers" / f"rank{rank}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)
    load_started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        cache_dir=args.hf_cache,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    load_seconds = time.monotonic() - load_started
    text_config = model.config.text_config
    num_layers = int(text_config.num_hidden_layers)
    if int(text_config.num_experts_per_tok) != NATIVE_K:
        raise ValueError(f"expected K={NATIVE_K}, got {text_config.num_experts_per_tok}")
    if len(text_config.layer_types) != num_layers:
        raise ValueError("layer_types length does not match num_hidden_layers")

    aggregate = ProfileStats(num_layers=num_layers)
    recorder = Recorder(num_layers=num_layers)
    handles = install_hooks(model, recorder)

    rendered_validation = tokenizer.apply_chat_template(
        all_examples[0]["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    validation_ids = tokenizer(
        rendered_validation, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, : args.validation_tokens].to(device)
    with torch.inference_mode():
        ordinary = model(input_ids=validation_ids, use_cache=False, logits_to_keep=1).logits.float()
        validation_stats = ProfileStats(num_layers=num_layers)
        recorder.begin("prompt", validation_stats)
        instrumented = model(input_ids=validation_ids, use_cache=False, logits_to_keep=1).logits.float()
        recorder.end()
    logit_error = (ordinary - instrumented).abs()
    logit_validation = {
        "tokens": int(validation_ids.shape[-1]),
        "max_abs_error": float(logit_error.max().item()),
        "mean_abs_error": float(logit_error.mean().item()),
        "top1_match": bool(torch.equal(ordinary.argmax(dim=-1), instrumented.argmax(dim=-1))),
    }
    if not logit_validation["top1_match"] or logit_validation["max_abs_error"] > args.max_logit_error:
        raise RuntimeError(f"instrumentation changed logits: {logit_validation}")
    if recorder.validation["route_id_mismatches"]:
        raise RuntimeError(f"router ID validation failed: {recorder.validation}")
    if recorder.validation["max_selected_weight_abs_error"] > args.max_weight_error:
        raise RuntimeError(f"router weight validation failed: {recorder.validation}")
    if recorder.validation["max_total_reconstruction_relative_l2"] > args.max_reconstruction_error:
        raise RuntimeError(f"shared/routed reconstruction failed: {recorder.validation}")

    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    context_limit = int(text_config.max_position_embeddings)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    examples_path = worker_dir / "examples.jsonl.gz"
    with gzip.open(examples_path, "wt") as output_file, torch.inference_mode():
        for number, example in enumerate(examples, start=1):
            started = time.monotonic()
            rendered = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_ids = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            prompt_tokens = int(input_ids.shape[-1])
            max_new_tokens = min(args.max_new_tokens, context_limit - prompt_tokens)
            if max_new_tokens <= 0:
                raise ValueError(f"prompt exceeds context: {example['task']}:{example['doc_id']}")
            per_example = ProfileStats(num_layers=num_layers)
            recorder.begin("prompt", aggregate, per_example)
            outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
            recorder.end()
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            generator = torch.Generator(device=device).manual_seed(int(example["seed"]))
            generated: list[int] = []
            phase = "reasoning"
            finish_reason = "length"
            for generation_index in range(max_new_tokens):
                token = sample_token(
                    next_logits,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    generator=generator,
                )
                token_id = int(token.item())
                if token_id in eos_ids:
                    finish_reason = "eos"
                    break
                generated.append(token_id)
                recorder.begin(phase, aggregate, per_example)
                outputs = model(input_ids=token, past_key_values=past, use_cache=True, logits_to_keep=1)
                recorder.end()
                past = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
                if end_think_ids and generated[-len(end_think_ids) :] == end_think_ids:
                    phase = "final"

            output_file.write(
                json.dumps(
                    {
                        **{
                            key: example.get(key)
                            for key in ("task", "doc_id", "native_id", "baseline_score", "seed")
                        },
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated),
                        "finish_reason": finish_reason,
                        "generated_token_ids": generated,
                        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                        "elapsed_seconds": time.monotonic() - started,
                        **per_example.payload(include_variance=False),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            aggregate.save(worker_dir / "summary.partial.npz")
            (worker_dir / "summary.partial.json").write_text(
                json.dumps(aggregate.payload(include_variance=True))
            )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "example": number,
                        "total": len(examples),
                        "key": f"{example['task']}:{example['doc_id']}",
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated),
                        "finish_reason": finish_reason,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )

    for handle in handles:
        handle.remove()
    aggregate.save(worker_dir / "summary.npz")
    (worker_dir / "summary.json").write_text(json.dumps(aggregate.payload(include_variance=True)))
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
        "layer_types": list(text_config.layer_types),
        "logit_validation": logit_validation,
        "online_validation": recorder.validation,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": True,
            "active_experts": NATIVE_K,
        },
    }
    (worker_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged = ProfileStats.load(args.output / "workers" / "rank0" / "summary.npz")
        for other_rank in range(1, world_size):
            merged.merge(ProfileStats.load(args.output / "workers" / f"rank{other_rank}" / "summary.npz"))
        merged.save(args.output / "summary.npz")
        (args.output / "summary.json").write_text(json.dumps(merged.payload(include_variance=True)))
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "world_size": world_size,
                    "manifest_examples": len(all_examples),
                    "active_experts": NATIVE_K,
                    "layer_types": list(text_config.layer_types),
                    "fresh_full_trajectory_generation": True,
                    "shared_expert_profiled": True,
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
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument(
        "--hf-cache", type=Path, default=Path("/weka/oe-eval-default/oyvindt/hf-cache")
    )
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--validation-tokens", type=int, default=16)
    parser.add_argument("--max-logit-error", type=float, default=0.02)
    parser.add_argument("--max-weight-error", type=float, default=0.002)
    parser.add_argument("--max-reconstruction-error", type=float, default=0.02)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--distributed-timeout-hours", type=float, default=24.0)
    return parser.parse_args()


if __name__ == "__main__":
    profile(parse_args())

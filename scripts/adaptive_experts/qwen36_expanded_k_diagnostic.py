#!/usr/bin/env python3
"""Diagnose above-native-K routing in Qwen3.6 on matched token trajectories.

The model generates every continuation with its untouched native K=8 router.
We then replay the exact prompt+continuation tokens and, at every MoE layer,
compare four routed updates on the same incoming hidden state:

* native K=8;
* router-K=32 with only its first eight ranks retained (equivalence control);
* router-K=32 reference-scaled to the first-eight mass;
* router-K=32 conventionally normalized over all 32 selected experts.

The counterfactuals include the always-on shared expert and measure both router
probability mass and actual expert-output vectors.  This isolates the local MoE
effect; a short end-to-end teacher-forced probe additionally measures how the
policies compound through all 40 layers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import time
import types
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PHASES = ("prompt", "generated")
NATIVE_K = 8
EXPANDED_K = 32
METRICS = (
    "raw_top8_mass",
    "raw_top16_mass",
    "raw_top32_mass",
    "raw_rank9_16_mass",
    "raw_rank9_32_mass",
    "reference_k32_weight_sum",
    "normalized_k32_top8_share",
    "native_selected_weight_sum",
    "native_routed_norm",
    "k32_top8_routed_norm",
    "k32_tail_reference_norm",
    "reference_k32_routed_norm",
    "normalized_k32_routed_norm",
    "shared_output_norm",
    "native_moe_norm",
    "reference_k32_moe_norm",
    "normalized_k32_moe_norm",
    "tail_to_native_routed_norm_ratio",
    "tail_to_native_moe_norm_ratio",
    "shared_to_native_moe_norm_ratio",
    "reference_delta_relative_l2",
    "normalized_delta_relative_l2",
    "k32_top8_only_relative_l2",
    "native_recompute_control_relative_l2",
    "native_reconstruction_relative_l2",
    "reference_vs_native_cosine",
    "normalized_vs_native_cosine",
    "tail_vs_k32_top8_cosine",
    "tail_projection_on_native_moe",
    "tail_projection_on_native_routed",
    "native_k8_vs_k32_top8_id_mismatch",
)


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    return (left * right).sum(dim=-1) / denominator.clamp_min(1e-12)


class DiagnosticStats:
    def __init__(self, *, num_layers: int):
        shape = (len(PHASES), num_layers)
        self.count = np.zeros(shape, dtype=np.int64)
        self.metric_sum = np.zeros((*shape, len(METRICS)), dtype=np.float64)
        self.metric_sumsq = np.zeros_like(self.metric_sum)
        self.rank_probability_sum = np.zeros((*shape, EXPANDED_K), dtype=np.float64)
        self.rank_probability_sumsq = np.zeros_like(self.rank_probability_sum)

    def add(
        self,
        *,
        layer: int,
        phase_ids: torch.Tensor,
        metrics: torch.Tensor,
        rank_probabilities: torch.Tensor,
    ) -> None:
        for phase_id in range(len(PHASES)):
            selected = phase_ids == phase_id
            count = int(selected.sum().item())
            if not count:
                continue
            values = metrics[selected].double()
            ranks = rank_probabilities[selected].double()
            self.count[phase_id, layer] += count
            self.metric_sum[phase_id, layer] += values.sum(dim=0).cpu().numpy()
            self.metric_sumsq[phase_id, layer] += values.square().sum(dim=0).cpu().numpy()
            self.rank_probability_sum[phase_id, layer] += ranks.sum(dim=0).cpu().numpy()
            self.rank_probability_sumsq[phase_id, layer] += ranks.square().sum(dim=0).cpu().numpy()

    def merge(self, other: "DiagnosticStats") -> None:
        for name in (
            "count",
            "metric_sum",
            "metric_sumsq",
            "rank_probability_sum",
            "rank_probability_sumsq",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            count=self.count,
            metric_sum=self.metric_sum,
            metric_sumsq=self.metric_sumsq,
            rank_probability_sum=self.rank_probability_sum,
            rank_probability_sumsq=self.rank_probability_sumsq,
        )

    @classmethod
    def load(cls, path: Path) -> "DiagnosticStats":
        data = np.load(path)
        result = cls(num_layers=data["count"].shape[1])
        for name in (
            "count",
            "metric_sum",
            "metric_sumsq",
            "rank_probability_sum",
            "rank_probability_sumsq",
        ):
            getattr(result, name)[:] = data[name]
        return result

    def payload(self) -> dict[str, Any]:
        denominator = np.maximum(self.count, 1)
        metric_mean = self.metric_sum / denominator[..., None]
        rank_mean = self.rank_probability_sum / denominator[..., None]
        return {
            "phases": PHASES,
            "metrics": METRICS,
            "count": self.count.tolist(),
            "metric_means": metric_mean.tolist(),
            "metric_variances": np.maximum(
                self.metric_sumsq / denominator[..., None] - metric_mean**2, 0
            ).tolist(),
            "rank_probability_means": rank_mean.tolist(),
            "rank_probability_variances": np.maximum(
                self.rank_probability_sumsq / denominator[..., None] - rank_mean**2, 0
            ).tolist(),
        }


class Recorder:
    def __init__(self, *, stats: DiagnosticStats, num_layers: int):
        self.stats = stats
        self.num_layers = num_layers
        self.phase_ids: torch.Tensor | None = None
        self.pending: dict[int, dict[str, torch.Tensor]] = {}
        self.in_counterfactual = False
        self.completed: set[int] = set()
        self.validation = {
            "calls": 0,
            "max_native_reconstruction_relative_l2": 0.0,
            "max_k32_top8_only_relative_l2": 0.0,
            "max_native_recompute_control_relative_l2": 0.0,
            "native_k8_vs_k32_top8_id_mismatch_events": 0,
            "route_events": 0,
        }

    @property
    def active(self) -> bool:
        return self.phase_ids is not None

    def begin(self, *, prompt_tokens: int, total_tokens: int, device: torch.device) -> None:
        if self.active or self.pending or self.completed:
            raise RuntimeError("recorder has unflushed state")
        phase_ids = torch.ones(total_tokens, dtype=torch.long, device=device)
        phase_ids[:prompt_tokens] = 0
        self.phase_ids = phase_ids

    def capture(self, layer: int, name: str, value: torch.Tensor) -> None:
        if not self.active or self.in_counterfactual:
            return
        row = self.pending.setdefault(layer, {})
        if name in row:
            raise RuntimeError(f"duplicate {name} capture at layer {layer}")
        row[name] = value

    def complete(self, layer: int, module: torch.nn.Module, output: torch.Tensor) -> None:
        if not self.active:
            return
        row = self.pending.pop(layer)
        required = {"input", "router_logits", "native_weights", "native_ids", "native_routed", "shared_raw", "shared_gate"}
        if set(row) != required:
            raise RuntimeError(
                f"layer {layer} captures {sorted(row)}, expected {sorted(required)}"
            )
        flat = row["input"].reshape(-1, row["input"].shape[-1])
        logits = row["router_logits"].reshape(-1, row["router_logits"].shape[-1])
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_values, top_ids = torch.topk(probabilities, EXPANDED_K, dim=-1)
        sum8 = top_values[:, :NATIVE_K].sum(dim=-1, keepdim=True)
        sum16 = top_values[:, :16].sum(dim=-1)
        sum32 = top_values.sum(dim=-1, keepdim=True)
        reference_weights = top_values / sum8.clamp_min(1e-12)

        self.in_counterfactual = True
        try:
            native_recomputed = module.experts(
                flat,
                row["native_ids"].reshape(-1, NATIVE_K),
                row["native_weights"].reshape(-1, NATIVE_K),
            )
            k32_top8_routed = module.experts(
                flat,
                top_ids[:, :NATIVE_K],
                reference_weights[:, :NATIVE_K].to(flat.dtype),
            )
            k32_tail_reference = module.experts(
                flat,
                top_ids[:, NATIVE_K:],
                reference_weights[:, NATIVE_K:].to(flat.dtype),
            )
        finally:
            self.in_counterfactual = False

        native_routed = row["native_routed"].reshape_as(flat).float()
        native_recomputed = native_recomputed.float()
        shared_raw = row["shared_raw"].reshape_as(flat).float()
        shared_gate = torch.sigmoid(row["shared_gate"].reshape(-1, 1).float())
        shared = shared_raw * shared_gate
        native_total = output.reshape_as(flat).float()
        k32_top8_routed = k32_top8_routed.float()
        tail = k32_tail_reference.float()
        reference_routed = k32_top8_routed + tail
        normalized_routed = reference_routed * (sum8 / sum32.clamp_min(1e-12))
        keep8_total = shared + k32_top8_routed
        reference_total = shared + reference_routed
        normalized_total = shared + normalized_routed

        native_norm = torch.linalg.vector_norm(native_total, dim=-1)
        native_routed_norm = torch.linalg.vector_norm(native_routed, dim=-1)
        k32_top8_norm = torch.linalg.vector_norm(k32_top8_routed, dim=-1)
        tail_norm = torch.linalg.vector_norm(tail, dim=-1)
        reference_routed_norm = torch.linalg.vector_norm(reference_routed, dim=-1)
        normalized_routed_norm = torch.linalg.vector_norm(normalized_routed, dim=-1)
        shared_norm = torch.linalg.vector_norm(shared, dim=-1)
        reference_norm = torch.linalg.vector_norm(reference_total, dim=-1)
        normalized_norm = torch.linalg.vector_norm(normalized_total, dim=-1)
        native_routed_norm_squared = native_routed_norm.square().clamp_min(1e-12)
        native_norm_squared = native_norm.square().clamp_min(1e-12)

        native_ids = row["native_ids"].reshape(-1, NATIVE_K)
        native_sorted = torch.sort(native_ids, dim=-1).values
        k32_top8_sorted = torch.sort(top_ids[:, :NATIVE_K], dim=-1).values
        id_mismatch = (native_sorted != k32_top8_sorted).any(dim=-1).float()
        native_reconstruction = native_routed + shared
        native_reconstruction_relative = torch.linalg.vector_norm(
            native_reconstruction - native_total, dim=-1
        ) / native_norm.clamp_min(1e-12)
        keep8_relative = torch.linalg.vector_norm(
            keep8_total - native_total, dim=-1
        ) / native_norm.clamp_min(1e-12)
        native_recompute_relative = torch.linalg.vector_norm(
            native_recomputed - native_routed, dim=-1
        ) / native_routed_norm.clamp_min(1e-12)

        metrics = torch.stack(
            (
                sum8.squeeze(-1),
                sum16,
                sum32.squeeze(-1),
                top_values[:, NATIVE_K:16].sum(dim=-1),
                top_values[:, NATIVE_K:].sum(dim=-1),
                (sum32 / sum8.clamp_min(1e-12)).squeeze(-1),
                (sum8 / sum32.clamp_min(1e-12)).squeeze(-1),
                row["native_weights"].reshape(-1, NATIVE_K).float().sum(dim=-1),
                native_routed_norm,
                k32_top8_norm,
                tail_norm,
                reference_routed_norm,
                normalized_routed_norm,
                shared_norm,
                native_norm,
                reference_norm,
                normalized_norm,
                tail_norm / native_routed_norm.clamp_min(1e-12),
                tail_norm / native_norm.clamp_min(1e-12),
                shared_norm / native_norm.clamp_min(1e-12),
                torch.linalg.vector_norm(reference_total - native_total, dim=-1)
                / native_norm.clamp_min(1e-12),
                torch.linalg.vector_norm(normalized_total - native_total, dim=-1)
                / native_norm.clamp_min(1e-12),
                keep8_relative,
                native_recompute_relative,
                native_reconstruction_relative,
                safe_cosine(reference_total, native_total),
                safe_cosine(normalized_total, native_total),
                safe_cosine(tail, k32_top8_routed),
                (tail * native_total).sum(dim=-1) / native_norm_squared,
                (tail * native_routed).sum(dim=-1) / native_routed_norm_squared,
                id_mismatch,
            ),
            dim=-1,
        )
        assert self.phase_ids is not None
        self.stats.add(
            layer=layer,
            phase_ids=self.phase_ids,
            metrics=metrics,
            rank_probabilities=top_values,
        )
        self.validation["calls"] += 1
        self.validation["route_events"] += int(id_mismatch.numel())
        self.validation["native_k8_vs_k32_top8_id_mismatch_events"] += int(
            id_mismatch.sum().item()
        )
        self.validation["max_native_reconstruction_relative_l2"] = max(
            self.validation["max_native_reconstruction_relative_l2"],
            float(native_reconstruction_relative.max().item()),
        )
        self.validation["max_k32_top8_only_relative_l2"] = max(
            self.validation["max_k32_top8_only_relative_l2"],
            float(keep8_relative.max().item()),
        )
        self.validation["max_native_recompute_control_relative_l2"] = max(
            self.validation["max_native_recompute_control_relative_l2"],
            float(native_recompute_relative.max().item()),
        )
        self.completed.add(layer)

    def end(self) -> None:
        if not self.active:
            raise RuntimeError("recorder is inactive")
        if self.pending:
            raise RuntimeError(f"incomplete captures at layers {sorted(self.pending)}")
        expected = set(range(self.num_layers))
        if self.completed != expected:
            raise RuntimeError(f"missing completed layers {sorted(expected - self.completed)}")
        self.phase_ids = None
        self.completed.clear()


def install_hooks(
    model: torch.nn.Module, recorder: Recorder
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index, layer in enumerate(model.model.language_model.layers):
        moe = layer.mlp

        def moe_pre_hook(_module, inputs, *, index=layer_index):
            recorder.capture(index, "input", inputs[0])

        handles.append(moe.register_forward_pre_hook(moe_pre_hook))

        def gate_hook(_module, _inputs, output, *, index=layer_index):
            if recorder.active and not recorder.in_counterfactual:
                logits, weights, ids = output
                recorder.capture(index, "router_logits", logits)
                recorder.capture(index, "native_weights", weights)
                recorder.capture(index, "native_ids", ids)

        handles.append(moe.gate.register_forward_hook(gate_hook))

        def routed_hook(_module, _inputs, output, *, index=layer_index):
            recorder.capture(index, "native_routed", output)

        handles.append(moe.experts.register_forward_hook(routed_hook))

        def shared_hook(_module, _inputs, output, *, index=layer_index):
            recorder.capture(index, "shared_raw", output)

        handles.append(moe.shared_expert.register_forward_hook(shared_hook))

        def shared_gate_hook(_module, _inputs, output, *, index=layer_index):
            recorder.capture(index, "shared_gate", output)

        handles.append(moe.shared_expert_gate.register_forward_hook(shared_gate_hook))

        def moe_hook(module, _inputs, output, *, index=layer_index):
            recorder.complete(index, module, output)

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
    sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
    remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities > top_p
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(sorted_scores, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def policy_forward(self, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    probabilities = torch.softmax(router_logits, dtype=torch.float, dim=-1)
    values, indices = torch.topk(probabilities, EXPANDED_K, dim=-1)
    policy = self._expanded_k_diagnostic_policy
    if policy == "keep8":
        scores = values / values[:, :NATIVE_K].sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scores[:, NATIVE_K:] = 0
    elif policy == "reference32":
        scores = values / values[:, :NATIVE_K].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    elif policy == "normalized32":
        scores = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"unknown diagnostic policy {policy}")
    return router_logits, scores.to(router_logits.dtype), indices


def compare_logits(native: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    native = native.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    native_log_probs = torch.log_softmax(native, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate, dim=-1)
    native_probs = native_log_probs.exp()
    native_top10 = set(torch.topk(native, 10).indices.cpu().tolist())
    candidate_top10 = set(torch.topk(candidate, 10).indices.cpu().tolist())
    return {
        "max_abs_error": float((native - candidate).abs().max().item()),
        "mean_abs_error": float((native - candidate).abs().mean().item()),
        "cosine": float(F.cosine_similarity(native, candidate, dim=0).item()),
        "native_to_candidate_kl": float(
            (native_probs * (native_log_probs - candidate_log_probs)).sum().item()
        ),
        "top1_match": bool(native.argmax().item() == candidate.argmax().item()),
        "top10_overlap": len(native_top10 & candidate_top10),
        "native_top1_id": int(native.argmax().item()),
        "candidate_top1_id": int(candidate.argmax().item()),
    }


def end_to_end_probe(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    gates = [layer.mlp.gate for layer in model.model.language_model.layers]
    with torch.inference_mode():
        native = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        native_repeat = model(
            input_ids=input_ids, use_cache=False, logits_to_keep=1
        ).logits[:, -1, :]
    originals = [gate.forward for gate in gates]
    results = {"native_repeat": compare_logits(native, native_repeat)}
    try:
        for policy in ("keep8", "reference32", "normalized32"):
            for gate in gates:
                gate._expanded_k_diagnostic_policy = policy
                gate.forward = types.MethodType(policy_forward, gate)
            with torch.inference_mode():
                candidate = model(
                    input_ids=input_ids, use_cache=False, logits_to_keep=1
                ).logits[:, -1, :]
            results[policy] = compare_logits(native, candidate)
    finally:
        for gate, original in zip(gates, originals, strict=True):
            gate.forward = original
            if hasattr(gate, "_expanded_k_diagnostic_policy"):
                del gate._expanded_k_diagnostic_policy
    return results


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
    examples = all_examples[rank::world_size]
    if args.limit is not None:
        examples = examples[: args.limit]
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
    if int(text_config.num_experts_per_tok) != NATIVE_K:
        raise ValueError(f"expected native K={NATIVE_K}, got {text_config.num_experts_per_tok}")
    if int(text_config.num_experts) < EXPANDED_K:
        raise ValueError(f"model has only {text_config.num_experts} experts")
    num_layers = int(text_config.num_hidden_layers)

    rendered_probe = tokenizer.apply_chat_template(
        all_examples[0]["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    probe_ids = tokenizer(
        rendered_probe, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, -args.validation_tokens :].to(device)
    probe = end_to_end_probe(model, probe_ids)
    if not probe["native_repeat"]["top1_match"]:
        raise RuntimeError(f"native model is nondeterministic on probe: {probe}")

    stats = DiagnosticStats(num_layers=num_layers)
    recorder = Recorder(stats=stats, num_layers=num_layers)
    handles = install_hooks(model, recorder)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    output_path = worker_dir / "examples.jsonl.gz"
    with gzip.open(output_path, "wt") as output_file, torch.inference_mode():
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
            ).input_ids[:, -args.max_prompt_tokens :].to(device)
            prompt_tokens = int(prompt_ids.shape[-1])
            outputs = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            generator = torch.Generator(device=device).manual_seed(int(example["seed"]))
            generated: list[int] = []
            finish_reason = "length"
            for _ in range(args.max_new_tokens):
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
                outputs = model(
                    input_ids=token, past_key_values=past, use_cache=True, logits_to_keep=1
                )
                past = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]

            replay_ids = prompt_ids
            if generated:
                replay_ids = torch.cat(
                    (prompt_ids, torch.tensor([generated], dtype=torch.long, device=device)),
                    dim=-1,
                )
            recorder.begin(
                prompt_tokens=prompt_tokens,
                total_tokens=int(replay_ids.shape[-1]),
                device=device,
            )
            model(input_ids=replay_ids, use_cache=False, logits_to_keep=1)
            recorder.end()
            stats.save(worker_dir / "summary.partial.npz")
            (worker_dir / "summary.partial.json").write_text(json.dumps(stats.payload()))
            output_file.write(
                json.dumps(
                    {
                        "task": example.get("task"),
                        "doc_id": example.get("doc_id"),
                        "native_id": example.get("native_id"),
                        "baseline_score": example.get("baseline_score"),
                        "seed": example.get("seed"),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated),
                        "finish_reason": finish_reason,
                        "generated_token_ids": generated,
                        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "example": example_number,
                        "total": len(examples),
                        "key": f"{example.get('task')}:{example.get('doc_id')}",
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )

    for handle in handles:
        handle.remove()
    stats.save(worker_dir / "summary.npz")
    (worker_dir / "summary.json").write_text(json.dumps(stats.payload()))
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
        "end_to_end_probe": probe,
        "validation": recorder.validation,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "enable_thinking": True,
        },
    }
    (worker_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged = DiagnosticStats.load(args.output / "workers" / "rank0" / "summary.npz")
        for other_rank in range(1, world_size):
            merged.merge(
                DiagnosticStats.load(
                    args.output / "workers" / f"rank{other_rank}" / "summary.npz"
                )
            )
        merged.save(args.output / "summary.npz")
        (args.output / "summary.json").write_text(json.dumps(merged.payload()))
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "world_size": world_size,
                    "manifest_examples": len(all_examples),
                    "native_k": NATIVE_K,
                    "expanded_k": EXPANDED_K,
                    "reference_k": NATIVE_K,
                    "matched_native_token_trajectories": True,
                    "layer_types": list(text_config.layer_types),
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
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--hf-cache", type=Path, default=Path("/weka/oe-eval-default/oyvindt/hf-cache")
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--validation-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--distributed-timeout-hours", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    profile(parse_args())

#!/usr/bin/env python3
"""Profile Qwen3 MoE router distributions during incremental Transformers decoding."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import os
import platform
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

PHASES = ("prompt", "reasoning", "final")
RANKS = (1, 2, 4, 8, 16)
METRICS = tuple(
    [f"c{k}" for k in RANKS]
    + [f"renorm{k}" for k in RANKS]
    + ["entropy", "effective_experts"]
    + ["prob_margin_1_2", "prob_margin_4_5", "prob_margin_8_9"]
    + ["logit_margin_1_2", "logit_margin_4_5", "logit_margin_8_9"]
)


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one token using temperature, top-k, then nucleus filtering."""
    scores = logits.float() / temperature
    if 0 < top_k < scores.shape[-1]:
        cutoff = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    probs = torch.softmax(sorted_scores, dim=-1)
    sampled_sorted = torch.multinomial(probs, num_samples=1, generator=generator)
    return sorted_indices.gather(-1, sampled_sorted)


def metrics_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scalar metrics, top-16 IDs, and top-16 probabilities for each token."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, max(RANKS), dim=-1)
    top_logits = logits.gather(-1, top_ids)
    cums = top_probs.cumsum(dim=-1)
    values = [cums[..., k - 1] for k in RANKS]
    values.extend(1.0 / cums[..., k - 1].clamp_min(1e-12) for k in RANKS)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)
    values.extend((entropy, entropy.exp()))
    for i, j in ((0, 1), (3, 4), (7, 8)):
        values.append(top_probs[..., i] - top_probs[..., j])
    for i, j in ((0, 1), (3, 4), (7, 8)):
        values.append(top_logits[..., i] - top_logits[..., j])
    return torch.stack(values, dim=-1), top_ids, top_probs


class RouterStats:
    """Online sufficient statistics; only a small priority sample keeps token-level rows."""

    def __init__(self, *, num_layers: int, num_experts: int, sample_size: int, seed: int):
        shape = (len(PHASES), num_layers)
        self.count = np.zeros(shape, dtype=np.int64)
        self.sums = np.zeros((*shape, len(METRICS)), dtype=np.float64)
        self.sumsq = np.zeros((*shape, len(METRICS)), dtype=np.float64)
        self.top1_count = np.zeros((*shape, num_experts), dtype=np.int64)
        self.top8_count = np.zeros((*shape, num_experts), dtype=np.int64)
        self.churn_sum = np.zeros(shape, dtype=np.float64)
        self.churn_count = np.zeros(shape, dtype=np.int64)
        self._previous: dict[tuple[int, int], np.ndarray] = {}
        self._sample_size = sample_size
        self._rng = random.Random(seed)
        self._sample_heap: list[tuple[float, int, dict[str, Any]]] = []
        self._sample_serial = 0

    def update(
        self,
        router_logits: tuple[torch.Tensor, ...],
        *,
        phase: str,
        token_offset: int,
        example_key: str,
    ) -> None:
        phase_id = PHASES.index(phase)
        # One device-to-host synchronization per update, rather than one per layer.
        logits = torch.stack(
            [layer_logits.reshape(-1, layer_logits.shape[-1]) for layer_logits in router_logits]
        )
        values, top_ids, top_probs = metrics_from_logits(logits)
        values_np_all = values.detach().cpu().numpy().astype(np.float64, copy=False)
        ids_np_all = top_ids.detach().cpu().numpy()
        probs_np_all = top_probs.detach().cpu().numpy()
        for layer, (values_np, ids_np, probs_np) in enumerate(
            zip(values_np_all, ids_np_all, probs_np_all, strict=True)
        ):
            n_tokens = values_np.shape[0]
            self.count[phase_id, layer] += n_tokens
            self.sums[phase_id, layer] += values_np.sum(axis=0)
            self.sumsq[phase_id, layer] += np.square(values_np).sum(axis=0)
            self.top1_count[phase_id, layer] += np.bincount(
                ids_np[:, 0], minlength=self.top1_count.shape[-1]
            )
            self.top8_count[phase_id, layer] += np.bincount(
                ids_np[:, :8].reshape(-1), minlength=self.top8_count.shape[-1]
            )

            key = (phase_id, layer)
            previous = self._previous.get(key)
            route_rows = ids_np[:, :8]
            if previous is not None:
                route_rows = np.concatenate((previous[None, :], route_rows), axis=0)
            if len(route_rows) > 1:
                intersections = np.array(
                    [
                        len(set(a.tolist()) & set(b.tolist()))
                        for a, b in zip(route_rows, route_rows[1:], strict=False)
                    ]
                )
                self.churn_sum[phase_id, layer] += np.sum(
                    1.0 - intersections / (16 - intersections)
                )
                self.churn_count[phase_id, layer] += len(intersections)
            self._previous[key] = ids_np[-1, :8].copy()

            if self._sample_size == 0:
                continue
            for token_index in range(n_tokens):
                priority = self._rng.random()
                if (
                    len(self._sample_heap) >= self._sample_size
                    and priority <= self._sample_heap[0][0]
                ):
                    continue
                record = {
                    "example_key": example_key,
                    "phase": phase,
                    "token_position": token_offset + token_index,
                    "layer": layer,
                    "top16_ids": ids_np[token_index].tolist(),
                    "top16_probs": probs_np[token_index].tolist(),
                    "metrics": dict(zip(METRICS, values_np[token_index].tolist(), strict=True)),
                }
                item = (priority, self._sample_serial, record)
                self._sample_serial += 1
                if len(self._sample_heap) < self._sample_size:
                    heapq.heappush(self._sample_heap, item)
                else:
                    heapq.heapreplace(self._sample_heap, item)

    def merge(self, other: RouterStats) -> None:
        for name in (
            "count",
            "sums",
            "sumsq",
            "top1_count",
            "top8_count",
            "churn_sum",
            "churn_count",
        ):
            getattr(self, name)[...] += getattr(other, name)
        for item in other._sample_heap:
            if len(self._sample_heap) < self._sample_size:
                heapq.heappush(self._sample_heap, item)
            elif item[0] > self._sample_heap[0][0]:
                heapq.heapreplace(self._sample_heap, item)

    @property
    def samples(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self._sample_heap, reverse=True)]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            count=self.count,
            sums=self.sums,
            sumsq=self.sumsq,
            top1_count=self.top1_count,
            top8_count=self.top8_count,
            churn_sum=self.churn_sum,
            churn_count=self.churn_count,
            phases=np.array(PHASES),
            metrics=np.array(METRICS),
        )

    @classmethod
    def load(cls, path: Path, *, sample_size: int = 0) -> RouterStats:
        data = np.load(path)
        obj = cls(
            num_layers=data["count"].shape[1],
            num_experts=data["top1_count"].shape[-1],
            sample_size=sample_size,
            seed=0,
        )
        for name in (
            "count",
            "sums",
            "sumsq",
            "top1_count",
            "top8_count",
            "churn_sum",
            "churn_count",
        ):
            setattr(obj, name, data[name])
        return obj


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def make_route_validator(model: torch.nn.Module) -> tuple[dict[str, int], list[Any]]:
    num_layers = int(model.config.num_hidden_layers)
    validation = {"router_calls": 0, "events": 0, "mismatches": 0}
    handles = []

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        # The first full prefill validates every layer. Avoid an extra top-k and
        # device synchronization on every subsequent decode token.
        if validation["router_calls"] >= num_layers:
            return
        logits, _scores, routed_ids = output
        reconstructed = torch.topk(logits.float(), routed_ids.shape[-1], dim=-1).indices
        reconstructed = torch.sort(reconstructed, dim=-1).values
        routed_ids = torch.sort(routed_ids, dim=-1).values
        equal = torch.equal(reconstructed, routed_ids)
        validation["router_calls"] += 1
        validation["events"] += routed_ids.numel()
        if not equal:
            validation["mismatches"] += int(
                (~torch.eq(reconstructed, routed_ids)).any(dim=-1).sum().item()
            )

    for module in model.modules():
        if module.__class__.__name__ == "Qwen3MoeTopKRouter":
            handles.append(module.register_forward_hook(hook))
    return validation, handles


def write_summary_json(stats: RouterStats, path: Path) -> None:
    denominator = np.maximum(stats.count[..., None], 1)
    means = stats.sums / denominator
    variances = np.maximum(stats.sumsq / denominator - means**2, 0)
    payload = {
        "phases": PHASES,
        "metrics": METRICS,
        "count": stats.count.tolist(),
        "mean": means.tolist(),
        "variance": variances.tolist(),
        "route_churn_mean": (stats.churn_sum / np.maximum(stats.churn_count, 1)).tolist(),
    }
    path.write_text(json.dumps(payload))


def profile(args: argparse.Namespace) -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # Ranks process independent prompts and may differ by hours when one
        # generation reaches the context limit. The default 30-minute process
        # group timeout killed the first full run while faster ranks waited at
        # the final merge barrier.
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
    num_experts = int(model.config.num_experts)
    num_layers = int(model.config.num_hidden_layers)
    if int(model.config.num_experts_per_tok) != 8:
        raise ValueError(f"expected checkpoint-default K=8, got {model.config.num_experts_per_tok}")
    validation, hook_handles = make_route_validator(model)
    aggregate = RouterStats(
        num_layers=num_layers,
        num_experts=num_experts,
        sample_size=args.sample_size,
        seed=args.seed + rank,
    )
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    context_limit = int(getattr(model.config, "max_position_embeddings", 40960))
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])

    examples_path = output_dir / "examples.jsonl.gz"
    with gzip.open(examples_path, "wt") as examples_file, torch.inference_mode():
        for example_index, example in enumerate(examples):
            example_started = time.monotonic()
            rendered = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_ids = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            prompt_tokens = input_ids.shape[-1]
            max_new_tokens = min(args.max_new_tokens, context_limit - prompt_tokens)
            if max_new_tokens <= 0:
                raise ValueError(f"prompt exceeds context: {example['task']}:{example['doc_id']}")
            example_key = f"{example['task']}:{example['doc_id']}"
            per_example = RouterStats(
                num_layers=num_layers,
                num_experts=num_experts,
                sample_size=args.sample_size,
                seed=example["seed"],
            )
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_router_logits=True,
                logits_to_keep=1,
            )
            router_logits = tuple(outputs.router_logits)
            per_example.update(
                router_logits, phase="prompt", token_offset=0, example_key=example_key
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            generator = torch.Generator(device=device).manual_seed(example["seed"])
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
                cache_position = torch.tensor(
                    [past.get_seq_length()], dtype=torch.long, device=device
                )
                outputs = model(
                    input_ids=token,
                    past_key_values=past,
                    cache_position=cache_position,
                    use_cache=True,
                    output_router_logits=True,
                    logits_to_keep=1,
                )
                router_logits = tuple(outputs.router_logits)
                per_example.update(
                    router_logits,
                    phase=phase,
                    token_offset=generation_index,
                    example_key=example_key,
                )
                past = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
                if end_think_ids and generated[-len(end_think_ids) :] == end_think_ids:
                    phase = "final"

            aggregate.merge(per_example)
            denominator = np.maximum(per_example.count[..., None], 1)
            identity_keys = ("task", "doc_id", "native_id", "baseline_score", "seed")
            example_output = {
                **{key: example[key] for key in identity_keys},
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated),
                "finish_reason": finish_reason,
                "elapsed_seconds": time.monotonic() - example_started,
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
                "phase_counts": per_example.count.tolist(),
                "metric_means": (per_example.sums / denominator).tolist(),
            }
            examples_file.write(json.dumps(example_output, ensure_ascii=False) + "\n")
            examples_file.flush()
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "example": example_index + 1,
                        "total": len(examples),
                        "key": example_key,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(generated),
                        "finish_reason": finish_reason,
                        "peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )

    for handle in hook_handles:
        handle.remove()
    aggregate.save(output_dir / "summary.npz")
    write_summary_json(aggregate, output_dir / "summary.json")
    with (output_dir / "samples.jsonl").open("w") as f:
        for row in aggregate.samples:
            f.write(json.dumps(row) + "\n")
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
        "router_validation": validation,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": True,
            "active_experts": 8,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged = RouterStats.load(args.output / "workers" / "rank0" / "summary.npz")
        for other_rank in range(1, world_size):
            merged.merge(
                RouterStats.load(args.output / "workers" / f"rank{other_rank}" / "summary.npz")
            )
        merged.save(args.output / "summary.npz")
        write_summary_json(merged, args.output / "summary.json")
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "model": args.model,
                    "world_size": world_size,
                    "manifest_examples": len(all_examples),
                    "completed_examples": sum(
                        len(read_manifest(args.manifest)[r::world_size]) for r in range(world_size)
                    ),
                    "active_experts": 8,
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
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--distributed-timeout-hours", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    profile(parse_args())

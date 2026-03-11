"""No-GPU oracle-vs-corrupted task pipeline tests.

These tests validate task-level formatting + scoring behavior without real model
inference by constructing synthetic responses for real task requests.
"""

from __future__ import annotations

import os
import random
from itertools import islice

import pytest

from olmo_eval.common.types import LMOutput, RequestType, Response
from olmo_eval.evals.tasks.common import get_task

RUN_REAL_DATASET_TESTS = os.getenv("RUN_REAL_DATASET_TESTS") == "1"


def _build_char_logprobs(
    *,
    text: str,
    rng: random.Random,
    corrupted: bool,
) -> list[dict[str, float | str]]:
    """Build deterministic token logprobs from text, with optional corruption."""
    tokens = list(text) if text else [""]
    n_tokens = len(tokens)

    corrupted_indices: set[int] = set()
    if corrupted:
        count = max(1, n_tokens // 5)
        corrupted_indices = set(rng.sample(range(n_tokens), count))

    logprobs: list[dict[str, float | str]] = []
    for idx, token in enumerate(tokens):
        logprob = -3.0 if idx in corrupted_indices else -0.05
        logprobs.append({"token": token, "logprob": logprob})

    return logprobs


def _quality_from_metrics(
    *,
    metrics: dict[str, dict[str, float]],
    mode: str,
) -> float:
    """Normalize metrics to a higher-is-better quality scalar."""
    if mode == "bpb":
        bpb = metrics["bits_per_byte"]["bits_per_byte"]
        return -bpb
    if mode == "labbench_accuracy":
        return metrics["accuracy"]["multiple_choice"]
    if mode == "minerva_accuracy":
        return metrics["accuracy"]["minerva_math_flex"]
    raise ValueError(f"Unknown quality mode: {mode}")


@pytest.mark.anyio
@pytest.mark.skipif(
    not RUN_REAL_DATASET_TESTS,
    reason="Set RUN_REAL_DATASET_TESTS=1 to run real-dataset task tests.",
)
async def test_humaneval_bpb_perfect_vs_corrupted_quality() -> None:
    task = get_task(spec="humaneval:bpb", config_overrides={"num_fewshot": 0})

    loaded_instances = list(islice(task.instances, 3))
    assert loaded_instances

    requests = [task.format_request(instance=instance) for instance in loaded_instances]

    for request in requests:
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations is not None
        assert len(request.continuations) == 1

    perfect_rng = random.Random(13)
    corrupted_rng = random.Random(13)

    perfect_responses: list[Response] = []
    corrupted_responses: list[Response] = []

    for instance, request in zip(loaded_instances, requests, strict=True):
        continuation = request.continuations[0]

        perfect_logprobs = _build_char_logprobs(
            text=continuation,
            rng=perfect_rng,
            corrupted=False,
        )
        corrupted_logprobs = _build_char_logprobs(
            text=continuation,
            rng=corrupted_rng,
            corrupted=True,
        )

        perfect_output = LMOutput(
            text=continuation,
            logprobs=perfect_logprobs,
            metadata={"total_logprob": float(sum(entry["logprob"] for entry in perfect_logprobs))},
        )
        corrupted_output = LMOutput(
            text=continuation,
            logprobs=corrupted_logprobs,
            metadata={
                "total_logprob": float(sum(entry["logprob"] for entry in corrupted_logprobs))
            },
        )

        perfect_responses.append(
            Response(instance=instance, request=request, outputs=[perfect_output])
        )
        corrupted_responses.append(
            Response(instance=instance, request=request, outputs=[corrupted_output])
        )

    perfect_scored = await task.score_responses(responses=perfect_responses)
    corrupted_scored = await task.score_responses(responses=corrupted_responses)

    perfect_metrics = task.compute_metrics(responses=perfect_scored)
    corrupted_metrics = task.compute_metrics(responses=corrupted_scored)

    perfect_quality = _quality_from_metrics(metrics=perfect_metrics, mode="bpb")
    corrupted_quality = _quality_from_metrics(metrics=corrupted_metrics, mode="bpb")

    assert corrupted_quality < perfect_quality


@pytest.mark.anyio
@pytest.mark.skipif(
    not RUN_REAL_DATASET_TESTS,
    reason="Set RUN_REAL_DATASET_TESTS=1 to run real-dataset task tests.",
)
async def test_labbench_mc_perfect_vs_corrupted_quality() -> None:
    task = get_task(spec="lab_bench_litqa2:mc", config_overrides={"num_fewshot": 0})

    loaded_instances = list(islice(task.instances, 3))
    assert loaded_instances

    requests = [task.format_request(instance=instance) for instance in loaded_instances]

    for request, instance in zip(requests, loaded_instances, strict=True):
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations is not None
        assert instance.choices is not None
        assert len(request.continuations) == len(instance.choices)

    rng = random.Random(29)

    perfect_responses: list[Response] = []
    corrupted_responses: list[Response] = []

    for instance, request in zip(loaded_instances, requests, strict=True):
        assert request.continuations is not None
        gold_idx = int(instance.metadata["gold_idx"])

        wrong_candidates = [idx for idx in range(len(request.continuations)) if idx != gold_idx]
        wrong_idx = rng.choice(wrong_candidates)

        perfect_outputs: list[LMOutput] = []
        corrupted_outputs: list[LMOutput] = []

        for idx, continuation in enumerate(request.continuations):
            perfect_total = -0.1 if idx == gold_idx else -10.0
            corrupted_total = -0.1 if idx == wrong_idx else -10.0

            perfect_outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=[{"token": continuation, "logprob": perfect_total}],
                    metadata={"total_logprob": perfect_total},
                )
            )
            corrupted_outputs.append(
                LMOutput(
                    text=continuation,
                    logprobs=[{"token": continuation, "logprob": corrupted_total}],
                    metadata={"total_logprob": corrupted_total},
                )
            )

        perfect_responses.append(
            Response(instance=instance, request=request, outputs=perfect_outputs)
        )
        corrupted_responses.append(
            Response(instance=instance, request=request, outputs=corrupted_outputs)
        )

    perfect_scored = await task.score_responses(responses=perfect_responses)
    corrupted_scored = await task.score_responses(responses=corrupted_responses)

    perfect_metrics = task.compute_metrics(responses=perfect_scored)
    corrupted_metrics = task.compute_metrics(responses=corrupted_scored)

    perfect_quality = _quality_from_metrics(metrics=perfect_metrics, mode="labbench_accuracy")
    corrupted_quality = _quality_from_metrics(metrics=corrupted_metrics, mode="labbench_accuracy")

    assert corrupted_quality < perfect_quality


@pytest.mark.anyio
@pytest.mark.skipif(
    not RUN_REAL_DATASET_TESTS,
    reason="Set RUN_REAL_DATASET_TESTS=1 to run real-dataset task tests.",
)
async def test_minerva_end_metric_perfect_vs_corrupted_quality() -> None:
    task = get_task(spec="minerva_math_algebra", config_overrides={"num_fewshot": 0})

    loaded_instances = list(islice(task.instances, 3))
    assert loaded_instances

    requests = [task.format_request(instance=instance) for instance in loaded_instances]

    for request in requests:
        assert request.request_type == RequestType.COMPLETION
        assert request.continuations is None

    rng = random.Random(41)

    perfect_responses: list[Response] = []
    corrupted_responses: list[Response] = []

    for instance, request in zip(loaded_instances, requests, strict=True):
        solution_text = str(instance.metadata.get("solution_text", ""))
        if solution_text:
            perfect_text = solution_text
        else:
            perfect_text = (
                "Final Answer: The final answer is "
                f"$\\boxed{{{instance.gold_answer}}}$. I hope it is correct."
            )

        # Keep format parseable but force likely-wrong extracted answer.
        wrong_value = str(3141592653589793 + rng.randint(1, 99))
        corrupted_text = (
            f"Final Answer: The final answer is $\\boxed{{{wrong_value}}}$. I hope it is correct."
        )

        perfect_output = LMOutput(text=perfect_text)
        corrupted_output = LMOutput(text=corrupted_text)

        perfect_responses.append(
            Response(instance=instance, request=request, outputs=[perfect_output])
        )
        corrupted_responses.append(
            Response(instance=instance, request=request, outputs=[corrupted_output])
        )

    perfect_scored = await task.score_responses(responses=perfect_responses)
    corrupted_scored = await task.score_responses(responses=corrupted_responses)

    perfect_metrics = task.compute_metrics(responses=perfect_scored)
    corrupted_metrics = task.compute_metrics(responses=corrupted_scored)

    perfect_quality = _quality_from_metrics(metrics=perfect_metrics, mode="minerva_accuracy")
    corrupted_quality = _quality_from_metrics(metrics=corrupted_metrics, mode="minerva_accuracy")

    assert corrupted_quality < perfect_quality

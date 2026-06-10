"""Parity tests against the released mm_olmo Molmo2-4B prediction dumps.

For every image-QA benchmark this re-scores the predictions saved by the
*original* mm_olmo evaluation (``predictions-ck2000-*``) with the new
task/scorer/metric stack and asserts:

1. **Prompt parity** — the user-turn text of each saved prompt equals the
   ``instance.question`` produced by the new task (style prefixes, MC
   formatting, and the PixMo-Count RNG templates must all match exactly).
2. **Metric parity** — the recomputed metrics equal the reference
   ``metrics.json`` values within a small tolerance.

The dumps are reference ground truth and are opened **read-only**; nothing
in this test writes to them.

Opt-in:

    RUN_DUMP_PARITY_TESTS=1 \
    HF_DATASETS_CACHE=/weka/oe-training-default/mm-olmo/hf_datasets \
    HF_DATASETS_OFFLINE=1 \
    pytest tests/evals/tasks/test_image_qa_dump_parity.py -v

``MOLMO2_PREDICTIONS_ROOT`` overrides the dump location (default: the
released Molmo2-4B directory).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from olmo_eval.common.types import Instance, LMOutput, Response

if not os.environ.get("RUN_DUMP_PARITY_TESTS"):
    pytest.skip(
        "Set RUN_DUMP_PARITY_TESTS=1 (and HF_DATASETS_CACHE for the HF-hub tasks) "
        "to run dump-parity tests",
        allow_module_level=True,
    )

from olmo_eval.evals.tasks.common.registry import get_task  # noqa: E402

DEFAULT_PREDICTIONS_ROOT = "/weka/oe-training-default/mm-olmo/released-models-molmo2-1225/Molmo2-4B"

# Per-task plumbing: (task spec, dump dir name, join key fn, metric tolerance)
TOLERANCE_DEFAULT = 2e-4
TOLERANCE_MMMU = 2e-3


def _root() -> Path:
    return Path(os.environ.get("MOLMO2_PREDICTIONS_ROOT", DEFAULT_PREDICTIONS_ROOT))


def _load_dump(dump_name: str) -> tuple[list[dict], dict[str, float]]:
    dump_dir = _root() / f"predictions-ck2000-{dump_name}"
    if not dump_dir.exists():
        pytest.skip(f"reference dump not found: {dump_dir}")
    with open(dump_dir / "predictions.json") as f:
        rows = json.load(f)
    with open(dump_dir / "metrics.json") as f:
        metrics = json.load(f)["metrics"]
    return rows, {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


def _user_text(prompt: str) -> str:
    """Extract the user-turn text from a decoded native prompt."""
    text = prompt.split("<|im_start|>user\n", 1)[1]
    return text.split("<|im_end|>", 1)[0]


def _score_against_dump(task, joined: list[tuple[Instance, str]]) -> dict[str, float]:
    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=prediction)],
        )
        for instance, prediction in joined
    ]
    responses = asyncio.run(task.score_responses(responses))
    nested = task.compute_metrics(responses)
    return {name: next(iter(by_scorer.values())) for name, by_scorer in nested.items()}


def _assert_metrics(mine: dict[str, float], ref: dict[str, float], tol: float) -> None:
    compared = 0
    for name, value in mine.items():
        if name not in ref:
            continue
        assert value == pytest.approx(ref[name], abs=tol), (
            f"{name}: recomputed {value:.6f} != reference {ref[name]:.6f}"
        )
        compared += 1
    assert compared > 0, "no overlapping metric names with the reference"


# ---------------------------------------------------------------------------
# Simple joined tasks: example_id-keyed, full prompt parity
# ---------------------------------------------------------------------------


def _join_by(instances, rows, instance_key, row_key):
    by_key = {instance_key(inst): inst for inst in instances}
    assert len(by_key) == len(instances), "join keys are not unique"
    joined = []
    for row in rows:
        key = row_key(row)
        assert key in by_key, f"dump row {key!r} has no matching instance"
        joined.append((by_key[key], row))
    assert len(joined) == len(rows)
    return joined


@pytest.mark.parametrize(
    ("spec", "dump_name", "tol"),
    [
        ("chart_qa", "chart_qa-validation", TOLERANCE_DEFAULT),
        ("vqa2", "coco_2014_vqa_8192-validation", TOLERANCE_DEFAULT),
        ("doc_qa", "doc_qa-validation", TOLERANCE_DEFAULT),
        ("info_qa", "info_qa-validation", TOLERANCE_DEFAULT),
        ("text_vqa", "text_vqa-validation", TOLERANCE_DEFAULT),
        ("mmmu", "mmmu_test-validation", TOLERANCE_MMMU),
        ("ai2d", "ai2_diagram_v2_mix_transparent-validation", TOLERANCE_DEFAULT),
        ("countbench_qa", "countbench_qa-huggingface", TOLERANCE_DEFAULT),
        ("pixmo_count", "pixmo_count_counting-validation", TOLERANCE_DEFAULT),
    ],
)
def test_dump_parity(spec: str, dump_name: str, tol: float) -> None:
    rows, ref = _load_dump(dump_name)
    task = get_task(spec)
    instances = list(task.instances)
    assert len(instances) == len(rows)

    if spec == "chart_qa":
        joined = _join_by(
            instances,
            rows,
            lambda inst: (inst.metadata["example_id"], inst.metadata["is_human"]),
            lambda row: (row["example_id"], row["is_human"]),
        )
    elif spec == "pixmo_count":
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["image_url"],
            lambda row: row["image_url"],
        )
    elif spec == "countbench_qa":
        # image_url is not unique in CountBench; the dump saves the integer
        # example_id under "image_id".
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["example_id"],
            lambda row: row["image_id"],
        )
    else:
        joined = _join_by(
            instances,
            rows,
            lambda inst: inst.metadata["example_id"],
            lambda row: row["example_id"],
        )

    # 1. Prompt parity
    mismatches = [
        (instance.metadata.get("example_id"), _user_text(row["prompt"]), instance.question)
        for instance, row in joined
        if _user_text(row["prompt"]) != instance.question
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(joined)} prompt mismatches; first: {mismatches[0]}"
    )

    # 2. Metric parity
    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    _assert_metrics(mine, ref, tol)


# ---------------------------------------------------------------------------
# Unlabeled test-split variants (eval-server submissions): answers are not
# public, so only prompt parity is asserted against the native test dumps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "dump_name"),
    [
        ("doc_qa:test", "doc_qa-test-base_native_test"),
        ("info_qa:test", "info_qa-test-base_native_test"),
    ],
)
def test_dump_prompt_parity_unlabeled_test_split(spec: str, dump_name: str) -> None:
    rows, _ = _load_dump(dump_name)
    task = get_task(spec)
    instances = list(task.instances)
    assert len(instances) == len(rows)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )
    mismatches = [
        (instance.metadata["example_id"], _user_text(row["prompt"]), instance.question)
        for instance, row in joined
        if _user_text(row["prompt"]) != instance.question
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(joined)} prompt mismatches; first: {mismatches[0]}"
    )


# ---------------------------------------------------------------------------
# RealWorldQA: the dump's `prompt` field is the *original* HF question (it is
# overwritten by metadata["prompt"] in SavePredictions), so prompt parity is
# checked against the documented derivation instead of the decoded input.
# ---------------------------------------------------------------------------


def test_dump_parity_real_world_qa() -> None:
    rows, ref = _load_dump("real_world_qa_no_instruction-test")
    task = get_task("real_world_qa")
    instances = list(task.instances)
    assert len(instances) == len(rows)

    # RealWorldQA has duplicate question texts, so join as a multiset keyed by
    # (question, answer, question_type) — duplicates beyond that are
    # interchangeable for scoring purposes.
    pools: dict[tuple, list[Instance]] = {}
    for inst in instances:
        key = (
            inst.metadata["original_question"],
            inst.metadata["answer"],
            inst.metadata["question_type"],
        )
        pools.setdefault(key, []).append(inst)
    joined = []
    for row in rows:
        key = (row["prompt"], row["answer"], row["question_type"])
        assert pools.get(key), f"dump row has no matching instance: {key[0][:80]!r}"
        joined.append((pools[key].pop(), row))
    assert len(joined) == len(rows)

    for instance, row in joined:
        original = row["prompt"]
        if row["question_type"] == "short_answer":
            expected = f"vqa2: {original.split(chr(10))[0]}"
        else:
            expected = original
        assert instance.question == expected, instance.metadata["example_id"]

    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    _assert_metrics(mine, ref, TOLERANCE_DEFAULT)


# ---------------------------------------------------------------------------
# MathVista: prompt parity is exact; the reference `score` (0.5670) used GPT-4
# answer extraction, so the offline score is only asserted as a sanity band.
# The `math_vista:gpt` variant can be asserted against the reference with
# RUN_MATHVISTA_GPT_PARITY=1 + OPENAI_API_KEY (fresh API calls, own cache).
# ---------------------------------------------------------------------------


def test_dump_parity_math_vista_offline() -> None:
    rows, ref = _load_dump("math_vista_v2-validation")
    task = get_task("math_vista")
    instances = list(task.instances)
    assert len(instances) == len(rows)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )

    mismatches = [
        (instance.metadata["example_id"], _user_text(row["prompt"]), instance.question)
        for instance, row in joined
        if _user_text(row["prompt"]) != instance.question
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(joined)} prompt mismatches; first: {mismatches[0]}"
    )

    mine = _score_against_dump(task, [(inst, row["prediction"]) for inst, row in joined])
    # Offline extraction is not the GPT protocol that produced ref["score"];
    # assert a sanity band and report the delta.
    assert mine["score"] >= 0.50, f"offline MathVista score suspiciously low: {mine['score']}"
    print(f"math_vista offline={mine['score']:.4f} vs GPT reference={ref['score']:.4f}")


@pytest.mark.skipif(
    not os.environ.get("RUN_MATHVISTA_GPT_PARITY"),
    reason="Set RUN_MATHVISTA_GPT_PARITY=1 + OPENAI_API_KEY for GPT parity (~1000 API calls)",
)
def test_dump_parity_math_vista_gpt() -> None:
    rows, ref = _load_dump("math_vista_v2-validation")
    task = get_task("math_vista:gpt")
    instances = list(task.instances)

    joined = _join_by(
        instances,
        rows,
        lambda inst: inst.metadata["example_id"],
        lambda row: row["example_id"],
    )

    from olmo_eval.common.execution import ScoringContext

    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=row["prediction"])],
        )
        for instance, row in joined
    ]
    responses = asyncio.run(task.score_responses(responses, ScoringContext()))
    nested = task.compute_metrics(responses)
    score = next(iter(nested["score"].values()))
    assert score == pytest.approx(ref["score"], abs=0.01), (
        f"GPT-extraction score {score:.4f} vs reference {ref['score']:.4f}"
    )

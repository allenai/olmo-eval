"""Validation harness for SAGE's deterministic title matcher.

This module keeps two validation layers separate. The adversarial layer is a
small, deterministic set of hand-written cases that documents current
``NormalizedStringMatcher`` behavior, including known substring limitations. The
human-agreement layer evaluates that same matcher against the shipped,
human-labeled SAGE validation JSONL and reports aggregate agreement metrics.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, TypedDict, cast

from olmo_eval.evals.tasks.sage import (
    GoldPaper,
    Matcher,
    NormalizedStringMatcher,
    strip_think,
)

VALIDATION_DATA_FILE = "sage_matcher_validation.jsonl"


@dataclass(frozen=True, slots=True)
class SageMatchAdversarialCase:
    """A hand-constructed SAGE matcher case with documented expected behavior."""

    name: str
    description: str
    gold_title: str
    output: str
    expected: bool


@dataclass(frozen=True, slots=True)
class SageMatcherValidationExample:
    """A human-labeled SAGE matcher validation row."""

    task: str
    gold_title: str
    output_text: str
    identified: bool


class AgreementMetrics(TypedDict):
    """Aggregate matcher agreement with human labels."""

    matcher: str
    n: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int


ADVERSARIAL_CASES: tuple[SageMatchAdversarialCase, ...] = (
    SageMatchAdversarialCase(
        name="exact_title_present",
        description="The output names the exact gold title.",
        gold_title="Attention Is All You Need",
        output="Final answer: Attention Is All You Need.",
        expected=True,
    ),
    SageMatchAdversarialCase(
        name="gold_substring_of_longer_title",
        description="The normalized gold title appears inside a longer listed title.",
        gold_title="Attention Is All You Need",
        output="Candidate: Attention Is All You Need for Long-Context Retrieval.",
        expected=True,
    ),
    SageMatchAdversarialCase(
        name="punctuation_and_case_normalized",
        description="Case and punctuation differences are normalized away.",
        gold_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        output="retrieval augmented generation for knowledge intensive nlp tasks",
        expected=True,
    ),
    SageMatchAdversarialCase(
        name="unrelated_absent",
        description="The output names an unrelated paper and never includes the gold title.",
        gold_title="Dense Passage Retrieval for Open-Domain Question Answering",
        output="Final answer: REALM: Retrieval-Augmented Language Model Pre-Training.",
        expected=False,
    ),
    SageMatchAdversarialCase(
        name="empty_output",
        description="No visible output cannot identify the gold title.",
        gold_title="Dense Passage Retrieval for Open-Domain Question Answering",
        output="",
        expected=False,
    ),
    # Known substring false-positive limitation: the matcher does not model
    # rejection language and only checks whether the normalized title appears.
    SageMatchAdversarialCase(
        name="mentioned_but_rejected",
        description="The output rejects the gold title, but the substring matcher still fires.",
        gold_title="Predictive Coding Networks for Temporal Prediction",
        output=(
            'The closest match is "Predictive Coding Networks for Temporal Prediction", '
            "but no match was found."
        ),
        expected=True,
    ),
    SageMatchAdversarialCase(
        name="paraphrased_same_paper_title",
        description="A paraphrased same-paper title is not a normalized substring match.",
        gold_title="ConR: Contrastive Regularizer for Deep Imbalanced Regression",
        output=(
            "The answer is the ConR paper on contrastive regularization for imbalanced "
            "deep regression."
        ),
        expected=False,
    ),
)


def _gold_from_title(title: str) -> GoldPaper:
    return cast(GoldPaper, {"title": title})


async def run_adversarial(
    matcher: Matcher | None = None,
) -> list[tuple[SageMatchAdversarialCase, bool, bool]]:
    """Run the documented adversarial cases against a SAGE matcher."""
    matcher = matcher or NormalizedStringMatcher()
    results: list[tuple[SageMatchAdversarialCase, bool, bool]] = []
    for case in ADVERSARIAL_CASES:
        actual = await matcher.matched(_gold_from_title(case.gold_title), strip_think(case.output))
        results.append((case, actual, actual == case.expected))
    return results


def load_human_validation_examples() -> list[SageMatcherValidationExample]:
    """Load the shipped human-labeled SAGE matcher validation set."""
    validation_path = files("olmo_eval.evals.tasks") / VALIDATION_DATA_FILE
    examples: list[SageMatcherValidationExample] = []
    for line_number, line in enumerate(validation_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(_parse_validation_row(row, line_number))
    return examples


def _parse_validation_row(row: dict[str, Any], line_number: int) -> SageMatcherValidationExample:
    try:
        task = row["task"]
        gold_title = row["gold_title"]
        output_text = row["output_text"]
        identified = row["identified"]
    except KeyError as exc:
        location = f"{VALIDATION_DATA_FILE}:{line_number}"
        raise ValueError(f"Missing {exc.args[0]!r} in {location}") from exc

    if not isinstance(task, str):
        raise ValueError(f"Invalid task in {VALIDATION_DATA_FILE}:{line_number}")
    if not isinstance(gold_title, str):
        raise ValueError(f"Invalid gold_title in {VALIDATION_DATA_FILE}:{line_number}")
    if not isinstance(output_text, str):
        raise ValueError(f"Invalid output_text in {VALIDATION_DATA_FILE}:{line_number}")
    if not isinstance(identified, bool):
        raise ValueError(f"Invalid identified label in {VALIDATION_DATA_FILE}:{line_number}")

    return SageMatcherValidationExample(
        task=task,
        gold_title=gold_title,
        output_text=output_text,
        identified=identified,
    )


async def evaluate_matcher_agreement(matcher: Matcher | None = None) -> AgreementMetrics:
    """Compute matcher agreement metrics against human ``identified`` labels."""
    matcher = matcher or NormalizedStringMatcher()
    tp = fp = fn = tn = 0

    for example in load_human_validation_examples():
        actual = await matcher.matched(
            _gold_from_title(example.gold_title),
            strip_think(example.output_text),
        )
        expected = example.identified
        if actual and expected:
            tp += 1
        elif actual and not expected:
            fp += 1
        elif not actual and expected:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "matcher": matcher.name,
        "n": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


async def _main() -> None:
    matcher = NormalizedStringMatcher()

    print(f"SAGE matcher validation: {matcher.name}")
    print()
    print("Adversarial cases:")
    print(f"{'result':<6} {'expected':<8} {'actual':<6} name")
    for case, actual, ok in await run_adversarial(matcher):
        result = "PASS" if ok else "FAIL"
        print(f"{result:<6} {str(case.expected):<8} {str(actual):<6} {case.name}")
        print(f"       {case.description}")

    metrics = await evaluate_matcher_agreement(matcher)
    print()
    print("Human agreement:")
    print("n   tp  fp  fn  tn  precision  recall  f1     accuracy")
    print(
        f"{metrics['n']:<3} {metrics['tp']:<3} {metrics['fp']:<3} "
        f"{metrics['fn']:<3} {metrics['tn']:<3} "
        f"{metrics['precision']:.3f}      {metrics['recall']:.3f}   "
        f"{metrics['f1']:.3f}  {metrics['accuracy']:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(_main())

"""Tests for SAGE paper matchers."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.evals.tasks.sage import (
    GoldPaper,
    Matcher,
    NormalizedStringMatcher,
    exact_match,
    make_gold,
    normalize_title,
    strip_think,
    weighted_recall,
)


@dataclass(frozen=True)
class AdversarialMatchCase:
    """A constructed SAGE match case with known-correct expected behavior."""

    name: str
    description: str
    gold: GoldPaper
    output: str
    expect: bool


def build_cases() -> list[AdversarialMatchCase]:
    """Construct adversarial matcher cases."""
    attention = make_gold(
        "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        "Attention Is All You Need",
        "The Transformer uses self-attention for sequence transduction.",
        arxiv_id="1706.03762",
        doi="10.5555/3295222.3295349",
        corpus_id="13756489",
    )
    rag = make_gold(
        "0f5b1f4a7f2f65b5f4d9128b6a3262b195ad0f2a",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "RAG combines parametric generation with non-parametric retrieval.",
        arxiv_id="2005.11401",
        corpus_id="219570639",
    )
    dense = make_gold(
        "55076bc106ad7f7faf9756d955b2741741c84e3f",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "DPR learns dense retrievers for open-domain question answering.",
        arxiv_id="2004.04906",
        corpus_id="215416146",
    )
    return [
        AdversarialMatchCase(
            name="verbatim_title_present",
            description="The final output names the exact gold title.",
            gold=attention,
            output="Final answer: Attention Is All You Need introduced the Transformer.",
            expect=True,
        ),
        AdversarialMatchCase(
            name="paraphrased_reference",
            description=(
                "The answer is an author/year-style paraphrase; normalized title "
                "matching is expected to miss this."
            ),
            gold=rag,
            output=(
                "The answer is the Lewis et al. 2020 paper that introduced retrieval-"
                "augmented generation for knowledge-intensive NLP."
            ),
            expect=True,
        ),
        AdversarialMatchCase(
            name="sibling_near_miss_title",
            description=(
                "A near-miss sibling that reuses all gold title tokens is selected "
                "instead of the gold paper."
            ),
            gold=rag,
            output=(
                "Final answer: Knowledge-Intensive NLP Tasks for Retrieval-Augmented "
                "Generation: A Survey is the target paper."
            ),
            expect=False,
        ),
        AdversarialMatchCase(
            name="title_only_in_think",
            description="The gold title appears only in hidden reasoning and must not count.",
            gold=attention,
            output=(
                "<think>Maybe the answer is Attention Is All You Need.</think>\n"
                "Final answer: I could not identify the target paper."
            ),
            expect=False,
        ),
        AdversarialMatchCase(
            name="candidate_list_not_selected",
            description=(
                "The gold title is listed as a candidate but a different paper is selected."
            ),
            gold=dense,
            output=(
                "Candidates considered: Dense Passage Retrieval for Open-Domain "
                "Question Answering; REALM. Final answer: REALM is the paper."
            ),
            expect=False,
        ),
        AdversarialMatchCase(
            name="id_echo",
            description="The output echoes an external gold identifier without naming the title.",
            gold=dense,
            output="Final answer: the relevant Semantic Scholar Corpus ID is 215416146.",
            expect=True,
        ),
    ]


async def evaluate_matcher(
    matcher: Matcher, cases: Sequence[AdversarialMatchCase]
) -> dict[str, float]:
    """Evaluate a matcher against expected case labels."""
    tp = fp = tn = fn = 0
    for case in cases:
        predicted = bool(await exact_match(matcher, case.gold, case.output))
        if predicted and case.expect:
            tp += 1
        elif predicted and not case.expect:
            fp += 1
        elif not predicted and case.expect:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(cases) if cases else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n": float(len(cases)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


ATTENTION = make_gold("S2-ATTN", "Attention Is All You Need", "Transformer paper.")
DPR = make_gold(
    "S2-DPR",
    "Dense Passage Retrieval for Open-Domain Question Answering",
    "Dense retrieval paper.",
    arxiv_id="2004.04906",
    corpus_id="215416146",
)
RAG = make_gold(
    "S2-RAG",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "RAG paper.",
    arxiv_id="2005.11401",
)


def run_bool(coro: Coroutine[Any, Any, bool]) -> bool:
    return asyncio.run(coro)


def run_float(coro: Coroutine[Any, Any, float]) -> float:
    return asyncio.run(coro)


def run_metrics(coro: Coroutine[Any, Any, dict[str, float]]) -> dict[str, float]:
    return asyncio.run(coro)


def test_normalize_title() -> None:
    assert normalize_title("Attention: Is ALL You Need!") == "attention is all you need"
    assert normalize_title("  Dense\nPassage\tRetrieval  ") == "dense passage retrieval"


def test_strip_think() -> None:
    assert strip_think("Final answer") == "Final answer"
    assert strip_think("<think>scratch</think>Final answer") == "Final answer"
    assert strip_think("<think>first</think><think>second</think>Final") == "Final"
    assert strip_think("<think>reasoning <think>nested</think> Attention Is All You Need") == ""
    assert strip_think("Before </think> After") == "Before </think> After"
    assert strip_think("<think>gold (truncated)") == ""
    assert strip_think("Final <think>truncated Attention Is All You Need") == "Final "
    assert strip_think("<think>closed</think>Final <think>truncated") == "Final "


def test_normalized_string_matcher() -> None:
    matcher = NormalizedStringMatcher()

    assert run_bool(matcher.matched(ATTENTION, "Final answer: attention is all you need."))
    assert run_bool(matcher.matched(ATTENTION, "Final answer: Attention -- Is All You Need!"))
    assert not run_bool(matcher.matched(ATTENTION, "Final answer: Attention Is Not Enough."))
    assert not run_bool(
        matcher.matched(
            RAG,
            "Final answer: Retrieval-Augmented Models for Knowledge-Intensive NLP Tasks.",
        )
    )


@dataclass(frozen=True)
class StubMatcher:
    matched_ids: set[str]
    name: str = "stub"

    async def matched(self, gold: GoldPaper, output: str) -> bool:
        return gold["paperId"] in self.matched_ids and "no-match" not in output


def test_exact_match_strips_think_with_stub() -> None:
    matcher: Matcher = StubMatcher({"S2-ATTN"})

    assert (
        run_float(
            exact_match(
                matcher,
                ATTENTION,
                "<think>Attention Is All You Need</think>Final answer: no-match",
            )
        )
        == 0.0
    )
    assert run_float(exact_match(matcher, ATTENTION, "Final answer")) == 1.0


def test_weighted_recall_math_with_stub() -> None:
    matcher: Matcher = StubMatcher({"S2-ATTN", "S2-DPR"})

    score = run_float(weighted_recall(matcher, [(ATTENTION, 2), (DPR, 1), (RAG, 1)], ""))

    assert score == 0.75
    assert run_float(weighted_recall(matcher, [(ATTENTION, 0)], "")) == 0.0


def test_evaluate_matcher_over_constructed_cases() -> None:
    cases = build_cases()
    expected_counts = {
        "normalized_string": {"tp": 1.0, "fp": 1.0, "tn": 2.0, "fn": 2.0},
    }

    matcher = NormalizedStringMatcher()
    metrics = run_metrics(evaluate_matcher(matcher, cases))

    assert metrics["n"] == float(len(cases))
    for key, expected in expected_counts[matcher.name].items():
        assert metrics[key] == expected


def test_constructed_case_specific_behaviors() -> None:
    cases = {case.name: case for case in build_cases()}
    normalized = NormalizedStringMatcher()

    candidate = cases["candidate_list_not_selected"]
    normalized_metrics = run_metrics(evaluate_matcher(normalized, list(cases.values())))
    assert candidate.expect is False
    assert normalized_metrics["fp"] >= 1.0
    assert run_float(exact_match(normalized, candidate.gold, candidate.output)) == 1.0

    think = cases["title_only_in_think"]
    assert run_float(exact_match(normalized, think.gold, think.output)) == 0.0

    id_echo = cases["id_echo"]
    assert run_float(exact_match(normalized, id_echo.gold, id_echo.output)) == 0.0

    near_miss = cases["sibling_near_miss_title"]
    assert run_float(exact_match(normalized, near_miss.gold, near_miss.output)) == 0.0

"""Tests for ALCE's AutoAIS citation scoring, against a rule-based stand-in NLI model."""

from __future__ import annotations

import re
import types

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.alce import (
    AUTOAIS_KEY,
    AutoAisScorer,
    citation_sentences,
)
from olmo_eval.common.scorers.execution import ScoringIncompleteError
from olmo_eval.common.types import Instance, LMOutput, Response

DOCS = [
    {"title": "Paris", "text": "Paris is the capital of France."},
    {"title": "Europe", "text": "France is in Europe."},
    {"title": "Cats", "text": "Cats sleep a lot."},
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


class EntailmentProvider:
    """Says "1" when every word of the hypothesis appears in the premise."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def agenerate(self, requests, params):  # noqa: ANN001, ANN201
        prompt = requests[0].prompt
        self.prompts.append(prompt)
        premise, hypothesis = re.match(r"premise: (.*) hypothesis: (.*)", prompt, re.S).groups()
        verdict = "1" if _words(hypothesis) <= _words(premise) else "0"
        return [[LMOutput(text=verdict)]]


def _context(provider: object) -> ScoringContext:
    return ScoringContext(
        inference_pool=types.SimpleNamespace(get=lambda name: provider, names=["autoais"])
    )


async def _score(text: str, *, qampari: bool = False, question: str = "q", docs=DOCS):
    provider = EntailmentProvider()
    instance = Instance(
        question="PROMPT", metadata={"citation_docs": docs, "judge_question": question}
    )
    output = LMOutput(text=text)
    score = await AutoAisScorer(qampari=qampari).ascore_with_context(
        instance, output, _context(provider)
    )
    return score, output.metadata[AUTOAIS_KEY], provider.prompts


@pytest.mark.anyio
async def test_recall_and_precision_follow_compute_autoais():
    # 1. cites [1][3]: entailed jointly; [1] alone entails it, while [3] neither
    #    does nor is missed when left out, so it is an over-citation
    # 2. cites [2], which entails it
    # 3. cites nothing, so it is not entailed
    score, result, prompts = await _score(
        "Paris is the capital of France [1][3]. France is in Europe [2]. No source here."
    )

    assert score == result["citation_rec"] == pytest.approx(2 / 3)
    assert result["citation_prec"] == pytest.approx(2 / 3)
    assert result["claims"] == 3
    # the joint check, [1] alone, [3] alone, [3] left out, then sentence two
    assert len(prompts) == len(result["nli_calls"]) == 5
    assert prompts[0] == (
        "premise: Title: Paris\nParis is the capital of France.\nTitle: Cats\nCats sleep a lot. "
        "hypothesis: Paris is the capital of France."
    )


@pytest.mark.anyio
async def test_citations_past_the_shown_documents_are_not_entailed():
    score, result, prompts = await _score("Paris is the capital of France [4].")

    assert score == 0.0
    assert result["citation_prec"] == 0.0
    assert prompts == []


@pytest.mark.anyio
async def test_only_the_first_three_citations_are_checked():
    _, result, _ = await _score("Paris is the capital of France [1][2][3][1].")

    assert result["nli_calls"][0]["cited"] == [1, 2, 3]


@pytest.mark.anyio
async def test_qampari_checks_each_listed_answer_with_the_question():
    score, result, prompts = await _score("Paris [1], Europe [2].", qampari=True, question="France")

    assert score == 1.0
    assert [call["claim"] for call in result["nli_calls"]] == ["France Paris", "France Europe"]
    assert prompts[1].endswith("hypothesis: France Europe")


@pytest.mark.anyio
async def test_an_output_with_no_claims_has_no_citation_scores():
    score, result, prompts = await _score("")

    assert score == 0.0
    assert result == {"claims": 0, "nli_calls": []}
    assert prompts == []


def test_sentences_drop_fragments_and_keep_citations():
    text = "Paris is big [1].\n\nIt is old [2][3]. 1."

    assert citation_sentences(text, "q", qampari=False) == [
        "Paris is big [1].",
        "It is old [2][3].",
    ]


@pytest.mark.anyio
async def test_a_failed_nli_call_leaves_the_output_unscored():
    async def agenerate(requests, params):  # noqa: ANN001, ANN202
        raise RuntimeError("model down")

    provider = types.SimpleNamespace(agenerate=agenerate)
    instance = Instance(question="P", metadata={"citation_docs": DOCS, "judge_question": "q"})

    with pytest.raises(ScoringIncompleteError, match="model down"):
        await AutoAisScorer().ascore_with_context(
            instance, LMOutput(text="Paris is the capital of France [1]."), _context(provider)
        )


@pytest.mark.anyio
async def test_an_instance_without_its_documents_is_refused():
    with pytest.raises(ValueError, match="citation_docs"):
        await AutoAisScorer().ascore_with_context(
            Instance(question="P"), LMOutput(text="x [1]."), _context(EntailmentProvider())
        )


def _response(autoais: dict | None, **scores: float) -> Response:
    metadata = {} if autoais is None else {AUTOAIS_KEY: autoais}
    response = Response(
        instance=Instance(question="P"),
        request=None,
        outputs=[LMOutput(text="x", metadata=metadata)],
    )
    response.scores.update(scores)
    return response


def test_citation_metrics_leave_out_outputs_without_claims():
    from olmo_eval.evals.tasks.helmet import STR_EM, CitationMetric, CiteScoreMetric

    recall = CitationMetric(name="citation_rec")
    precision = CitationMetric(name="citation_prec")
    responses = [
        _response({"claims": 2, "citation_rec": 1.0, "citation_prec": 0.5}, str_em=1.0),
        _response({"claims": 4, "citation_rec": 0.5, "citation_prec": 0.0}, str_em=0.0),
        _response({"claims": 0, "nli_calls": []}, str_em=0.5),
    ]

    assert recall.compute(responses) == 0.75
    assert precision.compute(responses) == 0.25
    assert recall.compute_instance(responses[2]) is None
    cite = CiteScoreMetric(parts=(STR_EM, recall, precision))
    assert cite.compute(responses) == pytest.approx((0.5 + 0.75 + 0.25) / 3)

"""Tests for SAGE short-form answer-only, commit and abstention metrics.

These cover the contract the short-form prompt states ("give your single best
answer: state the most likely paper's title, or explicitly say no match was
found") and, critically, that adding them leaves ``exact_match`` untouched.
"""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.sage import (
    NormalizedStringMatcher,
    SageAccuracyGivenCommitMetric,
    SageAnswerOnlyMatchMetric,
    SageAnswerUnparsedRateMetric,
    SageCommitRateMetric,
    SageExactMatchMetric,
    SageShortForm,
    answer_only_match,
    answer_region,
    answer_statements,
    committed_title,
    extract_answer,
    make_gold,
)

GOLD_TITLE = "Compact Retrieval Benchmarks for Deep Research Agents"


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def gold():
    return make_gold("gold-paper-id", GOLD_TITLE, "A benchmark paper.")


@pytest.fixture
def task():
    return get_task("sage_short_form")


@pytest.fixture
def short_form_doc():
    return {
        "paper_id": "seed-paper-id",
        "complete_query": "Find the paper that introduced a compact retrieval benchmark.",
        "ground_truth": {
            "paperId": "gold-paper-id",
            "title": GOLD_TITLE,
            "abstract": "A benchmark paper.",
        },
    }


def _response(task, doc, *texts):
    instance = task.process_doc(doc)
    assert instance is not None
    return Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=t) for t in texts],
    )


# --- answer region ----------------------------------------------------------


def test_answer_region_drops_think_block():
    text = "<think>The gold paper is X</think>No match was found."
    assert answer_region(text) == "No match was found."


def test_answer_region_drops_unopened_reasoning_region():
    """A closing tag with no opening tag means everything before it is reasoning."""
    text = f"Maybe it is {GOLD_TITLE}, let me check.</think>\nNo match was found."
    assert answer_region(text) == "No match was found."
    # strip_think alone keeps it, which is why exact_match still sees it.
    from olmo_eval.evals.tasks.sage import strip_think

    assert GOLD_TITLE in strip_think(text)


def test_answer_region_drops_trailing_reference_list():
    text = (
        "No match was found among the retrieved candidates, and the constraints "
        "on venue and year could not all be satisfied by any single paper.\n\n"
        f"## References\n\n[1] {GOLD_TITLE}\n[2] Another Paper Title\n"
    )
    region = answer_region(text)
    assert GOLD_TITLE not in region
    assert "No match was found" in region


def test_answer_region_keeps_an_early_reference_mention():
    """A heading in the first 40% is not treated as the bibliography."""
    text = "References\n" + "x" * 400 + f'\nThe paper is "{GOLD_TITLE}".'
    assert GOLD_TITLE in answer_region(text)


# --- statement segmentation -------------------------------------------------


def test_statements_strip_heading_and_bullet_markers():
    assert answer_statements("## Direct Answer\n- one\n1. two") == [
        "Direct Answer",
        "one",
        "two",
    ]


def test_statements_join_a_colon_line_with_its_answer():
    statements = answer_statements(f"The most likely paper is:\n**{GOLD_TITLE}**")
    assert statements == [f"The most likely paper is: **{GOLD_TITLE}**"]


def test_statements_split_sentences():
    assert answer_statements("First one. Second one.") == ["First one.", "Second one."]


# --- commit / decline classification ---------------------------------------


@pytest.mark.parametrize(
    "text",
    (
        f'The paper that matches the query is "{GOLD_TITLE}".',
        f"The single best match is **{GOLD_TITLE}**.",
        f"The most likely paper is [{GOLD_TITLE}](https://example.org/p).",
        f"Answer: {GOLD_TITLE}",
        f"**Final answer:** {GOLD_TITLE}",
        f"The most likely paper is:\n**{GOLD_TITLE}**",
        f"**{GOLD_TITLE}**",
    ),
)
def test_extract_answer_reads_a_committed_title(text):
    answer = extract_answer(text)
    assert answer.kind == "commit"
    assert answer.title is not None
    assert GOLD_TITLE in answer.title


@pytest.mark.parametrize(
    "text",
    (
        "No match was found.",
        "No matching paper was found among the 31 candidate papers retrieved.",
        "Based on the provided summaries, no single paper matches the query.",
        "The provided literature does not contain a paper matching the query.",
        "The specific paper requested cannot be identified from the summaries.",
        "I was unable to identify the paper with the available search tools.",
        "None of the provided candidates match the venue and year constraints.",
        "The target paper does not appear to exist in the indexed corpora.",
    ),
)
def test_extract_answer_reads_a_decline(text):
    assert extract_answer(text).kind == "decline"


def test_extract_answer_reports_unparsed_rather_than_guessing():
    survey = (
        "The dimension of retrieval benchmarks encompasses evaluation design. "
        "Paper [1] contributes a compact benchmark and Paper [2] extends it."
    )
    answer = extract_answer(survey)
    assert answer.kind == "unparsed"
    assert answer.title is None


def test_prompt_echo_heading_is_not_a_commit():
    """A section heading quoting the prompt announces an answer without giving one."""
    heading = "Direct Answer: Identification of the Most Likely Paper's Title"
    assert committed_title(heading) is None


def test_incidental_prose_about_the_paper_is_not_a_commit():
    prose = "The identified paper confirms that the benchmark is widely reused."
    assert committed_title(prose) is None


def test_hedged_output_is_flagged_and_classified_by_reading_order():
    text = f'No single paper satisfies all constraints.\n\nPrimary candidate: "{GOLD_TITLE}".'
    answer = extract_answer(text)
    assert answer.kind == "decline"
    assert answer.hedged is True


def test_first_answer_statement_wins_over_later_mentions():
    """Writing more must not change the answer the metric reads."""
    text = (
        'The paper that matches the query is "A Wrong Paper Title Entirely".\n\n'
        f"For comparison, {GOLD_TITLE} covers a different setting.\n"
    )
    answer = extract_answer(text)
    assert answer.kind == "commit"
    assert answer.title == "A Wrong Paper Title Entirely"


# --- answer_only_match ------------------------------------------------------


@pytest.mark.anyio
async def test_answer_only_match_scores_the_stated_answer(gold):
    matcher = NormalizedStringMatcher()
    hit = f'The paper that matches the query is "{GOLD_TITLE}".'
    assert await answer_only_match(matcher, gold, hit) == 1.0


@pytest.mark.anyio
async def test_answer_only_match_ignores_a_reference_list(gold):
    matcher = NormalizedStringMatcher()
    text = (
        "Based on the provided summaries, no single paper matches the query.\n\n"
        f"## References\n\n[1] {GOLD_TITLE}\n"
    )
    assert await answer_only_match(matcher, gold, text) == 0.0


@pytest.mark.anyio
async def test_answer_only_match_ignores_rejected_candidates(gold):
    matcher = NormalizedStringMatcher()
    text = (
        'The paper that matches the query is "A Wrong Paper Title Entirely". '
        f"I considered {GOLD_TITLE} but rejected it on the year constraint."
    )
    assert await answer_only_match(matcher, gold, text) == 0.0


@pytest.mark.anyio
async def test_answer_only_match_is_zero_for_a_decline(gold):
    matcher = NormalizedStringMatcher()
    assert await answer_only_match(matcher, gold, "No match was found.") == 0.0


# --- task wiring ------------------------------------------------------------


def test_short_form_declares_the_new_metrics_without_changing_primary():
    names = [m.name for m in SageShortForm.metrics]
    assert names == [
        "exact_match",
        "answer_only_match",
        "commit_rate",
        "accuracy_given_commit",
        "answer_unparsed_rate",
    ]
    assert SageShortForm.primary_metric.name == "exact_match"


@pytest.mark.anyio
async def test_exact_match_is_unchanged_by_the_new_metrics(task, short_form_doc):
    """A gold title that only appears in a reference list still scores exact_match=1."""
    text = (
        "Based on the provided summaries, no single paper matches the query.\n\n"
        f"## References\n\n[1] {GOLD_TITLE}\n"
    )
    scored = await task.score_responses([_response(task, short_form_doc, text)])

    assert scored[0].scores["exact_match"] == 1.0
    assert scored[0].outputs[0].metadata["exact_match"] == 1.0
    assert scored[0].outputs[0].metadata["sage_matched"] is True
    # ... and the answer-only view disagrees, which is the point.
    assert scored[0].scores["answer_only_match"] == 0.0
    assert scored[0].scores["commit_rate"] == 0.0


@pytest.mark.anyio
async def test_score_responses_records_answer_metadata(task, short_form_doc):
    text = f'The paper that matches the query is "{GOLD_TITLE}".'
    scored = await task.score_responses([_response(task, short_form_doc, text)])
    metadata = scored[0].outputs[0].metadata

    assert metadata["sage_answer_kind"] == "commit"
    assert metadata["sage_answer_title"] == GOLD_TITLE
    assert metadata["sage_answer_hedged"] is False
    assert metadata["answer_only_match"] == 1.0
    assert metadata["score:answer_only_match"] == 1.0
    assert scored[0].scores["answer_unparsed_rate"] == 0.0


@pytest.mark.anyio
async def test_metrics_split_declining_from_being_wrong(task, short_form_doc):
    right = f'The best match is "{GOLD_TITLE}".'
    wrong = 'The best match is "A Wrong Paper Title Entirely".'
    declined = "No match was found."
    scored = await task.score_responses(
        [
            _response(task, short_form_doc, right),
            _response(task, short_form_doc, wrong),
            _response(task, short_form_doc, declined),
            _response(task, short_form_doc, declined),
        ]
    )

    assert SageExactMatchMetric().compute(scored) == pytest.approx(0.25)
    assert SageAnswerOnlyMatchMetric().compute(scored) == pytest.approx(0.25)
    assert SageCommitRateMetric().compute(scored) == pytest.approx(0.5)
    # exact_match restricted to the two committed instances
    assert SageAccuracyGivenCommitMetric().compute(scored) == pytest.approx(0.5)
    assert SageAnswerUnparsedRateMetric().compute(scored) == pytest.approx(0.0)


@pytest.mark.anyio
async def test_accuracy_given_commit_is_zero_when_nothing_is_committed(task, short_form_doc):
    scored = await task.score_responses([_response(task, short_form_doc, "No match was found.")])
    assert SageAccuracyGivenCommitMetric().compute(scored) == 0.0
    assert SageCommitRateMetric().compute(scored) == 0.0


@pytest.mark.anyio
async def test_unparsed_output_counts_as_neither_commit_nor_decline(task, short_form_doc):
    survey = (
        "The dimension of retrieval benchmarks encompasses evaluation design. "
        "Paper [1] contributes a compact benchmark and Paper [2] extends it."
    )
    scored = await task.score_responses([_response(task, short_form_doc, survey)])

    assert scored[0].scores["commit_rate"] == 0.0
    assert scored[0].scores["answer_unparsed_rate"] == 1.0
    assert scored[0].outputs[0].metadata["sage_answer_kind"] == "unparsed"


def test_metrics_compute_on_empty_responses():
    for metric in (
        SageAnswerOnlyMatchMetric(),
        SageCommitRateMetric(),
        SageAccuracyGivenCommitMetric(),
        SageAnswerUnparsedRateMetric(),
    ):
        assert metric.compute([]) == 0.0


def test_accuracy_given_commit_has_no_per_instance_value():
    """It is a conditional mean, so pairwise per-instance fallback must be off."""
    metric = SageAccuracyGivenCommitMetric()
    instance = Instance(question="q", metadata={})
    request = LMRequest(request_type=RequestType.CHAT, messages=())
    response = Response(
        instance=instance,
        request=request,
        outputs=[],
        scores={"exact_match": 1.0, "commit_rate": 1.0},
    )
    assert metric.compute_instance(response) is None
    assert metric.supports_pairwise_scorer_fallback() is False

"""Tests for SAGE short-form answer-only, commit and abstention metrics.

These cover the contract the short-form prompt states ("give your single best
answer: state the most likely paper's title, or explicitly say no match was
found") and, critically, that adding them leaves ``exact_match`` untouched.
"""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.sage import (
    _MAX_RECORDED_STATEMENT_CHARS,
    NormalizedStringMatcher,
    SageAccuracyGivenCommitMetric,
    SageAnswerOnlyMatchMetric,
    SageAnswerUnparsedRateMetric,
    SageCommitCountMetric,
    SageCommitRateMetric,
    SageDeclineRateMetric,
    SageExactMatchMetric,
    SageHedgeRateMetric,
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
    # This test was written when strip_think kept the unopened region, so a title
    # guessed in the monologue still scored under exact_match and answer_region was
    # the only thing that dropped it. `Treat an unopened </think> as reasoning in
    # SAGE` closed that path too, so both drop it now and the contrast is gone.
    # Recorded rather than deleted: it means exact_match is NOT byte-identical to
    # the metric the published SAGE numbers were measured under -- a run whose gold
    # title appeared only inside a leaked monologue used to score it and no longer
    # does.
    from olmo_eval.evals.tasks.sage import strip_think

    assert GOLD_TITLE not in strip_think(text)


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


def test_answer_announcing_heading_is_not_a_commit():
    """A section heading announces an answer without giving one.

    What rejects it is the pair of requirements every commit has to meet -- a
    predicate *and* a title the statement sets apart -- not a blocklist of
    prompt-echo phrasings: the heading delimits no title, and ``_ANSWER_LABEL``
    is anchored at the start of the statement so it cannot skip the leading
    "Direct ".
    """
    heading = "Direct Answer: Identification of the Most Likely Paper's Title"
    assert committed_title(heading) is None
    assert extract_answer(heading).kind == "unparsed"


@pytest.mark.parametrize(
    "text",
    (
        f'The most likely paper title is **"{GOLD_TITLE}"**.',
        f"The most likely paper's title is: **{GOLD_TITLE}**",
        f'Based on the analysis, the most likely paper title is **"{GOLD_TITLE}"**.',
    ),
)
def test_the_commonest_real_answer_phrasing_is_a_commit(text):
    """These are answers, and a guard on the phrase "most likely paper title" ate them.

    That guard suppressed 25 statements over 19 instances across the five runs we
    have -- 6 of those statements naming the gold paper -- and rejected no heading
    that :func:`committed_title`'s two requirements did not already reject.
    """
    answer = extract_answer(text)
    assert answer.kind == "commit"
    assert answer.title == GOLD_TITLE


def test_incidental_prose_about_the_paper_is_not_a_commit():
    prose = "The identified paper confirms that the benchmark is widely reused."
    assert committed_title(prose) is None


def test_hedged_output_is_neither_an_unambiguous_commit_nor_decline():
    text = f'No single paper satisfies all constraints.\n\nPrimary candidate: "{GOLD_TITLE}".'
    answer = extract_answer(text)
    assert answer.states_decline is True
    assert answer.states_commit is True
    assert answer.hedged is True
    # ... so it counts towards neither rate.
    assert answer.commits_without_hedging is False
    assert answer.declines_without_hedging is False
    # The single answer answer_only_match scores is still first-in-reading-order.
    assert answer.kind == "decline"


def test_hedge_classification_does_not_depend_on_reading_order():
    """Which of the two comes first is the whole 0.404-to-0.933 band on AgentDisCo.

    The rate metrics must not read it, so commit/decline/hedge classification is
    identical whichever way round the output puts them.
    """
    decline_first = f'No match was found. The best match is "{GOLD_TITLE}".'
    commit_first = f'The best match is "{GOLD_TITLE}". No match was found.'
    for text in (decline_first, commit_first):
        answer = extract_answer(text)
        assert answer.hedged is True
        assert answer.commits_without_hedging is False
        assert answer.declines_without_hedging is False


def test_first_answer_statement_wins_over_later_ones():
    """Reading order decides, and writing more must not change the answer read.

    Both mentions are delimited, so both are commit candidates under every
    tie-break rule and reading order is the only thing separating them. An
    undelimited second mention would prove the delimiter requirement instead,
    and pass unchanged under a last-wins rule.
    """
    text = (
        'The paper that matches the query is "A Wrong Paper Title Entirely".\n\n'
        f'For comparison, the paper that matches the query is "{GOLD_TITLE}", '
        "which covers a different setting.\n"
    )
    # Both statements really are commits, so last-wins would return the gold one.
    assert [committed_title(s) for s in answer_statements(answer_region(text))] == [
        "A Wrong Paper Title Entirely",
        GOLD_TITLE,
    ]

    answer = extract_answer(text)
    assert answer.kind == "commit"
    assert answer.title == "A Wrong Paper Title Entirely"


def test_a_described_paper_is_not_read_as_the_title():
    """The first delimited span after the predicate is not always the title."""
    text = f'The best match is **the 2024 ACL paper** "{GOLD_TITLE}".'
    answer = extract_answer(text)
    assert answer.kind == "commit"
    assert answer.title == GOLD_TITLE


def test_markdown_bold_pairing_does_not_manufacture_a_title():
    """``**Primary candidate:** The paper **"<Title>"**`` pairs the label's closing
    ``**`` with the title's opening ``**``, so the first bold span is ``The paper``.

    Real, on AgentDisCo docs 31, 321 and 326; on 326 the span it displaced is the
    gold title.
    """
    text = f'**Primary candidate:** The paper **"{GOLD_TITLE}"** is the strongest match.'
    assert extract_answer(text).title == GOLD_TITLE


def test_a_bold_section_heading_is_not_a_bare_answer():
    """``**Final Answer**`` above the answer used to be extracted as the title."""
    text = "**Final Answer**\n\nThe candidate list was inconclusive.\n"
    answer = extract_answer(text)
    assert answer.kind == "unparsed"
    assert answer.title is None


def test_a_bold_heading_opening_a_section_is_not_a_title():
    """``**Claims with Strong Consensus**`` opens a section on Arman's docs 22, 83."""
    text = (
        "**Claims with Strong Consensus**\n\n"
        "There is broad agreement in the literature that generative models "
        "transfer poorly across domains.\n"
    )
    answer = extract_answer(text)
    assert answer.kind == "unparsed"
    assert answer.title is None


def test_a_bare_bold_title_is_still_the_answer():
    """The heading rule must not cost a genuine bare answer.

    A gold title is a heading only if it is five words or fewer with no colon and
    no digit -- 20 of SAGE short-form's 599 gold titles -- and only when text
    follows it. All 12 bare answers in the five runs survive it.
    """
    assert extract_answer(f"**{GOLD_TITLE}**").title == GOLD_TITLE
    with_explanation = (
        f"**{GOLD_TITLE}**\n\nThis paper, published in 2024, matches every constraint.\n"
    )
    assert extract_answer(with_explanation).title == GOLD_TITLE


def test_a_long_answer_statement_keeps_its_title():
    """600 chars bounds what is recorded, not what is matched.

    Applied to matching it destroyed the committed title on 13 AgentDisCo
    instances, 3 of them the gold paper, because truncation does not reject a
    long statement -- it searches its first 600 characters anyway.
    """
    padding = "It rules each retrieved candidate out on the venue constraint, " * 12
    statement = f'{padding}so the paper that matches the query is "{GOLD_TITLE}".'
    assert len(statement) > _MAX_RECORDED_STATEMENT_CHARS

    answer = extract_answer(statement)
    assert answer.kind == "commit"
    assert answer.title == GOLD_TITLE
    # ... and what gets written into the predictions file is still bounded.
    assert len(answer.statement) == _MAX_RECORDED_STATEMENT_CHARS


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
        "decline_rate",
        "hedge_rate",
        "answer_unparsed_rate",
        "accuracy_given_commit",
        "commit_count",
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
    assert metadata["sage_states_commit"] is True
    assert metadata["sage_states_decline"] is False
    assert metadata["sage_committed"] == 1.0
    assert metadata["sage_declined"] == 0.0
    assert metadata["answer_only_match"] == 1.0
    assert metadata["score:answer_only_match"] == 1.0
    assert scored[0].scores["answer_unparsed_rate"] == 0.0
    assert scored[0].scores["hedge_rate"] == 0.0
    assert scored[0].scores["decline_rate"] == 0.0


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
    assert SageDeclineRateMetric().compute(scored) == pytest.approx(0.5)
    assert SageHedgeRateMetric().compute(scored) == pytest.approx(0.0)
    # the stated answer, restricted to the two committed instances
    assert SageAccuracyGivenCommitMetric().compute(scored) == pytest.approx(0.5)
    assert SageCommitCountMetric().compute(scored) == 2.0
    assert SageAnswerUnparsedRateMetric().compute(scored) == pytest.approx(0.0)


@pytest.mark.anyio
async def test_the_four_rate_metrics_partition_the_run(task, short_form_doc):
    """commit / decline / hedge / unparsed cover every instance exactly once."""
    texts = (
        f'The best match is "{GOLD_TITLE}".',
        "No match was found.",
        f'No match was found. The best match is "{GOLD_TITLE}".',
        "The dimension of retrieval benchmarks encompasses evaluation design.",
    )
    scored = await task.score_responses([_response(task, short_form_doc, t) for t in texts])

    rates = [
        SageCommitRateMetric().compute(scored),
        SageDeclineRateMetric().compute(scored),
        SageHedgeRateMetric().compute(scored),
        SageAnswerUnparsedRateMetric().compute(scored),
    ]
    assert rates == [pytest.approx(0.25)] * 4
    assert sum(rates) == pytest.approx(1.0)


@pytest.mark.anyio
async def test_hedge_rate_is_the_width_of_the_commit_rate_band(task, short_form_doc):
    """No tie-break rule can put a commit rate outside [cr, cr + hedge_rate].

    One unambiguous commit, one unambiguous decline and two hedges: a
    prefer-decline rule reports 0.25, a prefer-commit rule 0.75, and those are
    exactly ``commit_rate`` and ``commit_rate + hedge_rate``. On AgentDisCo the
    same two edges are 0.404 and 0.933, which is why the single number the metric
    used to report was reporting the tie-break rule.
    """
    texts = (
        f'The best match is "{GOLD_TITLE}".',
        "No match was found.",
        f'No match was found. The best match is "{GOLD_TITLE}".',
        f'The best match is "{GOLD_TITLE}". No match was found.',
    )
    scored = await task.score_responses([_response(task, short_form_doc, t) for t in texts])

    commit_rate = SageCommitRateMetric().compute(scored)
    hedge_rate = SageHedgeRateMetric().compute(scored)
    assert commit_rate == pytest.approx(0.25)
    assert hedge_rate == pytest.approx(0.5)
    assert commit_rate + hedge_rate == pytest.approx(0.75)


@pytest.mark.anyio
async def test_accuracy_given_commit_does_not_credit_a_reference_list(task, short_form_doc):
    """Numerator and denominator must read the same text.

    Conditioning ``exact_match`` on committing scores this instance 1.000: the
    system named the wrong paper and the gold title turned up in its
    bibliography. That was 42% of AgentDisCo's numerator and 100% of Allyson's.
    """
    text = (
        'The paper that matches the query is "A Wrong Paper Title Entirely".\n\n'
        f"## References\n\n[1] {GOLD_TITLE}\n"
    )
    scored = await task.score_responses([_response(task, short_form_doc, text)])

    # SAGE's own metric still sees the gold title, and the system did commit ...
    assert scored[0].scores["exact_match"] == 1.0
    assert scored[0].scores["commit_rate"] == 1.0
    # ... but it did not answer with it.
    assert scored[0].scores["answer_only_match"] == 0.0
    assert SageAccuracyGivenCommitMetric().compute(scored) == 0.0
    assert SageCommitCountMetric().compute(scored) == 1.0


@pytest.mark.anyio
async def test_commit_count_exports_the_denominator(task, short_form_doc):
    """0.3333 from 3 instances must not read like 0.3333 from 309."""
    right = f'The best match is "{GOLD_TITLE}".'
    wrong = 'The best match is "A Wrong Paper Title Entirely".'
    scored = await task.score_responses(
        [
            _response(task, short_form_doc, right),
            _response(task, short_form_doc, wrong),
            _response(task, short_form_doc, wrong),
        ]
        + [_response(task, short_form_doc, "No match was found.") for _ in range(9)]
    )

    assert SageAccuracyGivenCommitMetric().compute(scored) == pytest.approx(1 / 3)
    assert SageCommitCountMetric().compute(scored) == 3.0
    assert SageCommitRateMetric().compute(scored) == pytest.approx(0.25)


@pytest.mark.anyio
async def test_a_thin_denominator_is_logged(task, short_form_doc, caplog):
    scored = await task.score_responses(
        [_response(task, short_form_doc, f'The best match is "{GOLD_TITLE}".')]
    )
    with caplog.at_level("WARNING", logger="olmo_eval.evals.tasks.sage"):
        SageAccuracyGivenCommitMetric().compute(scored)
    assert "accuracy_given_commit" in caplog.text
    assert "commit_count" in caplog.text


@pytest.mark.anyio
async def test_accuracy_given_commit_is_still_gameable_by_abstaining(task, short_form_doc):
    """Documented, not fixed.

    Conditioning on answering is gameable by answering less, whatever the
    numerator is: a system that declines on everything it would get wrong scores
    1.000. On the five runs the ceiling sits at ``commit_rate`` 0.084 for
    AgentDisCo, 0.089 baseline, 0.082 dp1 and 0.052 Arman's. That is why
    ``commit_rate`` and ``commit_count`` are exported beside it, and why
    ``exact_match`` remains the primary metric.
    """
    texts = [f'The best match is "{GOLD_TITLE}".'] + ["No match was found."] * 3
    scored = await task.score_responses([_response(task, short_form_doc, t) for t in texts])

    assert SageAccuracyGivenCommitMetric().compute(scored) == pytest.approx(1.0)
    assert SageCommitRateMetric().compute(scored) == pytest.approx(0.25)
    assert SageCommitCountMetric().compute(scored) == 1.0


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
        SageDeclineRateMetric(),
        SageHedgeRateMetric(),
        SageAccuracyGivenCommitMetric(),
        SageCommitCountMetric(),
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

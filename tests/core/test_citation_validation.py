"""Deterministic (oracle-judge) layer of the citation scorer adversarial validation.

These tests prove that, GIVEN a correct judge, the scoring pipeline penalizes
adversarial citations as expected. Whether the real gpt-4o-mini judge is itself
correct is the separate judge-layer kill test, run manually via
`python -m olmo_eval.common.scorers.citation_validation`.
"""

import pytest

from olmo_eval.common.scorers.citation_validation import (
    build_cases,
    check_expectations,
    citation_scorer_agreement,
    format_report,
    make_oracle_judge,
    run_case,
    run_validation,
)


@pytest.mark.anyio
async def test_all_cases_pass_with_oracle_judge():
    """Each adversarial case meets its expectations when the judge is correct."""
    cases = build_cases()
    for case in cases:
        result = await run_case(case, make_oracle_judge(case))
        assert result.passed, f"{case.name} failed: {result.failures} (scores={result.scores})"


@pytest.mark.anyio
async def test_case_names_are_unique_and_cover_failure_modes():
    cases = build_cases()
    names = {c.name for c in cases}
    assert len(names) == len(cases)
    # The failure modes called out in the review must be present.
    for expected in (
        "supporting_control",
        "topical_non_supporting",
        "title_only_half_credit",
        "citation_stuffed",
        "shuffled_citations",
        "uncited_claim",
    ):
        assert expected in names


@pytest.mark.anyio
async def test_stuffing_hurts_precision_not_recall():
    """Citation stuffing should tank precision while leaving recall high."""
    case = next(c for c in build_cases() if c.name == "citation_stuffed")
    result = await run_case(case, make_oracle_judge(case))
    assert result.scores["citation_precision"] <= 0.3
    assert result.scores["citation_recall"] >= 0.9


@pytest.mark.anyio
async def test_title_only_earns_half_credit():
    case = next(c for c in build_cases() if c.name == "title_only_half_credit")
    result = await run_case(case, make_oracle_judge(case))
    assert result.scores["citation_recall"] == pytest.approx(0.5, abs=0.05)
    assert result.scores["citation_precision"] == pytest.approx(0.5, abs=0.05)


@pytest.mark.anyio
async def test_oracle_validation_detects_a_broken_scorer_via_expectations():
    """A wrong judge (claims everything supported) should fail adversarial cases.

    This guards the harness itself: if the pipeline rubber-stamped citations, the
    expectations would not catch it. Here an always-supports judge makes the
    non-supporting cases violate their upper bounds.
    """

    async def always_supports_judge(prompt, **kwargs):
        import json

        return json.dumps(
            {
                "claims": [
                    {
                        "text": "claim",
                        "supporting": ["[1]"],
                        "non_supporting": [],
                        "is_fully_supported": True,
                    }
                ]
            }
        )

    topical = next(c for c in build_cases() if c.name == "topical_non_supporting")
    result = await run_case(topical, always_supports_judge)
    assert not result.passed  # an over-permissive judge must not pass the kill test


def test_check_expectations_ops():
    assert check_expectations({"x": 0.9}, [("x", ">=", 0.8)]) == []
    assert check_expectations({"x": 0.9}, [("x", "<=", 0.8)]) == ["x=0.900 not <= 0.8"]
    assert check_expectations({}, [("missing", ">=", 0.1)]) == ["missing=0.000 not >= 0.1"]


@pytest.mark.anyio
async def test_format_report_runs():
    cases = build_cases()
    results = await run_validation(make_oracle_judge(cases[0]), cases=[cases[0]])
    report = format_report(results)
    assert "supporting_control" in report
    assert "passed" in report


@pytest.mark.anyio
async def test_agreement_scaffold_empty_returns_zero():
    async def judge(prompt, **kwargs):
        return "{}"

    assert await citation_scorer_agreement([], judge) == {"accuracy": 0.0, "n": 0.0}

"""Tests for the SAGE matcher validation harness."""

from __future__ import annotations

import pytest

from olmo_eval.evals.tasks.sage_match_validation import (
    evaluate_matcher_agreement,
    run_adversarial,
)


@pytest.mark.anyio
async def test_adversarial_cases_match_documented_behavior() -> None:
    results = await run_adversarial()

    assert results
    for case, actual, ok in results:
        assert actual is case.expected
        assert ok is True


@pytest.mark.anyio
async def test_human_validation_agreement_regression_guard() -> None:
    metrics = await evaluate_matcher_agreement()

    assert metrics["n"] == 111
    assert metrics["f1"] >= 0.90

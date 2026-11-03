"""Tests for failure accounting shared by agentic external evals."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import LMOutput
from olmo_eval.evals.external.failures import (
    add_harness_failure,
    failure_budget_error,
    is_harness_failure,
    scaffold_error,
)
from olmo_eval.harness.result import HarnessResult


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, False),
        ("model_error: invalid tool arguments", False),
        ("agent_error: connection refused", True),
        ("image_timeout: pull stalled", True),
        ("verifier_timeout", True),
    ],
)
def test_is_harness_failure(error: str | None, expected: bool) -> None:
    assert is_harness_failure(error) is expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (HarnessResult(final_output=LMOutput(text="")), None),
        (
            HarnessResult(
                final_output=LMOutput(text=""), max_turns_reached=True, error="Max turns exceeded"
            ),
            None,
        ),
        (
            HarnessResult(
                final_output=LMOutput(text=""),
                error="server down",
                metadata={"completion_reason": "error"},
            ),
            "agent_error: server down",
        ),
        (
            HarnessResult(final_output=LMOutput(text=""), error="bad tool arguments"),
            "model_error: bad tool arguments",
        ),
    ],
)
def test_scaffold_error(result: HarnessResult, expected: str | None) -> None:
    assert scaffold_error(result) == expected


def test_budget_ignores_model_errors() -> None:
    assert failure_budget_error(["model_error: x", None], max_error_rate=0.0) is None


def test_budget_is_strict_at_the_boundary() -> None:
    errors = ["agent_error: x"] + [None] * 19
    assert failure_budget_error(errors, max_error_rate=0.05) is None
    breach = failure_budget_error(errors, max_error_rate=0.04)
    assert breach is not None
    assert "1 of 20 instances (5.0%)" in breach
    assert "First failure: agent_error: x" in breach


def test_budget_fails_when_nothing_was_measured_at_any_rate() -> None:
    breach = failure_budget_error(["image_error: x"] * 2, max_error_rate=1.0, unit="trials")
    assert breach is not None
    assert breach.startswith("All 2 trials hit a harness or infrastructure failure.")


def test_budget_with_no_instances_is_not_a_breach() -> None:
    assert failure_budget_error([], max_error_rate=0.05) is None


@pytest.mark.parametrize(
    ("earlier", "expected"),
    [
        (None, "grading_error: boom"),
        ("agent_error: first", "agent_error: first"),
        ("model_error: refusal", "grading_error: boom (after model_error: refusal)"),
    ],
)
def test_add_harness_failure(earlier: str | None, expected: str) -> None:
    combined = add_harness_failure(earlier, "grading_error: boom")
    assert combined == expected
    assert is_harness_failure(combined)

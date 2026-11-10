"""Failure accounting for external evals that run an agent on each instance.

An instance that the harness or infrastructure failed (an image that would not build,
a container that died, a model server that stopped answering, a grader that could not
run) measured nothing about the model. It is left out of the scores and counted
instead, and a run with too many such instances fails, as the runner's failure budget
does for tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.result import HarnessResult

#: Fraction of a run's instances that may hit a harness or infrastructure failure
#: before the run fails. Matches the runner's default failure budget.
DEFAULT_MAX_ERROR_RATE = 0.05

#: Prefix of an instance error that the model itself caused, such as a reply that
#: broke the tool-call protocol. The instance is still scored: the failure is the
#: agent's.
MODEL_ERROR = "model_error"


def is_harness_failure(error: str | None) -> bool:
    """Whether an instance's recorded error means the harness, not the model, failed it."""
    return error is not None and not error.startswith(MODEL_ERROR)


def add_harness_failure(earlier: str | None, error: str) -> str:
    """Record a harness failure on an instance that may already carry an error.

    The first harness failure is kept. A model error recorded earlier gives way, since
    the failure that followed it stopped the instance from being measured, but its
    message is kept after the new one.
    """
    if earlier is None:
        return error
    if is_harness_failure(earlier):
        return earlier
    return f"{error} (after {earlier})"


def scaffold_error(harness_result: HarnessResult) -> str | None:
    """Classify the error a scaffold reports at the end of an agent run.

    A run that used up its turns is the agent's failure and reports no error. vanillux
    reports ``completion_reason="error"`` when the provider could not serve the model,
    which is the harness's failure. Any other reported error, such as openai_agents'
    ``ModelBehaviorError`` or a refusal, is the model's own and stays scored.
    """
    if not harness_result.error or harness_result.max_turns_reached:
        return None
    if harness_result.metadata.get("completion_reason") == "error":
        return f"agent_error: {harness_result.error}"
    return f"{MODEL_ERROR}: {harness_result.error}"


def failure_budget_error(
    errors: list[str | None], max_error_rate: float, unit: str = "instances"
) -> str | None:
    """Describe how a run breached its failure budget, or None when it is within it.

    Args:
        errors: Each instance's recorded error, None for a clean instance.
        max_error_rate: Largest tolerated fraction of harness failures. The comparison
            is strict, so a run exactly on the budget passes. A run in which every
            instance failed measured nothing and fails at any budget.
        unit: What the instances are called in the message.
    """
    failures = [e for e in errors if is_harness_failure(e)]
    if not errors or not failures:
        return None
    detail = f" First failure: {failures[0]}"
    if len(failures) == len(errors):
        return f"All {len(errors)} {unit} hit a harness or infrastructure failure.{detail}"
    rate = len(failures) / len(errors)
    if rate > max_error_rate:
        return (
            f"{len(failures)} of {len(errors)} {unit} ({rate:.1%}) hit a harness or "
            f"infrastructure failure, above max_error_rate={max_error_rate}.{detail}"
        )
    return None

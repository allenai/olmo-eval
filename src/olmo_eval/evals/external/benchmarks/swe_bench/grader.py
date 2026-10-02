"""Grading for SWE-bench predictions.

A prediction is graded in a fresh container built from the instance image, so
nothing the agent did to its own environment can affect the result. The patch
is applied with the same fallback chain as the upstream harness, the dataset's
eval script runs the tests, and the upstream ``swebench`` log parsers decide
whether the instance is resolved.
"""

from __future__ import annotations

import logging
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .task import SWEBenchInstance

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

TESTBED_DIR = "/testbed"
PATCH_PATH = "/tmp/patch.diff"
EVAL_SCRIPT_PATH = "/eval.sh"
EVAL_LOG_PATH = "/tmp/eval_output.log"

# Mirrors the upstream harness: later commands are more lenient.
GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
)

MAX_STORED_LOG_CHARS = 20000


@dataclass
class GradeResult:
    """Outcome of grading one prediction.

    Attributes:
        resolved: Whether every FAIL_TO_PASS and PASS_TO_PASS test passes.
        patch_applied: Whether the patch applied to a fresh checkout.
        tests_ran: Whether the test log showed a test run the grader could read.
        empty_patch: Whether the prediction contained no changes.
        tests_status: Upstream per-test report, when the log was parsed.
        test_output: Tail of the eval script output.
        error: Description of an infrastructure failure, if any.
    """

    resolved: bool = False
    patch_applied: bool = False
    tests_ran: bool = False
    empty_patch: bool = False
    tests_status: dict[str, Any] = field(default_factory=dict)
    test_output: str = ""
    error: str | None = None


def _tail(text: str, limit: int = MAX_STORED_LOG_CHARS) -> str:
    return text if len(text) <= limit else text[-limit:]


def normalize_patch(patch: str) -> str:
    """Return the patch with a trailing newline, or an empty string if it has no content."""
    if not patch.strip():
        return ""
    return patch if patch.endswith("\n") else patch + "\n"


def grade_log(
    instance: SWEBenchInstance, patch: str, test_log: str
) -> tuple[bool, bool, dict[str, Any]]:
    """Grade an eval script log with the upstream ``swebench`` grader.

    Returns:
        Tuple of (resolved, tests_ran, tests_status).
    """
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    test_spec = make_test_spec(instance.to_swebench_dict())
    prediction = {
        "instance_id": instance.instance_id,
        "model_name_or_path": "olmo-eval",
        "model_patch": patch,
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test_output.txt"
        log_path.write_text(test_log)
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(log_path),
            include_tests_status=True,
        )[instance.instance_id]
    return (
        bool(report.get("resolved")),
        bool(report.get("patch_successfully_applied")),
        report.get("tests_status") or {},
    )


def _with_scratch_index(index_path: str) -> str:
    """Shell prefix that stages into a copy of the index, leaving the real one untouched."""
    return (
        f'cd {TESTBED_DIR} && cp "$(git rev-parse --git-path index)" {index_path} && '
        f"export GIT_INDEX_FILE={index_path} && git add -A"
    )


async def snapshot_worktree(executor: SandboxExecutor) -> str:
    """Record the current working tree as a git tree object and return its hash.

    Instance images can carry uncommitted environment changes and build
    artifacts. Diffing against this snapshot, instead of the base commit, keeps
    those out of the agent's patch. HEAD and the index are not modified.
    """
    result = await executor.execute_command(
        f"{_with_scratch_index('/tmp/_snapshot.index')} && git write-tree", timeout=300.0
    )
    lines = result.output.strip().splitlines()
    if not result.success or not lines:
        raise RuntimeError(f"Failed to snapshot working tree: {result.output[-2000:]}")
    return lines[-1].strip()


async def extract_patch(executor: SandboxExecutor, baseline: str) -> str:
    """Return every change in the working tree relative to ``baseline``.

    Args:
        executor: Executor for the container the agent worked in.
        baseline: Tree or commit to diff against, usually from ``snapshot_worktree``.
    """
    out = "/tmp/_model.patch"
    result = await executor.execute_command(
        f"{_with_scratch_index('/tmp/_extract.index')} && "
        f"git diff --cached --binary {shlex.quote(baseline)} > {out}",
        timeout=300.0,
    )
    if not result.success:
        raise RuntimeError(f"Failed to extract patch: {result.output[-2000:]}")
    cat = await executor.execute_command(f"cat {out}", timeout=120.0)
    return normalize_patch(cat.output)


async def apply_patch(executor: SandboxExecutor, patch: str) -> tuple[bool, str]:
    """Apply a patch to the testbed, trying progressively more lenient commands.

    Returns:
        Tuple of (applied, output of the last attempt).
    """
    await executor.write_files({PATCH_PATH: patch})
    output = ""
    for attempt, cmd in enumerate(GIT_APPLY_CMDS):
        if attempt:
            # A failed attempt can leave partial changes behind.
            await executor.execute_command(
                f"cd {TESTBED_DIR} && git checkout -- . ; git clean -fd", timeout=120.0
            )
        result = await executor.execute_command(
            f"cd {TESTBED_DIR} && {cmd} {PATCH_PATH}", timeout=300.0
        )
        output = result.output
        if result.success:
            return True, output
    reverse = await executor.execute_command(
        f"cd {TESTBED_DIR} && git apply --check --reverse {PATCH_PATH}", timeout=120.0
    )
    return reverse.success, output


async def grade_in_sandbox(
    executor: SandboxExecutor,
    instance: SWEBenchInstance,
    patch: str,
    timeout: float,
) -> GradeResult:
    """Apply ``patch`` in a fresh instance container, run the tests, and grade the log."""
    patch = normalize_patch(patch)
    if not patch:
        return GradeResult(empty_patch=True)

    applied, apply_output = await apply_patch(executor, patch)
    if not applied:
        return GradeResult(test_output=_tail(apply_output), error="patch_apply_failed")

    await executor.write_files({EVAL_SCRIPT_PATH: instance.eval_script})
    # The log goes to a file so the streaming executor's output cap cannot cut it.
    run = await executor.execute_command(
        f"/bin/bash {EVAL_SCRIPT_PATH} > {EVAL_LOG_PATH} 2>&1; echo eval exit code: $?",
        timeout=timeout,
        stream=True,
        log_prefix=f"swe-{instance.instance_id}-eval",
    )
    log = await executor.execute_command(f"cat {EVAL_LOG_PATH}", timeout=300.0)
    if "eval exit code:" not in run.output:
        return GradeResult(patch_applied=True, test_output=_tail(log.output), error="eval_timeout")

    test_log = log.output
    resolved, tests_ran, tests_status = grade_log(instance, patch, test_log)
    return GradeResult(
        resolved=resolved,
        patch_applied=True,
        tests_ran=tests_ran,
        tests_status=tests_status,
        test_output=_tail(test_log),
        error=None if tests_ran else "test_log_unparsed",
    )

"""SWE-Bench Pro patch capture and verification.

Grading follows the benchmark's locked protocol: the agent's changes are
captured as a diff, applied to a pristine container, and checked by the
task's unchanged verifier.
"""

from __future__ import annotations

import base64
import gzip
import logging
import re
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)

TESTS_DIR = "/tests"
REWARD_FILE = "/logs/verifier/reward.txt"
PATCH_PATH = "/tmp/swe_bench_pro_model.patch"
RUNNER_LOGS = ("/logs/verifier/run-script-stdout.txt", "/logs/verifier/run-script-stderr.txt")
RUNNER_LOG_MAX_BYTES = 2_000_000
RUNNER_LOG_TAIL_CHARS = 4000
_PATCH_START = "<<<SWE_BENCH_PRO_PATCH_START>>>"
_PATCH_END = "<<<SWE_BENCH_PRO_PATCH_END>>>"
_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


@dataclass
class VerificationResult:
    """Result of grading one patch.

    Attributes:
        resolved: Whether every required test passed.
        patch_applied: Whether the patch applied to the pristine repository.
        test_output: Combined output of the patch application and verifier.
        test_exit_code: Exit code of the verifier script, or -1 if it did not run.
        runner_logs: Test runner output keyed by file name, kept for unresolved tasks.
    """

    resolved: bool
    patch_applied: bool
    test_output: str
    test_exit_code: int
    runner_logs: dict[str, str] = field(default_factory=dict)


def snapshot_command(working_dir: str) -> str:
    """Return a shell command that prints a git tree of the repository's current files.

    The tree covers tracked and untracked files alike, so a diff against it
    leaves out files that already existed in the image. The index is restored
    afterwards.
    """
    wd = shlex.quote(working_dir)
    return (
        f"cd {wd} || exit 1; git add -A >/dev/null 2>&1; "
        "tree=$(git write-tree); rc=$?; git reset -q >/dev/null 2>&1; "
        '[ "$rc" -eq 0 ] || exit "$rc"; echo "$tree"'
    )


def parse_snapshot(output: str) -> str | None:
    """Extract the tree ID printed by :func:`snapshot_command`."""
    lines = output.strip().splitlines()
    if not lines:
        return None
    tree = lines[-1].strip()
    return tree if _OBJECT_ID_RE.fullmatch(tree) else None


def capture_patch_command(working_dir: str, base: str = "HEAD") -> str:
    """Return a shell command that prints the repository's changes as a diff.

    Changes are measured against ``base``, a commit or a tree from
    :func:`snapshot_command`. Untracked files are included, and the index is
    restored afterwards so the command has no lasting effect on the repository.
    """
    wd = shlex.quote(working_dir)
    return (
        f"set -o pipefail; cd {wd} || exit 1; git add -A >/dev/null 2>&1; "
        f"git diff --cached --binary {shlex.quote(base)} > {PATCH_PATH}; rc=$?; "
        "git reset -q >/dev/null 2>&1; "
        '[ "$rc" -eq 0 ] || exit "$rc"; '
        f"encoded=$(base64 < {PATCH_PATH} | tr -d '\\n') || exit 1; "
        f"echo '{_PATCH_START}'; echo \"$encoded\"; echo '{_PATCH_END}'"
    )


def parse_captured_patch(output: str) -> str | None:
    """Extract the diff printed by :func:`capture_patch_command`.

    Returns None when the output does not contain a complete capture.
    """
    start = output.find(_PATCH_START)
    end = output.find(_PATCH_END, start + 1)
    if start < 0 or end < 0:
        return None
    encoded = output[start + len(_PATCH_START) : end].strip()
    try:
        return base64.b64decode(encoded).decode("utf-8", errors="surrogateescape")
    except ValueError:
        return None


def apply_patch_command(working_dir: str, patch_path: str = PATCH_PATH) -> str:
    """Return a shell command that applies a patch file, falling back to fuzzier strategies."""
    wd = shlex.quote(working_dir)
    path = shlex.quote(patch_path)
    return (
        f"cd {wd} && (git apply --verbose --binary {path} "
        f"|| git apply --verbose --3way {path} "
        f"|| patch --batch --fuzz=3 -p1 -i {path})"
    )


async def write_files(
    executor: SandboxExecutor,
    files: Mapping[str, bytes],
    timeout: float = 120.0,
) -> None:
    """Write binary files to absolute paths inside the sandbox.

    Contents are staged as base64 text and decoded in the sandbox, so any byte
    sequence survives the transfer.
    """
    if not files:
        return
    staging = f"/tmp/_swe_bench_pro_{uuid.uuid4().hex[:12]}"
    staged: dict[str, str] = {}
    decode_steps = []
    for i, (path, content) in enumerate(files.items()):
        staged_path = f"{staging}/{i}.b64"
        staged[staged_path] = base64.b64encode(content).decode()
        target = shlex.quote(path)
        decode_steps.append(
            f'mkdir -p "$(dirname {target})" && base64 -d < {staged_path} > {target}'
        )

    await executor.write_files(staged, timeout=timeout)
    command = " && ".join([*decode_steps, f"rm -rf {staging}"])
    result = await executor.execute_command(command, timeout=timeout)
    if not result.success:
        raise RuntimeError(f"Failed to write files into sandbox: {result.output[-2000:]}")


async def read_runner_logs(executor: SandboxExecutor) -> dict[str, str]:
    """Return the test runner output that the task verifier leaves behind.

    Each file is truncated to its last ``RUNNER_LOG_MAX_BYTES`` bytes and
    compressed for the transfer. Missing files are skipped.
    """
    logs: dict[str, str] = {}
    for path in RUNNER_LOGS:
        quoted = shlex.quote(path)
        result = await executor.execute_command(
            f"set -o pipefail; [ -f {quoted} ] || exit 3; "
            f"tail -c {RUNNER_LOG_MAX_BYTES} {quoted} | gzip -c | base64 | tr -d '\\n'",
            timeout=60.0,
        )
        if not result.success:
            continue
        try:
            content = gzip.decompress(base64.b64decode(result.output.strip()))
        except (ValueError, OSError):
            logger.debug(f"Could not decode runner log {path}")
            continue
        logs[path.rsplit("/", 1)[-1]] = content.decode("utf-8", errors="replace")
    return logs


class SWEBenchProVerifier:
    """Applies a patch to a pristine repository and runs the task's verifier."""

    async def verify(
        self,
        executor: SandboxExecutor,
        patch: str,
        test_files: Mapping[str, bytes],
        working_dir: str,
        timeout: float,
        log_prefix: str | None = None,
    ) -> VerificationResult:
        """Grade ``patch`` against the task's hidden tests.

        Args:
            executor: Executor for a fresh sandbox of the task image.
            patch: Unified diff produced by the agent. May be empty.
            test_files: Verifier files keyed by path relative to ``/tests``.
            working_dir: Repository location inside the sandbox.
            timeout: Timeout for the verifier script in seconds.
            log_prefix: Prefix for streamed verifier logs.

        Returns:
            The verification result.
        """
        outputs: list[str] = []

        if patch.strip():
            await write_files(
                executor, {PATCH_PATH: patch.encode("utf-8", errors="surrogateescape")}
            )
            apply_result = await executor.execute_command(
                apply_patch_command(working_dir), timeout=300.0
            )
            outputs.append(f"$ apply patch (exit {apply_result.exit_code})\n{apply_result.output}")
            if not apply_result.success:
                logger.info(f"[{log_prefix}] Patch failed to apply")
                return VerificationResult(
                    resolved=False,
                    patch_applied=False,
                    test_output="\n".join(outputs),
                    test_exit_code=-1,
                )

        await write_files(
            executor, {f"{TESTS_DIR}/{path}": content for path, content in test_files.items()}
        )
        test_result = await executor.execute_command(
            f"mkdir -p /logs/verifier && cd {shlex.quote(working_dir)} && bash {TESTS_DIR}/test.sh",
            timeout=timeout,
            stream=True,
            log_prefix=log_prefix,
        )
        outputs.append(f"$ bash {TESTS_DIR}/test.sh (exit {test_result.exit_code})")
        outputs.append(test_result.output)

        reward_result = await executor.execute_command(f"cat {REWARD_FILE}", timeout=30.0)
        resolved = reward_result.success and reward_result.output.strip() == "1"
        logger.info(f"[{log_prefix}] Verifier exit={test_result.exit_code} resolved={resolved}")

        runner_logs: dict[str, str] = {}
        if not resolved:
            runner_logs = await read_runner_logs(executor)
            for name, content in runner_logs.items():
                outputs.append(f"=== {name} (tail) ===\n{content[-RUNNER_LOG_TAIL_CHARS:]}")

        return VerificationResult(
            resolved=resolved,
            patch_applied=True,
            test_output="\n".join(outputs),
            test_exit_code=test_result.exit_code,
            runner_logs=runner_logs,
        )

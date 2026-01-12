"""Code execution utilities for pass@k evaluation."""

import contextlib
import io
import math
import multiprocessing
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .types import Instance, LMOutput, Response


def _execute_code_unsafe(code: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Execute Python code and return (success, error_message).

    WARNING: This executes arbitrary code. Use with caution and only in
    sandboxed environments.
    """

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Code execution timed out")

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout))

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, {"__builtins__": __builtins__}, {})

        signal.alarm(0)  # Cancel alarm
        return True, ""

    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def _worker(args: tuple[str, str, float]) -> tuple[str, bool, str]:
    """Worker function for multiprocessing code execution."""
    task_id, code, timeout = args
    success, error = _execute_code_unsafe(code, timeout)
    return task_id, success, error


def execute_code_batch(
    code_samples: list[tuple[str, str]],  # List of (task_id, code)
    timeout: float = 5.0,
    num_workers: int = 4,
) -> dict[str, tuple[bool, str]]:
    """Execute multiple code samples in parallel.

    Args:
        code_samples: List of (task_id, code) tuples
        timeout: Timeout per sample in seconds
        num_workers: Number of parallel workers

    Returns:
        Dict mapping task_id to (success, error_message)
    """
    args = [(task_id, code, timeout) for task_id, code in code_samples]

    results = {}
    with multiprocessing.Pool(num_workers) as pool:
        for task_id, success, error in pool.map(_worker, args):
            results[task_id] = (success, error)

    return results


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0

    # Use log to avoid overflow for large n
    # pass@k = 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@dataclass(frozen=True, slots=True)
class CodeExecutionScorer:
    """Score code by executing it against test cases.

    Note: This scorer requires the instance metadata to contain a 'test' key
    with the test code to run, and the output.extracted_answer to contain
    the complete code to execute.
    """

    name: str = "code_execution"
    timeout: float = 5.0

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score by executing code + tests."""
        if output.extracted_answer is None:
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0

        # Combine generated code with tests
        full_code = f"{output.extracted_answer}\n\n{test_code}"

        success, _ = _execute_code_unsafe(full_code, self.timeout)
        return 1.0 if success else 0.0


@dataclass(frozen=True, slots=True)
class PassAtKMetric:
    """Compute pass@k metric for code generation tasks.

    This metric groups responses by task ID and computes pass@k
    across multiple samples per task.
    """

    name: str = "pass_at_k"
    k: int = 1
    scorer_name: str = "code_execution"

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute pass@k across all tasks."""
        if not responses:
            return 0.0

        # Group by task ID
        task_results: dict[str, list[float]] = {}
        for r in responses:
            task_id = r.instance.metadata.get("id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(r.scores.get(self.scorer_name, 0.0))

        # Compute pass@k for each task
        pass_at_k_values = []
        for scores in task_results.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)  # Count passing
            pass_at_k_values.append(compute_pass_at_k(n, c, min(self.k, n)))

        return sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

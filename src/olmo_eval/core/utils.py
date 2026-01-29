"""Utility functions for evaluation."""

import contextlib
import io
import math
import signal
from typing import Any


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

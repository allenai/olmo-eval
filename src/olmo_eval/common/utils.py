"""Utility functions for evaluation."""

from __future__ import annotations

import contextlib
import io
import math
import signal
from dataclasses import fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Serializable:
    """Mixin that provides to_dict() for dataclasses, excluding None values."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if value is None:
                continue
            result[f.name] = _serialize_value(value)
        return result


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "value"):  # Enum
        return value.value
    return value


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
    """Compute pass@k metric (unbiased estimator).

    Pass@k measures the probability that at least one of k samples
    is correct. When n < k, uses k = min(k, n) per standard convention.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        pass@k probability
    """
    if n == 0:
        return 0.0
    # Clamp k to available samples (standard convention when n < k)
    k = min(k, n)
    if n - c < k:
        return 1.0
    # Use math.prod to avoid overflow for large n
    # pass@k = 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    """Compute pass^k metric (all k trials succeed).

    Pass^k measures automation readiness: the probability that
    k consecutive runs all succeed. Computed as (success_rate)^k.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass^k

    Returns:
        pass^k probability (success_rate ** k)
    """
    if n == 0:
        return 0.0
    success_rate = c / n
    return success_rate**k

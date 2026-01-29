"""Code execution utilities for batch evaluation."""

import multiprocessing

from olmo_eval.core.utils import _execute_code_unsafe


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

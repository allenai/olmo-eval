"""Common utility functions for ID generation, sanitization, and metadata extraction."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from olmo_eval.core.types import SamplingParams


def generate_experiment_id() -> str:
    """Generate a unique experiment ID.

    Returns a short 12-character hex string from UUID4. Since experiments
    are partitioned by model/task, collision risk is minimal.

    Returns:
        A 12-character hex string to uniquely identify an experiment.

    Example:
        >>> exp_id = generate_experiment_id()
        >>> len(exp_id)
        12
    """
    return uuid.uuid4().hex[:12]


def get_author() -> str:
    """Get the current user for experiment attribution.

    Checks environment variables in order:
    1. BEAKER_AUTHOR - set by olmo-eval beaker launch for Beaker jobs
    2. USER, USERNAME, LOGNAME - standard Unix user env vars
    3. Falls back to getpass.getuser() if no env var is set.

    Returns:
        Username string.
    """
    import getpass

    return (
        os.environ.get("BEAKER_AUTHOR")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or os.environ.get("LOGNAME")
        or getpass.getuser()
    )


def get_git_ref() -> str:
    """Get the current git commit hash.

    Returns:
        Git commit hash (short form) or "unknown" if not in a git repo.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def sanitize_spec_for_filename(spec: str) -> str:
    """Sanitize a task spec string to be safe for use in filenames.

    Replaces characters that are not allowed or problematic in filenames:
    - ':' -> '_' (used in task variants like arc_challenge:bpb)
    - '/' -> '_' (could appear in model names or paths)
    - '\\' -> '_' (Windows path separator)
    - ' ' -> '_' (spaces)

    Args:
        spec: Task specification string (e.g., "arc_challenge:bpb::olmes")

    Returns:
        Sanitized string safe for use in filenames (e.g., "arc_challenge_bpb__olmes")

    Examples:
        >>> sanitize_spec_for_filename("arc_challenge:bpb")
        'arc_challenge_bpb'
        >>> sanitize_spec_for_filename("mmlu/history")
        'mmlu_history'
        >>> sanitize_spec_for_filename("task::key=value")
        'task__key=value'
    """
    # Replace problematic characters with underscores
    result = spec.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    return result


def serialize_sampling_params(params: SamplingParams | None) -> dict[str, Any] | None:
    """Serialize SamplingParams to a dictionary for JSON output.

    Args:
        params: SamplingParams instance or None

    Returns:
        Dictionary representation or None if params is None
    """
    if params is None:
        return None
    return {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "num_samples": params.num_samples,
    }


def compute_task_hash(config: dict) -> str:
    """Compute a hash from task config.

    Args:
        config: Task configuration dictionary

    Returns:
        16-character hex string hash of the config
    """
    import hashlib

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_primary_metric(
    metrics: dict[str, float],
    preferred: str | None = None,
) -> tuple[str, float] | None:
    """Get the primary metric name and value from a metrics dict.

    Priority:
    1. User-specified preferred metric (if provided and present)
    2. "accuracy" if present (most common metric)
    3. First metric alphabetically (for determinism)

    Args:
        metrics: Dictionary of metric names to values
        preferred: Optional preferred metric name (from task config)

    Returns:
        Tuple of (metric_name, metric_value), or None if metrics is empty
    """
    if not metrics:
        return None

    # Use preferred metric if specified and present
    if preferred and preferred in metrics:
        return (preferred, metrics[preferred])

    # Default fallback: accuracy first
    if "accuracy" in metrics:
        return ("accuracy", metrics["accuracy"])

    # Fallback: first metric alphabetically for determinism
    name = sorted(metrics.keys())[0]
    return (name, metrics[name])


def get_metric_metadata(task: Any) -> tuple[str | None, dict[str, str] | None]:
    """Extract metric metadata from task config.

    Args:
        task: Task instance with config.metrics

    Returns:
        Tuple of (primary_metric_name, metric_scorers).
        - primary_metric_name: Name of the primary metric, or None
        - metric_scorers: Dict mapping metric name to scorer name, or None if empty
    """
    # Get primary metric name
    primary_metric = task.config.get_primary_metric()
    primary_metric_name = primary_metric.name if primary_metric else None

    # Get metric-to-scorer mapping
    metric_scorers: dict[str, str] = {}
    if hasattr(task.config, "metrics"):
        for metric in task.config.metrics:
            if hasattr(metric, "scorer") and metric.scorer is not None:
                scorer_instance = metric.scorer()
                metric_scorers[metric.name] = scorer_instance.name

    return primary_metric_name, metric_scorers if metric_scorers else None

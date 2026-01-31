"""Model name formatting and S3 path building utilities."""

from __future__ import annotations


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in S3 paths.

    For paths like /weka/.../model_dir/step61007-hf/, extracts last 2 components
    and joins with underscore: model_dir_step61007-hf

    For HuggingFace-style names like meta-llama/Llama-3.1-8B, replaces / with _.

    Args:
        model_name: Model name or path.

    Returns:
        Sanitized model name safe for S3 paths.
    """
    # Strip trailing slashes
    model_name = model_name.rstrip("/")

    # Check if it looks like an absolute path (starts with / or contains /weka, /data, etc.)
    if model_name.startswith("/") or "/weka/" in model_name or "/data/" in model_name:
        # It's a filesystem path - take last 2 components
        parts = [p for p in model_name.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[-2]}_{parts[-1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return "unknown"

    # For HuggingFace-style names (org/model), just replace / with _
    return model_name.replace("/", "_")


def get_model_display_name(model_path: str, alias: str | None = None) -> str:
    """Get the display name for a model.

    Uses alias if provided, otherwise sanitizes the model path.
    This is the standard way to get a model name for file paths,
    S3 prefixes, and display purposes.

    Args:
        model_path: Model name or path (e.g., "meta-llama/Llama-3.1-8B" or "/weka/.../step1000-hf")
        alias: Optional alias override

    Returns:
        Display name: alias if provided, else sanitized model path
    """
    if alias:
        return alias
    return sanitize_model_name(model_path)


def build_s3_prefix(
    base_prefix: str,
    group: str,
    model_name: str,
    model_hash: str | None,
    experiment_id: str,
) -> str:
    """Build the S3 prefix for an experiment.

    Path structure: {prefix}/{group}/{model_name}_{hash_last_6}/{experiment_id}

    Args:
        base_prefix: Base prefix, e.g., "olmo-eval".
        group: Experiment group, e.g., "baseline", "ablation-lr".
        model_name: Model name or path (will be sanitized).
        model_hash: Model configuration hash.
        experiment_id: Unique experiment identifier.

    Returns:
        S3 prefix string (without bucket or s3:// prefix).
    """
    sanitized_model = sanitize_model_name(model_name)
    hash_suffix = model_hash[-6:] if model_hash else "000000"
    return "/".join(
        [
            base_prefix.rstrip("/"),
            group,
            f"{sanitized_model}_{hash_suffix}",
            experiment_id,
        ]
    )

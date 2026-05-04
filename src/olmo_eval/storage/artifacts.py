"""Helpers for task-level artifact URIs stored alongside results."""

from __future__ import annotations

from urllib.parse import urlparse

from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.processing.utils import sanitize_spec_for_filename


def build_predictions_uri(
    base: str,
    model_name: str,
    task_name: str,
    task_hash: str | None = None,
) -> str:
    """Build the canonical S3 URI for a task predictions artifact."""
    return _build_task_artifact_uri(
        base=base,
        artifact_dir="predictions",
        model_name=model_name,
        task_name=task_name,
        task_hash=task_hash,
        suffix=PREDICTIONS_SUFFIX,
    )


def build_requests_uri(
    base: str,
    model_name: str,
    task_name: str,
    task_hash: str | None = None,
) -> str:
    """Build the canonical S3 URI for a task requests artifact."""
    return _build_task_artifact_uri(
        base=base,
        artifact_dir="requests",
        model_name=model_name,
        task_name=task_name,
        task_hash=task_hash,
        suffix=REQUESTS_SUFFIX,
    )


def candidate_predictions_uris(
    uri: str,
    *,
    model_name: str | None = None,
    task_name: str | None = None,
    task_hash: str | None = None,
) -> tuple[str, ...]:
    """Return possible S3 URIs for a task predictions artifact.

    Older DB rows synthesized several incorrect path shapes. This helper returns
    the stored URI first, then any plausible corrected URIs based on the
    experiment prefix and, when available, the task/model metadata.
    """
    return _candidate_task_artifact_uris(
        uri,
        artifact_dir="predictions",
        suffix=PREDICTIONS_SUFFIX,
        model_name=model_name,
        task_name=task_name,
        task_hash=task_hash,
    )


def candidate_requests_uris(
    uri: str,
    *,
    model_name: str | None = None,
    task_name: str | None = None,
    task_hash: str | None = None,
) -> tuple[str, ...]:
    """Return possible S3 URIs for a task requests artifact."""
    return _candidate_task_artifact_uris(
        uri,
        artifact_dir="requests",
        suffix=REQUESTS_SUFFIX,
        model_name=model_name,
        task_name=task_name,
        task_hash=task_hash,
    )


def _candidate_task_artifact_uris(
    uri: str,
    *,
    artifact_dir: str,
    suffix: str,
    model_name: str | None = None,
    task_name: str | None = None,
    task_hash: str | None = None,
) -> tuple[str, ...]:
    if not uri.startswith("s3://") or not uri.endswith(suffix):
        return (uri,)

    candidates: list[str] = []
    _append_unique(candidates, uri)

    parsed = urlparse(uri)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    if not path:
        return tuple(candidates)

    parts = path.split("/")
    filename = parts[-1]
    if len(parts) < 2 or parts[-2] != artifact_dir:
        parent = "/".join(parts[:-1])
        corrected_path = (
            f"{parent}/{artifact_dir}/{filename}" if parent else f"{artifact_dir}/{filename}"
        )
        _append_unique(candidates, f"s3://{bucket}/{corrected_path}")

    if model_name and task_name:
        base = _extract_experiment_base(uri, artifact_dir=artifact_dir)
        _append_unique(
            candidates,
            _build_task_artifact_uri(
                base=base,
                artifact_dir=artifact_dir,
                model_name=model_name,
                task_name=task_name,
                task_hash=task_hash,
                suffix=suffix,
            ),
        )

    return tuple(candidates)


def _build_task_artifact_uri(
    *,
    base: str,
    artifact_dir: str,
    model_name: str,
    task_name: str,
    task_hash: str | None,
    suffix: str,
) -> str:
    sanitized_model_name = sanitize_spec_for_filename(model_name)
    sanitized_task_name = sanitize_spec_for_filename(task_name)
    hash_suffix = f"_{task_hash[-6:]}" if task_hash else ""
    return (
        f"{base}/{artifact_dir}/{sanitized_model_name}/{sanitized_task_name}{hash_suffix}{suffix}"
    )


def _extract_experiment_base(uri: str, *, artifact_dir: str) -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    if not path:
        return uri.rstrip("/")

    parts = path.split("/")
    artifact_idx = next(
        (idx for idx in range(len(parts) - 1, -1, -1) if parts[idx] == artifact_dir),
        None,
    )
    base_parts = parts[:artifact_idx] if artifact_idx is not None else parts[:-1]

    base_path = "/".join(base_parts)
    return f"s3://{bucket}/{base_path}".rstrip("/")


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)

"""S3 upload and storage backend save utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.cli.utils import console
from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.models import S3Config
from olmo_eval.runners.io.formatting import build_s3_prefix
from olmo_eval.runners.processing.utils import (
    generate_experiment_id,
    get_author,
    get_git_ref,
    get_workspace,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

logger = get_logger("runners.storage")


def upload_to_s3(
    output_dir: str,
    s3_config: S3Config,
    model_name: str,
    model_hash: str,
    experiment_id: str,
    max_workers: int = 16,
) -> str | None:
    """Upload evaluation output to S3.

    Uploads metrics.json, predictions/, and requests/ directories to S3
    using concurrent uploads for better performance.

    Path structure:
    s3://{bucket}/{prefix}/{group}/{model_name}_{hash_last_6}/{experiment_id}/

    Args:
        output_dir: Local output directory containing files to upload.
        s3_config: S3 configuration with bucket, prefix, group, etc.
        model_name: Model name or path.
        model_hash: Model configuration hash.
        experiment_id: Unique experiment identifier.
        max_workers: Maximum concurrent upload threads (default 16).

    Returns:
        S3 base URI if uploaded, None if upload failed.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.warning("boto3 not installed; skipping S3 upload.")
        return None

    # Log S3 configuration
    logger.info(
        f"Uploading to S3: bucket={s3_config.bucket}, prefix={s3_config.prefix}, "
        f"region={s3_config.region}, group={s3_config.group}"
    )
    if s3_config.endpoint_url:
        logger.info(f"  Using custom endpoint: {s3_config.endpoint_url}")

    # Build S3 prefix:
    # {prefix}/{group}/{sanitized_model_name}_{hash_last_6}/{experiment_id}
    prefix = build_s3_prefix(
        base_prefix=s3_config.prefix,
        group=s3_config.group,
        model_name=model_name,
        model_hash=model_hash,
        experiment_id=experiment_id,
    )

    retry_config = Config(
        retries={
            "max_attempts": 5,
            "mode": "adaptive",
        }
    )

    client_kwargs: dict[str, Any] = {"region_name": s3_config.region, "config": retry_config}
    if s3_config.endpoint_url:
        client_kwargs["endpoint_url"] = s3_config.endpoint_url
    s3 = boto3.client("s3", **client_kwargs)

    output_path = Path(output_dir)

    # Collect all files to upload
    files_to_upload = [p for p in output_path.rglob("*") if p.is_file()]

    if not files_to_upload:
        logger.warning(f"No files found in {output_dir}")
        return None

    def upload_single_file(local_path: Path) -> tuple[Path, bool, str | None]:
        """Upload a single file. Returns (path, success, error_message)."""
        relative = local_path.relative_to(output_path)
        key = f"{prefix}/{relative}"

        # Auto-detect content type
        if local_path.suffix == ".json":
            content_type = "application/json"
        elif local_path.suffix == ".jsonl":
            content_type = "application/x-ndjson"
        else:
            content_type = "application/octet-stream"

        try:
            s3.upload_file(
                str(local_path),
                s3_config.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            return (local_path, True, None)
        except Exception as e:
            return (local_path, False, str(e))

    # Upload files concurrently
    uploaded_count = 0
    failed_count = 0
    effective_workers = min(max_workers, len(files_to_upload))

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(upload_single_file, f): f for f in files_to_upload}

        for future in as_completed(futures):
            local_path, success, error = future.result()
            if success:
                uploaded_count += 1
            else:
                failed_count += 1
                relative = local_path.relative_to(output_path)
                logger.error(f"Failed to upload {relative}: {error}")
                console.print(f"[red]Failed to upload {relative}:[/red] {error}")

    s3_location = f"s3://{s3_config.bucket}/{prefix}"
    if failed_count > 0:
        logger.warning(
            f"S3 upload completed with errors: {uploaded_count} succeeded, {failed_count} failed"
        )
        console.print(
            f"[yellow]S3 upload:[/yellow] {uploaded_count} uploaded, "
            f"{failed_count} failed -> {s3_location}"
        )
    else:
        logger.info(f"Uploaded {uploaded_count} files to S3: {s3_location}")

    return s3_location if uploaded_count > 0 else None


def save_results(
    results: dict[str, Any],
    storages: list[StorageBackend],
    s3_config: S3Config | None = None,
    experiment_id: str | None = None,
    model_hash: str | None = None,
    s3_location: str | None = None,
    experiment_name: str | None = None,
    experiment_group: str | None = None,
    experiment_duration_seconds: float | None = None,
    provider_init_seconds: dict[str, float] | None = None,
) -> None:
    """Save results to all configured storage backends.

    Handles both single-model results (with 'model' key) and multi-model
    results (with 'models' dict). For multi-model results, saves each
    model's results separately.

    Args:
        results: The results dict from the runner.
        storages: List of storage backends to save to.
        s3_config: Optional S3 configuration for group/workspace info.
        experiment_id: Pre-generated experiment ID (for single-model only).
        model_hash: Model configuration hash (for single-model only).
        s3_location: S3 location where results were uploaded (for single-model only).
        experiment_name: Human-readable experiment name.
        experiment_group: Group for related experiments.
        experiment_duration_seconds: Total time for the experiment.
        provider_init_seconds: Dict mapping model name to provider init time.
    """
    if not storages:
        logger.info("No storage backend configured; skipping results save.")
        return

    from olmo_eval.storage.base import convert_runner_results

    # Determine if this is multi-model or single-model results
    if "models" in results:
        # Multi-model async results - save each model separately
        # For multi-model, we ignore the passed experiment_id/model_hash/s3_location
        # as each model needs its own values
        models_to_save: list[tuple[str, dict[str, Any], str, str | None, str | None]] = []
        for model_name, model_data in results["models"].items():
            # Build single-model results dict from multi-model structure
            single_model_results = {
                "model": model_data.get("model", model_name),
                "model_path": model_data.get("model_path"),  # Original full path
                "provider": model_data.get("provider", "unknown"),
                "timestamp": results.get("timestamp"),
                "tasks": model_data.get("tasks", {}),
                "suites": model_data.get("suites"),
                "model_config": model_data.get("model_config"),
            }
            # For multi-model, get per-model values from model_data if available
            m_experiment_id = model_data.get("_experiment_id") or generate_experiment_id()
            m_model_hash = model_data.get("_model_hash")
            m_s3_location = model_data.get("_s3_location")
            models_to_save.append(
                (model_name, single_model_results, m_experiment_id, m_model_hash, m_s3_location)
            )
        logger.info(f"Saving results for {len(models_to_save)} model(s) to storage")
    else:
        # Single-model results - use passed values or generate
        exp_id = experiment_id or generate_experiment_id()
        models_to_save = [
            (results.get("model", "unknown"), results, exp_id, model_hash, s3_location)
        ]
        logger.info(f"Saving results for model '{results.get('model')}' to storage")

    author = get_author()
    git_ref = get_git_ref()
    workspace = get_workspace()

    for model_name, model_results, exp_id, m_hash, s3_loc in models_to_save:
        task_count = len(model_results.get("tasks", {}))
        logger.info(
            f"Converting results: model={model_name}, tasks={task_count}, experiment_id={exp_id}"
        )

        model_cfg = model_results.get("model_config", {})
        revision = model_cfg.get("revision") or "unknown"
        if not m_hash:
            from olmo_eval.common.types import compute_model_hash

            m_hash = (compute_model_hash(model_cfg) if model_cfg else None) or "unknown"

        try:
            # experiment_group must always have a value - never empty
            effective_experiment_name = experiment_name or exp_id
            effective_experiment_group = experiment_group or effective_experiment_name

            eval_result = convert_runner_results(
                model_results,
                exp_id,
                s3_location=s3_loc,
                experiment_name=effective_experiment_name,
                workspace=workspace,
                author=author,
                git_ref=git_ref,
                model_hash=m_hash,
                revision=revision,
                model_path=model_results.get("model_path"),
                experiment_group=effective_experiment_group,
                experiment_duration_seconds=experiment_duration_seconds,
                provider_init_seconds=provider_init_seconds,
            )
            logger.info(f"Converted results for {model_name}, saving to {len(storages)} backend(s)")
        except Exception as e:
            logger.error(f"Failed to convert results for {model_name}: {e}")
            console.print(f"[red]Failed to convert results for {model_name}: {e}[/red]")
            continue

        # Build instances_by_task from predictions in model_results
        instances_by_task: dict[str, list[dict[str, Any]]] = {}
        for task_name, task_data in model_results.get("tasks", {}).items():
            predictions = task_data.get("predictions")
            if predictions:
                instances_by_task[task_name] = predictions

        for storage in storages:
            backend_name = type(storage).__name__
            logger.info(f"Saving to {backend_name}...")
            try:
                storage.save(eval_result, instances_by_task if instances_by_task else None)
                logger.info(
                    f"Saved to {backend_name}: model={model_name}, "
                    f"experiment_id={exp_id}, tasks={task_count}"
                )
                instance_count = sum(len(preds) for preds in instances_by_task.values())
                console.print(
                    f"[green]Saved to {backend_name}:[/green] {model_name} "
                    f"({task_count} tasks, {instance_count} instances, id={exp_id})"
                )
            except Exception as e:
                logger.error(f"Failed to save to {backend_name}: {e}")
                console.print(f"[red]Failed to save to {backend_name}: {e}[/red]")

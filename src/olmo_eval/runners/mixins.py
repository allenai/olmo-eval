"""Shared mixin classes for evaluation runners."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.runners.utils import (
    TaskResult,
    generate_experiment_id,
    get_author,
    get_git_ref,
    get_primary_metric,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class S3Config:
    """Configuration for S3 uploads.

    The S3 path structure is:
    s3://{bucket}/{prefix}/{group}/{model_name}_{model_hash_last_6}/{experiment_id}/
        - metrics.json
        - predictions/{task}-predictions.jsonl
        - requests/{task}-requests.jsonl
    """

    bucket: str
    prefix: str  # Base prefix, e.g., "olmo-eval"
    group: str  # Experiment group, e.g., "baseline", "ablation-lr"
    endpoint_url: str | None = None
    region: str = "us-east-1"


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


# -----------------------------------------------------------------------------
# Dataclasses for metrics.json output
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model: str
    backend: str
    dtype: str = "auto"
    tokenizer: str | None = None
    revision: str | None = None
    attention_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {
            "model": self.model,
            "backend": self.backend,
            "dtype": self.dtype,
        }
        if self.tokenizer:
            result["tokenizer"] = self.tokenizer
        if self.revision:
            result["revision"] = self.revision
        if self.attention_backend:
            result["attention_backend"] = self.attention_backend
        return result


@dataclass
class TaskMetricsEntry:
    """A task entry in the metrics output."""

    task: str
    metrics: dict[str, float]
    num_instances: int
    model: str | None = None  # Only set for multi-model format
    primary_metric: str | None = None
    config: dict[str, Any] | None = None
    duration_seconds: float | None = None
    task_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {
            "task": self.task,
            "metrics": self.metrics,
            "num_instances": self.num_instances,
        }
        if self.model is not None:
            result["model"] = self.model
        if self.primary_metric is not None:
            result["primary_metric"] = self.primary_metric
        if self.config is not None:
            result["config"] = self.config
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.task_hash is not None:
            result["task_hash"] = self.task_hash
        return result


@dataclass
class ScoreSummary:
    """Summary entry with metric name and score."""

    metric: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {"metric": self.metric, "score": self.score}


@dataclass
class MetricsOutput:
    """Top-level metrics.json output structure."""

    timestamp: str
    config: dict[str, Any]  # ModelConfig.to_dict() or {"models": {name: config}}
    tasks: list[dict[str, Any]]  # List of TaskMetricsEntry.to_dict()
    summary: dict[str, Any]  # task_name -> ScoreSummary or model -> task -> ScoreSummary
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Experiment identification fields for querying results
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, excluding None values."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "config": self.config,
            "tasks": self.tasks,
            "summary": self.summary,
            "errors": self.errors,
        }
        if self.experiment_id is not None:
            result["experiment_id"] = self.experiment_id
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name
        if self.experiment_group is not None:
            result["experiment_group"] = self.experiment_group
        return result


class RunnerResultsMixin:
    """Shared results-writing functionality for runners."""

    output_dir: str
    storages: list[StorageBackend]
    task_specs: list[str]

    # Optional S3 upload configuration
    s3_config: S3Config | None

    def _validate_task_specs(self) -> list[str]:
        """Validate task specs and return list of errors.

        Returns:
            List of error messages (empty if all specs are valid).
        """
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
        from olmo_eval.evals.tasks.core.registry import parse_task_spec

        errors: list[str] = []
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()
        variants_by_task = list_variants()

        for spec in self.task_specs:
            if suite_exists(spec):
                continue

            # Parse task_name[:variant1[:variant2...]][::key=value,...] format
            task_name, variants, _overrides = parse_task_spec(spec)

            if task_name not in available_tasks:
                errors.append(f"Unknown task or suite: '{spec}'")
                continue

            # Validate each variant/regime exists (check both registries)
            task_variants = set(variants_by_task.get(task_name, []))
            task_regimes = set(regimes_by_task.get(task_name, []))
            all_valid_variants = task_variants | task_regimes

            for variant in variants:
                if variant not in all_valid_variants:
                    available_list = sorted(all_valid_variants)
                    if available_list:
                        errors.append(
                            f"Unknown variant/regime '{variant}' for task '{task_name}'. "
                            f"Available: {', '.join(available_list)}"
                        )
                    else:
                        errors.append(
                            f"Unknown variant/regime '{variant}' for task '{task_name}'. "
                            f"This task has no registered variants or regimes."
                        )

        return errors

    def _save_results(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        model_hash: str | None = None,
        s3_location: str | None = None,
    ) -> None:
        """Save results to all configured storage backends.

        Handles both single-model results (with 'model' key) and multi-model
        results (with 'models' dict). For multi-model results, saves each
        model's results separately.

        Args:
            results: The results dict from the runner.
            experiment_id: Pre-generated experiment ID (for single-model only).
            model_hash: Model configuration hash (for single-model only).
            s3_location: S3 location where results were uploaded (for single-model only).
        """
        if not self.storages:
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
                    "backend": model_data.get("backend", "unknown"),
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
        s3_cfg = getattr(self, "s3_config", None)
        workspace = s3_cfg.group if s3_cfg else "default"
        runner_experiment_name = getattr(self, "experiment_name", None)
        runner_experiment_group = getattr(self, "experiment_group", None)

        for model_name, model_results, exp_id, m_hash, s3_loc in models_to_save:
            task_count = len(model_results.get("tasks", {}))
            logger.info(
                f"Converting results: model={model_name}, tasks={task_count}, "
                f"experiment_id={exp_id}"
            )

            model_cfg = model_results.get("model_config", {})
            revision = model_cfg.get("revision") or "unknown"
            if not m_hash:
                from olmo_eval.core.types import compute_model_hash

                m_hash = (compute_model_hash(model_cfg) if model_cfg else None) or "unknown"

            try:
                # experiment_group must always have a value - never empty
                effective_experiment_name = runner_experiment_name or exp_id
                effective_experiment_group = runner_experiment_group or effective_experiment_name

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
                )
                logger.info(
                    f"Converted results for {model_name}, saving to {len(self.storages)} backend(s)"
                )
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

            for storage in self.storages:
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

    def _upload_to_s3(
        self,
        model_name: str,
        model_hash: str,
        experiment_id: str,
    ) -> str | None:
        """Upload evaluation output to S3.

        Uploads metrics.json, predictions/, and requests/ directories to S3.
        Requires s3_config to be set.

        Path structure:
        s3://{bucket}/{prefix}/{group}/{model_name}_{hash_last_6}/{experiment_id}/

        Args:
            model_name: Model name or path.
            model_hash: Model configuration hash.
            experiment_id: Unique experiment identifier.

        Returns:
            S3 base URI if uploaded, None if S3 not configured.
        """
        s3_config = getattr(self, "s3_config", None)
        if not s3_config:
            logger.debug("S3 upload not configured; skipping.")
            return None

        try:
            import boto3
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

        # Create S3 client
        client_kwargs: dict[str, Any] = {"region_name": s3_config.region}
        if s3_config.endpoint_url:
            client_kwargs["endpoint_url"] = s3_config.endpoint_url
        s3 = boto3.client("s3", **client_kwargs)

        output_path = Path(self.output_dir)
        uploaded_count = 0
        failed_count = 0

        # Upload all files in output directory
        for local_path in output_path.rglob("*"):
            if local_path.is_file():
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
                    uploaded_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error(
                        f"Failed to upload {relative} to s3://{s3_config.bucket}/{key}: {e}"
                    )
                    console.print(f"[red]Failed to upload {relative}:[/red] {e}")

        s3_location = f"s3://{s3_config.bucket}/{prefix}"
        if failed_count > 0:
            logger.warning(
                f"S3 upload completed with errors: "
                f"{uploaded_count} succeeded, {failed_count} failed"
            )
            console.print(
                f"[yellow]S3 upload:[/yellow] {uploaded_count} uploaded, "
                f"{failed_count} failed -> {s3_location}"
            )
        else:
            logger.info(f"Uploaded {uploaded_count} files to S3: {s3_location}")

        return s3_location if uploaded_count > 0 else None

    def _log_summary(self, results: dict[str, Any], multi_model: bool = False) -> None:
        """Log summary of all task scores.

        Args:
            results: Results dictionary
            multi_model: If True, iterate results["models"][model]["tasks"],
                        otherwise iterate results["tasks"] directly
        """
        logger.info("Summary of primary scores:")

        if multi_model:
            for model_name, model_data in results.get("models", {}).items():
                logger.info(f"  {model_name}:")
                for task_name, task_data in model_data.get("tasks", {}).items():
                    metrics = task_data.get("metrics", {})
                    preferred = task_data.get("primary_metric")
                    primary = get_primary_metric(metrics, preferred)
                    if primary:
                        metric_name, score = primary
                        logger.info(f"    {task_name}: {score:.4f} ({metric_name})")

                for suite_name, suite_data in model_data.get("suites", {}).items():
                    metrics = suite_data.get("metrics", {})
                    primary = get_primary_metric(metrics)
                    if primary:
                        metric_name, score = primary
                        logger.info(f"    {suite_name}: {score:.4f} ({metric_name})")
        else:
            for task_name, task_data in results["tasks"].items():
                metrics = task_data.get("metrics", {})
                preferred = task_data.get("primary_metric")
                primary = get_primary_metric(metrics, preferred)
                if primary:
                    metric_name, score = primary
                    logger.info(f"  {task_name}: {score:.4f} ({metric_name})")

            for suite_name, suite_data in results.get("suites", {}).items():
                metrics = suite_data.get("metrics", {})
                primary = get_primary_metric(metrics)
                if primary:
                    metric_name, score = primary
                    logger.info(f"  {suite_name}: {score:.4f} ({metric_name})")

    def _write_metrics_json(
        self,
        results: dict[str, Any],
        multi_model: bool = False,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        model_hash: str | None = None,
    ) -> None:
        """Write metrics.json

        Args:
            results: Results dictionary
            multi_model: If True, use multi-model format with results["models"],
                        otherwise use single-model format with results["tasks"]
            experiment_id: Unique ID for this experiment run
            experiment_name: Human-readable experiment name
            experiment_group: Group for related experiments
            model_hash: Hash of model config (single-model only)
        """
        metrics_file = os.path.join(self.output_dir, "metrics.json")

        if multi_model:
            metrics_output = self._build_multi_model_metrics(
                results,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                experiment_group=experiment_group,
            )
        else:
            metrics_output = self._build_single_model_metrics(
                results,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                experiment_group=experiment_group,
                model_hash=model_hash,
            )

        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_output.to_dict(), f, indent=2)

        logger.info(f"Metrics written to {metrics_file}")
        console.print(f"[green]Metrics written to {metrics_file}[/green]")

    def _build_single_model_metrics(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        model_hash: str | None = None,
    ) -> MetricsOutput:
        """Build metrics output for single-model format."""
        # Build config from stored model config
        model_cfg = results.get("model_config", {})
        config = ModelConfig(
            model=model_cfg.get("model", results.get("model", "")),
            backend=model_cfg.get("backend", results.get("backend", "")),
            dtype=model_cfg.get("dtype", "auto"),
            tokenizer=model_cfg.get("tokenizer"),
            revision=model_cfg.get("revision"),
            attention_backend=model_cfg.get("attention_backend"),
        )

        # Build config dict with model_hash included
        config_dict = config.to_dict()
        if model_hash is not None:
            config_dict["model_hash"] = model_hash

        # Build task entries
        tasks_list: list[TaskMetricsEntry] = []
        for task_name, task_data in results.get("tasks", {}).items():
            entry = TaskMetricsEntry(
                task=task_name,
                metrics=task_data.get("metrics", {}),
                num_instances=task_data.get("num_instances", 0),
                primary_metric=task_data.get("primary_metric"),
                config=task_data.get("config"),
                duration_seconds=task_data.get("duration_seconds"),
                task_hash=task_data.get("task_hash"),
            )
            tasks_list.append(entry)

        # Build summary with primary metric for each task
        summary: dict[str, ScoreSummary] = {}
        for task_name, task_data in results.get("tasks", {}).items():
            metrics = task_data.get("metrics", {})
            preferred = task_data.get("primary_metric")
            primary = get_primary_metric(metrics, preferred)
            if primary:
                metric_name, score = primary
                summary[task_name] = ScoreSummary(metric=metric_name, score=score)

        # Add suite summaries
        if "suites" in results:
            for suite_name, suite_data in results["suites"].items():
                metrics = suite_data.get("metrics", {})
                primary = get_primary_metric(metrics)
                if primary:
                    metric_name, score = primary
                    summary[suite_name] = ScoreSummary(metric=metric_name, score=score)

        return MetricsOutput(
            timestamp=results.get("timestamp", ""),
            config=config_dict,
            tasks=[t.to_dict() for t in tasks_list],
            summary={k: v.to_dict() for k, v in summary.items()},
            errors=results.get("errors", []),
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
        )

    def _build_multi_model_metrics(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
    ) -> MetricsOutput:
        """Build metrics output for multi-model format."""
        # Build config for each model (include model_hash per model)
        models_config: dict[str, dict[str, Any]] = {}
        for model_name, model_data in results.get("models", {}).items():
            model_cfg = model_data.get("model_config", {})
            config = ModelConfig(
                model=model_cfg.get("model", model_data.get("model", "")),
                backend=model_cfg.get("backend", model_data.get("backend", "")),
                dtype=model_cfg.get("dtype", "auto"),
                tokenizer=model_cfg.get("tokenizer"),
                revision=model_cfg.get("revision"),
                attention_backend=model_cfg.get("attention_backend"),
            )
            config_dict = config.to_dict()
            # Include model_hash in per-model config if available
            if "_model_hash" in model_data:
                config_dict["model_hash"] = model_data["_model_hash"]
            models_config[model_name] = config_dict

        # Build task entries - flatten (model, task) pairs
        tasks_list: list[TaskMetricsEntry] = []
        for model_name, model_data in results.get("models", {}).items():
            for task_name, task_data in model_data.get("tasks", {}).items():
                entry = TaskMetricsEntry(
                    task=task_name,
                    model=model_name,
                    metrics=task_data.get("metrics", {}),
                    num_instances=task_data.get("num_instances", 0),
                    primary_metric=task_data.get("primary_metric"),
                    config=task_data.get("config"),
                    duration_seconds=task_data.get("duration_seconds"),
                    task_hash=task_data.get("task_hash"),
                )
                tasks_list.append(entry)

        # Build summary with primary metric for each (model, task) pair
        summary: dict[str, dict[str, ScoreSummary]] = {}
        for model_name, model_data in results.get("models", {}).items():
            summary[model_name] = {}
            for task_name, task_data in model_data.get("tasks", {}).items():
                metrics = task_data.get("metrics", {})
                preferred = task_data.get("primary_metric")
                primary = get_primary_metric(metrics, preferred)
                if primary:
                    metric_name, score = primary
                    summary[model_name][task_name] = ScoreSummary(metric=metric_name, score=score)

            # Add suite summaries to this model's summary
            if "suites" in model_data:
                for suite_name, suite_data in model_data["suites"].items():
                    metrics = suite_data.get("metrics", {})
                    primary = get_primary_metric(metrics)
                    if primary:
                        metric_name, score = primary
                        summary[model_name][suite_name] = ScoreSummary(
                            metric=metric_name, score=score
                        )

        return MetricsOutput(
            timestamp=results.get("timestamp", ""),
            config={"models": models_config},
            tasks=[t.to_dict() for t in tasks_list],
            summary={
                model: {task: s.to_dict() for task, s in tasks.items()}
                for model, tasks in summary.items()
            },
            errors=results.get("errors", []),
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
        )

    def _report_task_completion(self, model_name: str, result: TaskResult) -> None:
        """Report when a (model, task) pair completes."""
        label = f"{model_name}:{result.spec}"
        if result.error:
            console.print(f"  [red]x[/red] {label} (ERROR: {result.error})")
        else:
            console.print(
                f"  [green]v[/green] {label} ({result.num_instances} instances, "
                f"{result.duration_seconds:.1f}s)"
            )


class AsyncRunnerMixin(RunnerResultsMixin):
    """Additional shared functionality for async runners."""

    model_names: list[str]
    task_specs: list[str]
    num_workers: int | None
    gpus_per_worker: int
    num_shots_override: int | None
    limit_override: int | None

    # Subclasses should override these for print_config display
    _mode_name: str = "Async"
    _mode_description: str = "Async"

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.runners.constants import ValidationError

        if not self.model_names:
            raise ValidationError("model_names is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs using shared helper
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print configuration."""
        from rich.table import Table

        from olmo_eval.core import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        models_str = ", ".join(self.model_names)
        table.add_row("Models", models_str)
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Workers", str(self.num_workers or "auto-detect"))
        table.add_row("GPUs per Worker", str(self.gpus_per_worker))

        if self.num_shots_override is not None:
            table.add_row("Num Shots Override", str(self.num_shots_override))
        if self.limit_override is not None:
            table.add_row("Limit Override", str(self.limit_override))

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        total_pairs = len(self.model_names) * len(expanded)
        console.print(f"\n[bold]Models:[/bold] {len(self.model_names)}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded)}")
        console.print(f"[bold]Total (model, task) pairs:[/bold] {total_pairs}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def _get_num_workers(self) -> int:
        """Get number of workers based on available GPUs."""
        if self.num_workers is not None:
            return self.num_workers

        # Auto-detect GPUs
        try:
            import torch  # type: ignore[import-not-found]

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                return 1  # Fallback to single worker for CPU
            return max(1, num_gpus // self.gpus_per_worker)
        except ImportError:
            return 1  # Fallback to single worker if torch unavailable

    def _get_total_gpus(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch  # type: ignore[import-not-found]

            return torch.cuda.device_count()
        except ImportError:
            return 0

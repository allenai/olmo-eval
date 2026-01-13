"""File-based storage backend for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from olmo_eval.storage.base import EvalResult, StorageBackend, TaskResult


class FileBackend(StorageBackend):
    """File-based storage backend that saves results as JSON files.

    Results are organized by model:
        {output_dir}/{model_slug}/{date}_{run_id}.json

    An index file is maintained for queries:
        {output_dir}/.index.json
    """

    def __init__(self, output_dir: str | Path = "./results"):
        """Initialize the file backend.

        Args:
            output_dir: Directory to store result files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.output_dir / ".index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load the index file."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save the index file."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _model_slug(self, model_name: str) -> str:
        """Convert model name to filesystem-safe slug."""
        return model_name.replace("/", "_").replace(".", "-")

    def _result_path(self, result: EvalResult) -> Path:
        """Get the file path for a result."""
        model_slug = self._model_slug(result.model_name)
        date_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{date_str}_{result.run_id}.json"

    def save(self, result: EvalResult) -> str:
        """Save an evaluation result to a JSON file."""
        filepath = self._result_path(result)

        # Save the result
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Update index with queryable fields
        self._index[result.run_id] = {
            "path": str(filepath.relative_to(self.output_dir)),
            "model_name": result.model_name,
            "backend_name": result.backend_name,
            "timestamp": result.timestamp.isoformat(),
            "task_names": [t.task_name for t in result.tasks],
            # Additional queryable metadata
            "experiment_name": result.experiment_name,
            "workspace": result.workspace,
            "author": result.author,
            "model_hash": result.model_hash,
            "s3_location": result.s3_location,
        }
        self._save_index()

        return result.run_id

    def get(self, run_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by run_id."""
        if run_id not in self._index:
            return None

        filepath = self.output_dir / self._index[run_id]["path"]
        if not filepath.exists():
            # File was deleted but index wasn't updated
            del self._index[run_id]
            self._save_index()
            return None

        with open(filepath) as f:
            data = json.load(f)
        return EvalResult.from_dict(data)

    def query(
        self,
        model_name: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """Query evaluation results by filters."""
        results: list[EvalResult] = []

        for run_id, entry in self._index.items():
            # Apply filters
            if model_name and entry["model_name"] != model_name:
                continue

            if task_name and task_name not in entry.get("task_names", []):
                continue

            entry_time = datetime.fromisoformat(entry["timestamp"])
            if start_time and entry_time < start_time:
                continue
            if end_time and entry_time > end_time:
                continue

            # Load full result
            result = self.get(run_id)
            if result:
                results.append(result)

            if len(results) >= limit:
                break

        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]

    def delete(self, run_id: str) -> bool:
        """Delete an evaluation result."""
        if run_id not in self._index:
            return False

        filepath = self.output_dir / self._index[run_id]["path"]

        # Remove file if it exists
        if filepath.exists():
            filepath.unlink()

        # Update index
        del self._index[run_id]
        self._save_index()

        return True

    def list_models(self) -> list[str]:
        """List all models with stored results."""
        models = set()
        for entry in self._index.values():
            models.add(entry["model_name"])
        return sorted(models)

    def list_runs(self, model_name: str | None = None) -> list[dict[str, Any]]:
        """List all runs, optionally filtered by model.

        Returns summary information without loading full results.
        """
        runs = []
        for run_id, entry in self._index.items():
            if model_name and entry["model_name"] != model_name:
                continue
            runs.append(
                {
                    "run_id": run_id,
                    "model_name": entry["model_name"],
                    "backend_name": entry["backend_name"],
                    "timestamp": entry["timestamp"],
                    "task_names": entry.get("task_names", []),
                    "experiment_name": entry.get("experiment_name"),
                    "workspace": entry.get("workspace"),
                    "author": entry.get("author"),
                    "s3_location": entry.get("s3_location"),
                }
            )

        # Sort by timestamp descending
        runs.sort(key=lambda r: r["timestamp"], reverse=True)
        return runs


def convert_runner_results(
    results: dict[str, Any],
    run_id: str,
    s3_location: str | None = None,
    experiment_name: str | None = None,
    workspace: str | None = None,
    author: str | None = None,
    git_ref: str | None = None,
    model_hash: str | None = None,
    revision: str | None = None,
    tags: list[str] | None = None,
) -> EvalResult:
    """Convert EvalRunner results dict to EvalResult.

    Args:
        results: The results dict from EvalRunner.run()
        run_id: Unique identifier for this run.
        s3_location: Base S3 path where task results are stored.
        experiment_name: Descriptive name for the experiment.
        workspace: Beaker workspace name.
        author: Who ran the evaluation.
        git_ref: Git commit/ref for reproducibility.
        model_hash: Hash of model configuration.
        revision: Model revision/checkpoint.
        tags: List of tags for categorization.

    Returns:
        EvalResult instance.
    """
    tasks = []
    for task_idx, (spec, task_data) in enumerate(results.get("tasks", {}).items()):
        # Build S3 keys for this task if s3_location is provided
        s3_metrics_key = None
        s3_predictions_key = None
        if s3_location:
            base = s3_location.rstrip("/")
            s3_metrics_key = f"{base}/task-{task_idx:03d}-{spec}-metrics.json"
            s3_predictions_key = f"{base}/task-{task_idx:03d}-{spec}-predictions.jsonl"

        # Extract primary metric info
        metrics = task_data.get("metrics", {})
        primary_metric = task_data.get("primary_metric")
        primary_score = metrics.get(primary_metric) if primary_metric else None

        tasks.append(
            TaskResult(
                task_name=spec,
                metrics=metrics,
                num_instances=task_data.get("num_instances"),
                task_hash=task_data.get("task_hash"),
                primary_metric=primary_metric,
                primary_score=primary_score,
                s3_metrics_key=s3_metrics_key,
                s3_predictions_key=s3_predictions_key,
            )
        )

    return EvalResult(
        run_id=run_id,
        model_name=results["model"],
        backend_name=results["backend"],
        timestamp=datetime.fromisoformat(results["timestamp"]),
        tasks=tasks,
        experiment_name=experiment_name,
        workspace=workspace,
        author=author,
        tags=tags,
        git_ref=git_ref,
        model_hash=model_hash,
        revision=revision,
        s3_location=s3_location,
        config=results.get("config"),
        metadata=results.get("metadata"),
    )

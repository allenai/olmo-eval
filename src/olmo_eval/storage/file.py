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

        # Update index
        self._index[result.run_id] = {
            "path": str(filepath.relative_to(self.output_dir)),
            "model_name": result.model_name,
            "backend_name": result.backend_name,
            "timestamp": result.timestamp.isoformat(),
            "task_names": [t.task_name for t in result.tasks],
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
                }
            )

        # Sort by timestamp descending
        runs.sort(key=lambda r: r["timestamp"], reverse=True)
        return runs


def convert_runner_results(
    results: dict[str, Any],
    run_id: str,
) -> EvalResult:
    """Convert EvalRunner results dict to EvalResult.

    Args:
        results: The results dict from EvalRunner.run()
        run_id: Unique identifier for this run.

    Returns:
        EvalResult instance.
    """
    tasks = []
    for spec, task_data in results.get("tasks", {}).items():
        tasks.append(
            TaskResult(
                task_name=spec,
                metrics=task_data.get("metrics", {}),
                num_samples=task_data.get("num_instances"),
            )
        )

    return EvalResult(
        run_id=run_id,
        model_name=results["model"],
        backend_name=results["backend"],
        timestamp=datetime.fromisoformat(results["timestamp"]),
        tasks=tasks,
        config=None,  # Could include original config if available
        metadata=None,
    )

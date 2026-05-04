"""S3 download utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from olmo_eval.cli.utils import console
from olmo_eval.storage.artifacts import candidate_predictions_uris, candidate_requests_uris


def download_s3_files(
    experiment: Any,
    task_filter: tuple[str, ...],
    download_metrics: bool,
    download_predictions: bool,
    download_requests: bool,
    output_dir: str,
    s3_endpoint_url: str | None,
    s3_region: str,
) -> None:
    """Download files from S3 for an experiment.

    Uses the actual S3 paths stored in the database (s3_metrics_key, s3_predictions_key)
    rather than constructing paths from conventions.

    Args:
        experiment: The experiment ORM object.
        task_filter: Task names to filter (empty means all).
        download_metrics: Whether to download metrics.json.
        download_predictions: Whether to download predictions files.
        download_requests: Whether to download requests files.
        output_dir: Directory to save files.
        s3_endpoint_url: S3 endpoint URL (for LocalStack).
        s3_region: AWS region.
    """
    import boto3

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        region_name=s3_region,
    )

    downloaded_files: list[str] = []

    def parse_s3_uri(s3_uri: str) -> tuple[str, str] | None:
        """Parse s3://bucket/key into (bucket, key)."""
        if not s3_uri or not s3_uri.startswith("s3://"):
            return None
        path = s3_uri[5:]  # Remove 's3://'
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        return parts[0], parts[1]

    def download_file(
        s3_uri: str,
        label: str,
        *,
        candidates: tuple[str, ...] | None = None,
    ) -> str | None:
        """Download a file from S3 URI."""
        for candidate_uri in candidates or (s3_uri,):
            parsed = parse_s3_uri(candidate_uri)
            if not parsed:
                continue

            bucket, key = parsed
            # Use just the filename for local path to avoid deeply nested directories
            filename = Path(key).name
            local_file = output_path / experiment.experiment_id / filename
            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                s3_client.download_file(bucket, key, str(local_file))
                console.print(f"[green]Downloaded:[/green] {local_file}")
                return str(local_file)
            except Exception:
                continue

        console.print(f"[yellow]Warning:[/yellow] Failed to download {s3_uri}")
        return None

    # Download metrics.json from experiment's s3_location
    if download_metrics and experiment.s3_location:
        parsed = parse_s3_uri(experiment.s3_location.rstrip("/") + "/metrics.json")
        if parsed:
            bucket, key = parsed
            local_file = output_path / experiment.experiment_id / "metrics.json"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_client.download_file(bucket, key, str(local_file))
                console.print(f"[green]Downloaded:[/green] {local_file}")
                downloaded_files.append(str(local_file))
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to download metrics.json: {e}")

    # Download predictions files using paths stored in database
    tasks_to_download = experiment.tasks
    if task_filter:
        tasks_to_download = [t for t in tasks_to_download if t.task_name in task_filter]

    for task in tasks_to_download:
        if download_predictions and task.s3_predictions_key:
            result = download_file(
                task.s3_predictions_key,
                f"{task.task_name} predictions",
                candidates=candidate_predictions_uris(
                    task.s3_predictions_key,
                    model_name=experiment.model_name,
                    task_name=task.task_name,
                    task_hash=task.task_hash,
                ),
            )
            if result:
                downloaded_files.append(result)

        if download_requests and task.s3_requests_key:
            result = download_file(
                task.s3_requests_key,
                f"{task.task_name} requests",
                candidates=candidate_requests_uris(
                    task.s3_requests_key,
                    model_name=experiment.model_name,
                    task_name=task.task_name,
                    task_hash=task.task_hash,
                ),
            )
            if result:
                downloaded_files.append(result)
        elif download_requests and task.s3_predictions_key:
            requests_uri = task.s3_predictions_key.replace("predictions.jsonl", "requests.jsonl")
            result = download_file(
                requests_uri,
                f"{task.task_name} requests",
                candidates=candidate_requests_uris(
                    requests_uri,
                    model_name=experiment.model_name,
                    task_name=task.task_name,
                    task_hash=task.task_hash,
                ),
            )
            if result:
                downloaded_files.append(result)

    if not downloaded_files:
        console.print("[yellow]No files were downloaded.[/yellow]")

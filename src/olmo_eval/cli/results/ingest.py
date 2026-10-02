"""Ingest command: save a finished run's output into the results database."""

from __future__ import annotations

from pathlib import Path

import click

from olmo_eval.cli.results.options import db_options, get_database_session, s3_options
from olmo_eval.cli.utils import console
from olmo_eval.storage.ingest import ReadText, load_output_dir


def local_reader() -> ReadText:
    """Return a reader for files on the local filesystem."""

    def read_text(path: str) -> str | None:
        file = Path(path)
        return file.read_text() if file.is_file() else None

    return read_text


def s3_reader(s3_endpoint_url: str | None, s3_region: str) -> ReadText:
    """Return a reader for ``s3://bucket/key`` URIs."""
    import boto3

    client = boto3.client("s3", endpoint_url=s3_endpoint_url, region_name=s3_region)

    def read_text(uri: str) -> str | None:
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except client.exceptions.NoSuchKey:
            return None
        return response["Body"].read().decode()

    return read_text


@click.command()
@click.argument("source")
@click.option(
    "--model-name",
    default=None,
    help="Name to store the model under. Defaults to the name recorded in metrics.json.",
)
@click.option(
    "--s3-location",
    default=None,
    help="S3 prefix the run was uploaded to, used for artifact links. "
    "Defaults to SOURCE when it is an s3:// URI.",
)
@click.option(
    "--no-instances",
    is_flag=True,
    default=False,
    help="Save experiment and task rows only, without instance predictions.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be saved without connecting to the database.",
)
@db_options
@s3_options
def ingest(
    source: str,
    model_name: str | None,
    s3_location: str | None,
    no_instances: bool,
    dry_run: bool,
    db_host: str | None,
    db_port: int | None,
    db_name: str | None,
    db_user: str | None,
    db_password: str | None,
    s3_endpoint_url: str | None,
    s3_region: str,
) -> None:
    """Save a run's output directory or S3 prefix to the results database.

    SOURCE holds the run's metrics.json and predictions/ directory. Saving the
    same run again updates its stored rows rather than adding new ones.

    \b
    Examples:
        olmo-eval results ingest ./output
        olmo-eval results ingest s3://bucket/olmo-eval/group/model_abc123/exp-id
    """
    from olmo_eval.runners.io.storage import ResultsSaveError, save_results

    is_s3 = source.startswith("s3://")
    read_text = s3_reader(s3_endpoint_url, s3_region) if is_s3 else local_reader()
    if is_s3 and s3_location is None:
        s3_location = source.rstrip("/")

    try:
        loaded = load_output_dir(
            source,
            read_text,
            model_name=model_name,
            include_predictions=not no_instances,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    tasks = loaded.results["tasks"]
    instance_count = sum(len(task.get("predictions", [])) for task in tasks.values())
    console.print(
        f"Experiment [bold]{loaded.experiment_id}[/bold]: "
        f"model={loaded.results['model']}, model_hash={loaded.model_hash}, "
        f"tasks={len(tasks)}, instances={instance_count}"
    )
    for spec in loaded.missing_predictions:
        console.print(f"[yellow]Warning:[/yellow] no predictions file for {spec}")

    if dry_run:
        console.print("[dim]Dry run; nothing saved.[/dim]")
        return

    from olmo_eval.storage.backends.postgres import PostgresBackend

    db = get_database_session(db_host, db_port, db_name, db_user, db_password)
    backend = PostgresBackend.from_database_session(db)
    try:
        save_results(
            results=loaded.results,
            storages=[backend],
            experiment_id=loaded.experiment_id,
            model_hash=loaded.model_hash,
            s3_location=s3_location,
            experiment_name=loaded.experiment_name,
            experiment_group=loaded.experiment_group,
            experiment_duration_seconds=loaded.experiment_duration_seconds,
            provider_init_seconds=loaded.provider_init_seconds,
        )
    except ResultsSaveError as e:
        raise click.ClickException(str(e)) from e
    finally:
        backend.dispose()

    console.print(f"[green]Saved experiment {loaded.experiment_id}[/green]")

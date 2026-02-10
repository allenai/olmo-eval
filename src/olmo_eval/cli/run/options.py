"""Option decorators for the run command."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

import click

from olmo_eval.common.constants.infrastructure import LOCAL_RESULT_DIR

F = TypeVar("F", bound=Callable[..., Any])


def parallelism_options(func: F) -> F:  # noqa: UP047
    """Parallelism options."""

    @click.option(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    @click.option(
        "--parallelism",
        "-P",
        type=int,
        default=1,
        help="Number of model instances to run in parallel",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def storage_options(func: F) -> F:  # noqa: UP047
    """S3 and database storage options."""

    @click.option(
        "--store",
        is_flag=True,
        help="Persist results to the configured database",
    )
    @click.option("--s3-bucket", help="S3 bucket for storing evaluation results")
    @click.option("--s3-prefix", help="S3 prefix/path within bucket for results")
    @click.option("--s3-group", help="S3 group name (used in path structure)")
    @click.option(
        "--s3-endpoint-url",
        envvar="S3_ENDPOINT_URL",
        help="S3 endpoint URL (for S3-compatible storage)",
    )
    @click.option(
        "--s3-region",
        default="us-east-1",
        envvar="AWS_REGION",
        help="S3 region (default: us-east-1)",
    )
    @click.option(
        "--db-host",
        default="localhost",
        envvar="PGHOST",
        help="PostgreSQL host",
    )
    @click.option(
        "--db-port",
        default=5432,
        type=int,
        envvar="PGPORT",
        help="PostgreSQL port",
    )
    @click.option(
        "--db-name",
        default="olmo_eval",
        envvar="PGDATABASE",
        help="PostgreSQL database name",
    )
    @click.option(
        "--db-user",
        default="postgres",
        envvar="PGUSER",
        help="PostgreSQL user",
    )
    @click.option(
        "--db-password",
        default="postgres",
        envvar="PGPASSWORD",
        help="PostgreSQL password",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def experiment_options(func: F) -> F:  # noqa: UP047
    """Experiment metadata options."""

    @click.option(
        "--experiment-name",
        help="Human-readable experiment name for database storage",
    )
    @click.option(
        "--experiment-group",
        help="Experiment group for grouping related experiments",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def output_options(func: F) -> F:  # noqa: UP047
    """Output control options."""

    @click.option(
        "--output-dir",
        "-O",
        default=LOCAL_RESULT_DIR,
        help="Output directory",
    )
    @click.option(
        "--save-predictions/--no-save-predictions",
        "save_predictions",
        default=True,
        help="Save per-instance predictions to JSONL (default: enabled)",
    )
    @click.option(
        "--save-requests/--no-save-requests",
        "save_requests",
        default=True,
        help="Save per-instance requests to JSONL (default: enabled)",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Print config and exit without running",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def inspect_options(func: F) -> F:  # noqa: UP047
    """Debug and inspection options."""

    @click.option(
        "--debug-requests",
        is_flag=True,
        hidden=True,
        help="Log HTTP requests/responses to inference providers",
    )
    @click.option(
        "--debug-provider",
        is_flag=True,
        hidden=True,
        help="Enable verbose provider logging",
    )
    @click.option(
        "--inspect",
        is_flag=True,
        help="Enable all inspection flags (instance, formatted, tokens, request, response)",
    )
    @click.option(
        "--inspect-instance",
        is_flag=True,
        help="Print the first instance of each task before running evaluation",
    )
    @click.option(
        "--inspect-formatted",
        is_flag=True,
        help="Show formatted prompt (after template applied) before evaluation",
    )
    @click.option(
        "--inspect-tokens",
        is_flag=True,
        help="Show token array before evaluation",
    )
    @click.option(
        "--inspect-response",
        is_flag=True,
        help="Print the first response of each task after model generation",
    )
    @click.option(
        "--inspect-request",
        is_flag=True,
        help="Print the first request of each task before model generation",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def harness_options(func: F) -> F:  # noqa: UP047
    """Harness configuration options."""

    @click.option(
        "--harness",
        "harness_preset",
        type=str,
        default=None,
        help="Harness preset name (e.g., 'search') for tool/prompt configuration",
    )
    @click.option(
        "--harness-config",
        type=click.Path(exists=True),
        default=None,
        help="Path to harness config YAML/JSON file",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)

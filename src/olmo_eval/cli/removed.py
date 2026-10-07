"""Hidden stand-ins for CLI options and commands that the results dashboard replaced.

Old scripts that still pass them get a short explanation instead of a bare usage error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import click

from olmo_eval.cli.utils import console

F = TypeVar("F", bound=Callable[..., Any])

DASHBOARD_URL = "https://olmo-eval.allen.ai"

REMOVED_FLAGS = ("--store/--no-store",)
REMOVED_VALUE_OPTIONS = (
    "--s3-bucket",
    "--s3-prefix",
    "--s3-group",
    "--s3-endpoint-url",
    "--s3-region",
    "--db-host",
    "--db-port",
    "--db-name",
    "--db-user",
    "--db-password",
)


def _warn_removed(ctx: click.Context, param: click.Parameter, value: Any) -> None:
    if value is None or ctx.resilient_parsing:
        return
    source = ctx.get_parameter_source(param.name or "")
    if source is not click.core.ParameterSource.COMMANDLINE:
        return
    console.print(
        f"[yellow]Warning:[/yellow] {param.opts[0]} was removed and is ignored. Results now "
        f"upload to the dashboard at {DASHBOARD_URL} (pass --no-upload to keep them local)",
        highlight=False,
    )


def removed_storage_options(func: F) -> F:  # noqa: UP047
    """Accept the removed S3 and Postgres result-storage options and warn when they are used."""
    for name in REMOVED_VALUE_OPTIONS:
        func = click.option(
            name, hidden=True, expose_value=False, callback=_warn_removed, default=None
        )(func)
    for name in REMOVED_FLAGS:
        func = click.option(
            name, hidden=True, expose_value=False, callback=_warn_removed, default=None
        )(func)
    return func


def removed_command(name: str, replacement: str) -> click.Command:
    """A hidden command that explains what replaced it and exits with status 2."""

    @click.command(
        name,
        hidden=True,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        add_help_option=False,
    )
    @click.pass_context
    def command(ctx: click.Context) -> None:
        console.print(
            f"[red]Error:[/red] `{ctx.command_path}` was removed. {replacement}", highlight=False
        )
        ctx.exit(2)

    return command


BROWSE_IN_DASHBOARD = f"Browse and compare results in the dashboard: {DASHBOARD_URL}"

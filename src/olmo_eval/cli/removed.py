"""Hidden stand-ins for CLI commands that the results dashboard replaced.

Old scripts that still call them get a short explanation instead of a bare usage error.
"""

from __future__ import annotations

import click

from olmo_eval.cli.utils import console

DASHBOARD_URL = "https://olmo-eval.allen.ai"


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

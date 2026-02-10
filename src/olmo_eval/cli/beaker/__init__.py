"""Beaker commands for olmo-eval CLI."""

import click

from olmo_eval.cli.beaker.group import group
from olmo_eval.cli.beaker.launch import launch
from olmo_eval.cli.beaker.watch import watch


@click.group()
def beaker() -> None:
    """Beaker job management commands.

    Commands for launching, monitoring, and managing evaluation jobs on Beaker.
    """
    pass


# Register subcommands
beaker.add_command(launch)
beaker.add_command(watch)
beaker.add_command(group)

__all__ = ["beaker", "launch", "watch", "group"]

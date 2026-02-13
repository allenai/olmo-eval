"""Beaker commands for olmo-eval CLI."""

import click
from rich.table import Table

from olmo_eval.cli.beaker.group import group
from olmo_eval.cli.beaker.launch import launch
from olmo_eval.cli.beaker.watch import watch
from olmo_eval.cli.utils import console
from olmo_eval.common.constants.infrastructure import BEAKER_KNOWN_CLUSTERS


@click.group()
def beaker() -> None:
    """Beaker job management commands.

    Commands for launching, monitoring, and managing evaluation jobs on Beaker.
    """
    pass


@beaker.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def clusters(filter: str) -> None:
    """List available cluster aliases."""
    table = Table(title="Cluster Aliases")
    table.add_column("Alias", style="cyan")
    table.add_column("Clusters", style="dim")

    for alias, cluster_list in sorted(BEAKER_KNOWN_CLUSTERS.items()):
        if filter.lower() in alias.lower():
            table.add_row(alias, ", ".join(cluster_list))

    console.print(table)


# Register subcommands
beaker.add_command(launch)
beaker.add_command(watch)
beaker.add_command(group)

__all__ = ["beaker", "clusters", "launch", "watch", "group"]

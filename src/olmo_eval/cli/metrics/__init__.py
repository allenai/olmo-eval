"""CLI commands for querying and displaying inference metrics."""

import click

from olmo_eval.cli.metrics.plot import plot


@click.group()
def metrics() -> None:
    """Query and display inference metrics."""
    pass


# Register subcommands
metrics.add_command(plot)

__all__ = ["metrics", "plot"]

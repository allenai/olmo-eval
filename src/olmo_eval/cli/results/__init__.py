"""CLI commands for querying and displaying evaluation results."""

import click

from olmo_eval.cli.results.query import query


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


# Register subcommands
results.add_command(query)

__all__ = ["results", "query"]

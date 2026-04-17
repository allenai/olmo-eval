"""CLI commands for querying and displaying evaluation results."""

import click

from olmo_eval.cli.results.pairwise import pairwise
from olmo_eval.cli.results.query import query


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


# Register subcommands
results.add_command(query)
results.add_command(pairwise)

__all__ = ["results", "query", "pairwise"]

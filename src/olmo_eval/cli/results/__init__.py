"""Results CLI commands."""

import click

from olmo_eval.cli.results.discover import group, groups, suites
from olmo_eval.cli.results.pairwise import pairwise
from olmo_eval.cli.results.query import query


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


results.add_command(query)
results.add_command(pairwise)
results.add_command(suites)
results.add_command(groups)
results.add_command(group)

__all__ = ["results", "query", "pairwise", "suites", "groups", "group"]

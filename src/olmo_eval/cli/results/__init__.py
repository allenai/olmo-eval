"""Results CLI commands."""

import click

from olmo_eval.cli.results.discover import group, groups, suites
from olmo_eval.cli.results.query import query
from olmo_eval.cli.results.viewer import viewer


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


results.add_command(query)
results.add_command(viewer)
results.add_command(suites)
results.add_command(groups)
results.add_command(group)

__all__ = ["results", "query", "viewer", "suites", "groups", "group"]

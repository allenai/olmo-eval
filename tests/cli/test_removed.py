"""Removed result-storage options and commands explain what replaced them."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from olmo_eval.cli import main
from olmo_eval.cli.removed import removed_storage_options


@removed_storage_options
@click.command()
def demo() -> None:
    click.echo("ran")


def test_removed_options_are_accepted_with_a_warning() -> None:
    result = CliRunner().invoke(demo, ["--s3-bucket", "b", "--db-host", "h", "--no-store"])

    assert result.exit_code == 0, result.output
    assert "ran" in result.output
    for name in ("--s3-bucket", "--db-host", "--store"):
        assert f"{name} was removed and is ignored" in result.output


def test_removed_options_are_quiet_when_unused() -> None:
    result = CliRunner().invoke(demo, [])

    assert result.exit_code == 0
    assert "removed" not in result.output


def test_removed_options_are_hidden_from_help() -> None:
    result = CliRunner().invoke(demo, ["--help"])

    assert "--s3-bucket" not in result.output and "--store" not in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["results", "query", "--model", "x"],
        ["results", "groups"],
        ["results", "group", "g"],
        ["results", "suites"],
        ["results", "viewer", "--port", "8080"],
        ["metrics", "--recent"],
    ],
)
def test_removed_commands_point_to_the_dashboard(argv: list[str]) -> None:
    result = CliRunner().invoke(main, argv)

    assert result.exit_code == 2
    assert "was removed" in result.output
    assert "https://olmo-eval.allen.ai" in result.output


def test_removed_commands_are_hidden_from_help() -> None:
    result = CliRunner().invoke(main, ["results", "--help"])

    assert "query" not in result.output and "upload" in result.output

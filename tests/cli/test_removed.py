"""Removed results commands explain what replaced them."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from olmo_eval.cli import main


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

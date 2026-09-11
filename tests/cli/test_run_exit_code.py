"""Tests for the run command's all-instances-failed exit gate."""

import pytest

from olmo_eval.cli.run import _scored_nothing


class TestScoredNothing:
    """A run that scored no instances must not be reported as a success.

    ``aggregate_results`` writes ``num_instances`` only for tasks that produced
    results; a task whose instances all failed carries an ``error`` key instead.
    """

    @pytest.mark.parametrize(
        ("label", "results"),
        [
            ("task errored outright", {"tasks": {"t": {"error": "597 instances failed"}}}),
            ("explicit zero instances", {"tasks": {"t": {"num_instances": 0}}}),
            ("every task errored", {"tasks": {"a": {"error": "x"}, "b": {"error": "y"}}}),
        ],
    )
    def test_detects_a_run_with_nothing_scored(self, label, results):
        assert _scored_nothing(results) is True, label

    @pytest.mark.parametrize(
        ("label", "results"),
        [
            ("all instances scored", {"tasks": {"t": {"num_instances": 100}}}),
            (
                "partial failure still has data",
                {"tasks": {"a": {"error": "x"}, "b": {"num_instances": 10}}},
            ),
        ],
    )
    def test_leaves_runs_that_produced_data_alone(self, label, results):
        assert _scored_nothing(results) is False, label

    @pytest.mark.parametrize(
        ("label", "results"),
        [
            ("no tasks ran at all", {"tasks": {}}),
            ("no tasks key", {}),
            ("runner returned nothing", None),
            ("unexpected shape", "surprise"),
        ],
    )
    def test_never_fails_a_run_on_an_unexpected_shape(self, label, results):
        # This gates the process exit code, so anything unrecognised must be
        # treated as "cannot tell", never as failure.
        assert _scored_nothing(results) is False, label

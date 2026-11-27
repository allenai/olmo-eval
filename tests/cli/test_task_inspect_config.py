"""Tests for ``olmo-eval task inspect --config``."""

import json

import pytest
from click.testing import CliRunner

import olmo_eval.evals.tasks  # noqa: F401  (registers every task)
from olmo_eval.cli.task import task
from olmo_eval.common.inspection import describe_task_spec
from olmo_eval.evals.tasks.common.base import Task


@pytest.fixture(autouse=True)
def no_instances(monkeypatch):
    """--config must answer from the registry alone, without loading a dataset."""

    def fail(self):
        raise AssertionError("instances were loaded")

    monkeypatch.setattr(Task, "instances", property(fail))


def _invoke(*args: str):
    return CliRunner().invoke(task, ["inspect", *args])


def test_describe_names_task_class_variants_and_metrics() -> None:
    description = describe_task_spec("mbpp:olmo3base:bpb")

    assert description["task"] == "mbpp:olmo3base"
    assert description["task_class"] == "olmo_eval.evals.tasks.mbpp.MBPPOlmo3Base"
    assert description["variants"] == [{"name": "bpb", "sets": ["formatter", "metrics"]}]
    assert description["metrics"] == ["bits_per_byte:bits_per_byte"]
    assert description["primary_metric"] == "pass_at_1:code_exec"
    assert description["primary_metric_computed"] is False


def test_describe_reports_a_computed_primary() -> None:
    description = describe_task_spec("mbpp:bpb:olmo3base")

    assert description["task"] == "mbpp:bpb"
    assert description["primary_metric"] == "bits_per_byte:bits_per_byte"
    assert description["primary_metric_computed"] is True


def test_describe_reports_no_designated_primary() -> None:
    # Several metrics and no explicit primary: the runner picks one at display time.
    description = describe_task_spec("lab_bench_dbqa")

    assert len(description["metrics"]) > 1
    assert description["primary_metric"] == "accuracy:multiple_choice"


def test_config_flag_prints_resolution_and_flags_an_uncomputed_primary() -> None:
    result = _invoke("mbpp:olmo3base:bpb", "--config")

    assert result.exit_code == 0, result.output
    assert "mbpp:olmo3base" in result.output
    assert "MBPPOlmo3Base" in result.output
    assert "sets formatter, metrics" in result.output
    assert "not computed by this spec" in result.output


def test_config_json_prints_the_description_alone() -> None:
    result = _invoke("arc_easy:olmo3base:bpb", "--config", "--json")

    assert result.exit_code == 0, result.output
    description = json.loads(result.output)
    assert description["task"] == "arc_easy"
    assert [v["name"] for v in description["variants"]] == ["olmo3base", "bpb"]
    assert description["primary_metric_computed"] is True
    assert description["config"]["num_fewshot"] == 5


def test_unknown_variant_is_an_error_not_a_traceback() -> None:
    result = _invoke("arc_easy:nope", "--config")

    assert result.exit_code == 1
    assert "Error" in result.output
    assert "Traceback" not in result.output

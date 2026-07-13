import asyncio
import json
import stat
from pathlib import Path
from unittest import mock

import pytest

from olmo_eval.common.execution.environment import ExecutionResult
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import (
    PRIMARY_METRICS,
    DeepScholarArgs,
)
from olmo_eval.evals.external.benchmarks.deepscholar.eval import (
    LOTUS_REF,
    DeepScholarExternalEval,
)
from olmo_eval.evals.external.benchmarks.deepscholar.result_parser import (
    parse_aggregate_csv,
    parse_per_query_csv,
)
from olmo_eval.harness.sandbox.config import SandboxConfig, SandboxMode


def test_output_directories_are_bind_mounted(tmp_path: Path) -> None:
    evaluator = DeepScholarExternalEval()
    base_config = SandboxConfig(image="test", mode=SandboxMode.DOCKER)

    with mock.patch.object(
        SandboxedExternalEval, "_create_sandbox_config", return_value=base_config
    ):
        config = evaluator._create_sandbox_config("podman", str(tmp_path))

    destination = tmp_path / "deepscholar_results"
    assert config.inject_swerex
    assert config.volumes == (
        (str(destination / "generation"), evaluator._gen_dir),
        (str(destination / "evaluation"), evaluator._eval_dir),
    )
    assert (destination / "generation").is_dir()
    assert (destination / "evaluation").is_dir()
    assert stat.S_IMODE((destination / "generation").stat().st_mode) == 0o777
    assert stat.S_IMODE((destination / "evaluation").stat().st_mode) == 0o777


def test_setup_pins_lotus_revision() -> None:
    evaluator = DeepScholarExternalEval()

    assert any(LOTUS_REF in command for command in evaluator.setup_command)


def test_all_metrics_argument_expands_to_primary_metrics() -> None:
    assert DeepScholarArgs.from_dict({"evals": "all"}).evals == list(PRIMARY_METRICS)


def test_generation_runs_once_and_counts_only_scorable_artifacts(tmp_path: Path) -> None:
    evaluator = DeepScholarExternalEval()
    generation = tmp_path / "deepscholar_results" / "generation"
    complete = generation / "0"
    incomplete = generation / "1"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    for filename in ("final_report.md", "intro.md", "paper.csv"):
        (complete / filename).write_text("complete")
    (incomplete / "final_report.md").write_text("not enough")
    (generation / "summary.json").write_text(
        json.dumps([{"status": "success"}, {"status": "success"}])
    )

    executor = mock.Mock()
    executor.execute_command = mock.AsyncMock(
        side_effect=[
            ExecutionResult(success=True, output="63\n"),
            ExecutionResult(success=True, output="generation output"),
        ]
    )

    counts = asyncio.run(
        evaluator._run_generation(
            executor,
            DeepScholarArgs(limit=2, search_backend="s2"),
            [],
            str(tmp_path),
        )
    )

    assert counts == (1, 2, True)
    generation_commands = [
        call.args[0]
        for call in executor.execute_command.await_args_list
        if "--output-folder" in call.args[0]
    ]
    assert len(generation_commands) == 1
    assert "--start-idx 0 --end-idx 2" in generation_commands[0]
    assert "Chunk" not in generation_commands[0]


def test_metric_parsers_skip_invalid_values() -> None:
    text = (
        "folder_path,cite_p\n"
        "/workspace/outputs/generation/1,0.25\n"
        "/workspace/outputs/generation/2,nan\n"
        "/workspace/outputs/generation/3,invalid\n"
    )

    assert parse_per_query_csv(text, "cite_p") == {"1": 0.25}
    assert parse_aggregate_csv("baseline_name,cite_p\ndeepscholar_base,nan\n", "cite_p") is None


@pytest.mark.parametrize("metric", PRIMARY_METRICS)
def test_fixed_metrics_use_requested_query_denominator(metric: str) -> None:
    evaluator = DeepScholarExternalEval()
    aggregate_files = {
        f"{name}/aggregated_results.csv": f"baseline_name,{name}\ndeepscholar_base,0.5\n"
        for name in PRIMARY_METRICS
    }
    query_files = {
        f"{name}/deepscholar_base.csv": (
            f"folder_path,{name}\n/workspace/generation/0,0.25\n/workspace/generation/1,0.75\n"
        )
        for name in PRIMARY_METRICS
    }

    with mock.patch.object(
        evaluator,
        "_read_dir",
        new=mock.AsyncMock(side_effect=[aggregate_files, query_files]),
    ):
        result = asyncio.run(
            evaluator._extract_results(
                mock.Mock(),
                "raw output",
                0,
                n_success=2,
                n_total=3,
                generation_ok=True,
                requested_metrics=PRIMARY_METRICS,
            )
        )

    assert result.metrics[metric] == pytest.approx(0.5)
    assert result.metrics[f"{metric}_fixed"] == pytest.approx(1 / 3)
    assert result.metrics["geomean"] == pytest.approx(0.5)
    assert result.metrics["geomean_fixed"] == pytest.approx(1 / 3)
    assert result.metadata["queries_requested"] == 3
    assert result.metadata["queries_generated"] == 2
    assert result.metadata["queries_scored"] == 2
    assert result.metadata["generation_complete"] is False

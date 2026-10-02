"""Tests for the ``results ingest`` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from olmo_eval.cli.results import results
from olmo_eval.runners.io.storage import ResultsSaveError


def _write_metrics(output_dir: Path) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-10-01T12:00:00",
                "config": {"provider": {"kind": "vllm", "model": "org/m"}, "model_hash": "mh"},
                "tasks": [
                    {
                        "task": "gsm8k",
                        "metrics": {"exact_match": {"exact_match": 0.5}},
                        "num_instances": 1,
                        "task_hash": "gsm8k-hash",
                    }
                ],
                "summary": {},
                "experiment_id": "exp-1",
                "experiment_group": "grp",
                "model_name": "m",
            }
        )
    )


def test_dry_run_reports_without_saving(tmp_path: Path) -> None:
    _write_metrics(tmp_path)

    with patch("olmo_eval.runners.io.storage.save_results") as save:
        result = CliRunner().invoke(results, ["ingest", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run; nothing saved." in result.output
    assert "no predictions file for gsm8k" in result.output
    save.assert_not_called()


def _invoke_with_db(args: list[str], save: MagicMock) -> tuple[object, MagicMock]:
    backend = MagicMock()
    with (
        patch("olmo_eval.cli.results.ingest.get_database_session"),
        patch(
            "olmo_eval.storage.backends.postgres.PostgresBackend.from_database_session",
            return_value=backend,
        ),
        patch("olmo_eval.runners.io.storage.save_results", save),
    ):
        return CliRunner().invoke(results, ["ingest", *args]), backend


def test_saves_with_identifiers_from_metrics_json(tmp_path: Path) -> None:
    _write_metrics(tmp_path)
    save = MagicMock()

    result, backend = _invoke_with_db([str(tmp_path), "--s3-location", "s3://bucket/prefix"], save)

    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    kwargs = save.call_args.kwargs
    assert kwargs["storages"] == [backend]
    assert kwargs["experiment_id"] == "exp-1"
    assert kwargs["model_hash"] == "mh"
    assert kwargs["experiment_group"] == "grp"
    assert kwargs["s3_location"] == "s3://bucket/prefix"
    assert kwargs["results"]["model"] == "m"
    backend.dispose.assert_called_once()


def test_save_failure_exits_nonzero(tmp_path: Path) -> None:
    _write_metrics(tmp_path)
    save = MagicMock(side_effect=ResultsSaveError(["save m to PostgresBackend: boom"]))

    result, backend = _invoke_with_db([str(tmp_path)], save)

    assert result.exit_code == 1  # type: ignore[attr-defined]
    assert "boom" in result.output  # type: ignore[attr-defined]
    backend.dispose.assert_called_once()


def test_missing_metrics_exits_nonzero(tmp_path: Path) -> None:
    result = CliRunner().invoke(results, ["ingest", str(tmp_path), "--dry-run"])

    assert result.exit_code == 1
    assert "No metrics.json" in result.output

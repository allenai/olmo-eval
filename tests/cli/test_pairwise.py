"""Tests for the pairwise results CLI."""

from __future__ import annotations

import importlib
from pathlib import Path

from click.testing import CliRunner

from olmo_eval.analysis.pairwise import ModelMeta, PairStats, PairwiseResult


class _DummySession:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyDB:
    def session(self) -> _DummySession:
        return _DummySession()

    def dispose(self) -> None:
        pass


def _build_pairwise_result(*, dropped: int = 0) -> PairwiseResult:
    return PairwiseResult(
        task_name="olmobase:math",
        suite_name="olmobase:math",
        task_names=("minerva_math_algebra:olmo3base",),
        metric="accuracy:exact_match",
        margin=0.0,
        instance_count=12,
        models=[
            ModelMeta(
                label="model-a\n(abc12345)",
                model_name="model-a",
                model_hash="abc12345deadbeef",
                timestamp="2026-04-19T00:00:00+00:00",
            ),
            ModelMeta(
                label="model-b\n(def67890)",
                model_name="model-b",
                model_hash="def67890deadbeef",
                timestamp="2026-04-19T00:00:00+00:00",
            ),
        ],
        pairs=[
            PairStats(index_a=0, index_b=1, wins_a=7, wins_b=5, ties=0),
        ],
        n_experiments_matched=2,
        n_experiments_dropped=dropped,
    )


def test_pairwise_forwards_exclude_filters(monkeypatch, tmp_path: Path) -> None:
    """CLI exclusions should be threaded into compute_pairwise."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    pairwise_cli = importlib.import_module("olmo_eval.cli.results.pairwise")

    captured: dict[str, object] = {}

    def fake_compute_pairwise(**kwargs):
        captured.update(kwargs)
        return _build_pairwise_result()

    monkeypatch.setattr(analysis_pairwise, "compute_pairwise", fake_compute_pairwise)
    monkeypatch.setattr(pairwise_cli, "get_database_session", lambda *args: _DummyDB())

    output_path = tmp_path / "pairwise.json"
    runner = CliRunner()
    result = runner.invoke(
        pairwise_cli.pairwise,
        [
            "--model",
            "model-",
            "--exclude-model",
            "skip-",
            "--model-hash",
            "abc",
            "--exclude-model-hash",
            "dead",
            "--suite",
            "olmobase:math",
            "--exclude-task",
            "gsm8k:olmo3base",
            "--exclude-task-hash",
            "fff",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert captured["model_names"] == ["model-"]
    assert captured["exclude_model_names"] == ["skip-"]
    assert captured["model_hashes"] == ["abc"]
    assert captured["exclude_model_hashes"] == ["dead"]
    assert captured["exclude_task_names"] == ["gsm8k:olmo3base"]
    assert captured["exclude_task_hashes"] == ["fff"]


def test_pairwise_keep_all_status_uses_actual_flag_name(monkeypatch, tmp_path: Path) -> None:
    """The keep-all summary should print the real CLI flag without spaces."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    pairwise_cli = importlib.import_module("olmo_eval.cli.results.pairwise")

    monkeypatch.setattr(
        analysis_pairwise,
        "compute_pairwise",
        lambda **kwargs: _build_pairwise_result(),
    )
    monkeypatch.setattr(pairwise_cli, "get_database_session", lambda *args: _DummyDB())

    output_path = tmp_path / "pairwise.json"
    runner = CliRunner()
    result = runner.invoke(
        pairwise_cli.pairwise,
        [
            "--model",
            "model-",
            "--suite",
            "olmobase:math",
            "--all",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--all; no dedupe" in result.output
    assert "-- all; no dedupe" not in result.output

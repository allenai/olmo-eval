"""Tests for the results viewer CLI."""

from __future__ import annotations

import importlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
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


class _StaticExecuteResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _StaticTaskSession:
    def __init__(self, rows):
        self._rows = list(rows)

    def execute(self, _query):
        return _StaticExecuteResult(self._rows)


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


_BROWSER_PAYLOAD_RE = re.compile(
    r"window\.RESULTS_VIEWER_DATA = (?P<payload>.+?);\s*</script>",
    re.DOTALL,
)


def _extract_browser_payload(html: str) -> dict[str, Any]:
    match = _BROWSER_PAYLOAD_RE.search(html)
    assert match is not None, "RESULTS_VIEWER_DATA payload not found in viewer HTML"
    return json.loads(match.group("payload"))


def test_results_viewer_json_blob_forwards_exclude_filters(monkeypatch) -> None:
    """JSON dump mode should stream a blob and thread exclusions into compute_pairwise."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
    viewer_cli = importlib.import_module("olmo_eval.cli.results.viewer")

    captured: dict[str, object] = {}

    def fake_compute_pairwise(**kwargs):
        captured.update(kwargs)
        return _build_pairwise_result()

    monkeypatch.setattr(analysis_pairwise, "compute_pairwise", fake_compute_pairwise)
    monkeypatch.setattr(viewer_cli, "get_database_session", lambda *args: _DummyDB())

    runner = CliRunner()
    result = runner.invoke(
        results_cli.results,
        [
            "viewer",
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
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"scope_name": "olmobase:math"' in result.output
    assert '"model_a_label": "model-a (abc12345)"' in result.output
    assert '"shared_instance_mean_score"' in result.output
    assert '"task_scores_by_task_name"' in result.output
    assert '"task_name": "olmobase:math"' not in result.output
    assert '"model_a": "model-a (abc12345)"' not in result.output
    assert captured["model_names"] == ["model-"]
    assert captured["exclude_model_names"] == ["skip-"]
    assert captured["model_hashes"] == ["abc"]
    assert captured["exclude_model_hashes"] == ["dead"]
    assert captured["exclude_task_names"] == ["gsm8k:olmo3base"]
    assert captured["exclude_task_hashes"] == ["fff"]


def test_results_viewer_dump_repeated_runs_status_uses_plain_language(
    monkeypatch, tmp_path: Path
) -> None:
    """Repeated-run mode should be described without surfacing CLI flag syntax."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
    viewer_cli = importlib.import_module("olmo_eval.cli.results.viewer")

    monkeypatch.setattr(
        analysis_pairwise,
        "compute_pairwise",
        lambda **kwargs: _build_pairwise_result(),
    )
    monkeypatch.setattr(viewer_cli, "get_database_session", lambda *args: _DummyDB())

    output_path = tmp_path / "pairwise.json"
    runner = CliRunner()
    result = runner.invoke(
        results_cli.results,
        [
            "viewer",
            "--model",
            "model-",
            "--suite",
            "olmobase:math",
            "--repeated-runs",
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "repeated runs enabled" in result.output
    assert "--" not in result.output


def test_results_viewer_starts_server(monkeypatch) -> None:
    """`results viewer` should start the local results viewer server."""
    results_cli = importlib.import_module("olmo_eval.cli.results")
    viewer_cli = importlib.import_module("olmo_eval.cli.results.viewer")

    captured: dict[str, object] = {}

    def fake_serve(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(viewer_cli, "_serve_html_browser", fake_serve)

    runner = CliRunner()
    result = runner.invoke(
        results_cli.results,
        [
            "viewer",
            "-G",
            "my-benchmark",
            "-S",
            "olmobase:math",
            "--host",
            "0.0.0.0",
            "--port",
            "9900",
            "--repeated-runs",
            "--no-require-full-coverage",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["initial_group"] == "my-benchmark"
    assert captured["initial_scope_key"] == "suite::olmobase:math"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9900
    assert captured["margin"] == 0.0
    assert captured["keep_all"] is True
    assert captured["require_full_coverage"] is False


def test_results_cli_no_longer_registers_pairwise() -> None:
    """The old `results pairwise` entrypoint should be gone."""
    results_cli = importlib.import_module("olmo_eval.cli.results")

    runner = CliRunner()
    result = runner.invoke(results_cli.results, ["pairwise"])

    assert result.exit_code != 0
    assert "No such command 'pairwise'." in result.output


def test_results_cli_no_longer_has_pairwise_module() -> None:
    """The viewer command should not be implemented behind a `results.pairwise` module."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("olmo_eval.cli.results.pairwise")


def test_results_viewer_csv_dump_streams_to_stdout(monkeypatch) -> None:
    """CSV dump mode should still stream pairwise rows from `results viewer`."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
    viewer_cli = importlib.import_module("olmo_eval.cli.results.viewer")

    monkeypatch.setattr(
        analysis_pairwise,
        "compute_pairwise",
        lambda **kwargs: _build_pairwise_result(),
    )
    monkeypatch.setattr(viewer_cli, "get_database_session", lambda *args: _DummyDB())

    runner = CliRunner()
    result = runner.invoke(
        results_cli.results,
        [
            "viewer",
            "--model",
            "model-",
            "--suite",
            "olmobase:math",
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        "model_a,model_b,wins_a,wins_b,ties,n_contested,win_rate_a,"
        "win_rate_b,se,var_paired_diff,var_marginal_sum"
    ) in result.output


def test_build_results_table_keep_all_preserves_distinct_reruns(monkeypatch) -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    earlier = SimpleNamespace(
        id=10,
        model_name="model-a",
        model_hash="abc12345deadbeef",
        timestamp=datetime(2026, 4, 21, 8, 0, tzinfo=UTC),
    )
    later = SimpleNamespace(
        id=11,
        model_name="model-a",
        model_hash="abc12345deadbeef",
        timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
    )

    monkeypatch.setattr(
        viewer_server,
        "_group_experiments",
        lambda session, group_name, keep_all: [later, earlier] if keep_all else [later],
    )

    latest_table = viewer_server._build_results_table(
        _StaticTaskSession(
            [
                (
                    11,
                    "gsm8k:olmo3base",
                    "task-hash-11",
                    {"accuracy": {"exact_match": 0.65}},
                    "accuracy:exact_match",
                )
            ]
        ),
        "my-group",
        keep_all=False,
    )
    all_runs_table = viewer_server._build_results_table(
        _StaticTaskSession(
            [
                (
                    10,
                    "gsm8k:olmo3base",
                    "task-hash-10",
                    {"accuracy": {"exact_match": 0.55}},
                    "accuracy:exact_match",
                ),
                (
                    11,
                    "gsm8k:olmo3base",
                    "task-hash-11",
                    {"accuracy": {"exact_match": 0.65}},
                    "accuracy:exact_match",
                ),
            ]
        ),
        "my-group",
        keep_all=True,
    )

    assert len(latest_table["models"]) == 1
    assert len(all_runs_table["models"]) == 2
    assert (
        all_runs_table["models"][0]["display_label"] != all_runs_table["models"][1]["display_label"]
    )
    assert "2026-04-21 12:00" in all_runs_table["models"][0]["display_label"]
    assert "2026-04-21 08:00" in all_runs_table["models"][1]["display_label"]
    assert (
        viewer_server._model_key(all_runs_table["models"][0])
        == "abc12345deadbeef|2026-04-21T12:00:00+00:00"
    )
    assert (
        viewer_server._model_key(all_runs_table["models"][1])
        == "abc12345deadbeef|2026-04-21T08:00:00+00:00"
    )


def test_build_results_table_metric_options_do_not_double_count_primary_metric(
    monkeypatch,
) -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    experiment = SimpleNamespace(
        id=11,
        model_name="model-a",
        model_hash="abc12345deadbeef",
        timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
    )

    monkeypatch.setattr(
        viewer_server,
        "_group_experiments",
        lambda session, group_name, keep_all: [experiment],
    )

    results_table = viewer_server._build_results_table(
        _StaticTaskSession(
            [
                (
                    11,
                    "gsm8k:olmo3base",
                    "task-hash-11",
                    {
                        "accuracy": {"exact_match": 0.65},
                        "f1": {"exact_match": 0.72},
                    },
                    "accuracy:exact_match",
                )
            ]
        ),
        "my-group",
        keep_all=False,
    )

    metric_options = results_table["task_columns"][0]["metric_options"]
    metric_option_by_value = {option["value"]: option for option in metric_options}

    assert metric_option_by_value["accuracy:exact_match"]["model_count"] == 1
    assert metric_option_by_value["accuracy:exact_match"]["meta"] == "1 model"
    assert metric_option_by_value["f1:exact_match"]["model_count"] == 1


def test_build_results_table_splits_same_name_tasks_by_hash(monkeypatch) -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    experiment = SimpleNamespace(
        id=11,
        model_name="model-a",
        model_hash="abc12345deadbeef",
        timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
    )

    monkeypatch.setattr(
        viewer_server,
        "_group_experiments",
        lambda session, group_name, keep_all: [experiment],
    )

    results_table = viewer_server._build_results_table(
        _StaticTaskSession(
            [
                (
                    11,
                    "gsm8k:olmo3base",
                    "task-hash-alpha",
                    {"accuracy": {"exact_match": 0.65}},
                    "accuracy:exact_match",
                ),
                (
                    11,
                    "gsm8k:olmo3base",
                    "task-hash-beta",
                    {"accuracy": {"exact_match": 0.55}},
                    "accuracy:exact_match",
                ),
            ]
        ),
        "my-group",
        keep_all=False,
    )

    assert len(results_table["task_columns"]) == 2
    assert {column["id"] for column in results_table["task_columns"]} == {
        "task-hash-alpha",
        "task-hash-beta",
    }
    assert all("[" in column["full_label"] for column in results_table["task_columns"])
    assert results_table["models"][0]["task_scores"]["task-hash-alpha"] == pytest.approx(0.65)
    assert results_table["models"][0]["task_scores"]["task-hash-beta"] == pytest.approx(0.55)


def test_model_filter_score_label_uses_selected_scope_columns() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    results_table = {
        "task_columns": [
            {
                "id": "gsm8k:olmo3base",
                "score_display_format": "percentage",
                "score_unit": "proportion",
                "higher_is_better": True,
            },
            {
                "id": "minerva_math_algebra:olmo3base",
                "score_display_format": "percentage",
                "score_unit": "proportion",
                "higher_is_better": True,
            },
            {
                "id": "truthfulqa:mc:olmo3base",
                "score_display_format": "percentage",
                "score_unit": "proportion",
                "higher_is_better": True,
            },
        ]
    }
    selected_scope = {
        "task_ids": [
            "gsm8k:olmo3base",
            "minerva_math_algebra:olmo3base",
        ]
    }
    model = {
        "task_scores": {
            "gsm8k:olmo3base": 0.60,
            "minerva_math_algebra:olmo3base": 0.40,
            "truthfulqa:mc:olmo3base": 1.00,
        }
    }

    scoped_columns = viewer_server._scoped_task_columns(results_table, selected_scope)

    assert viewer_server._model_filter_score_label(model, scoped_columns) == "50.0%"


def test_model_filter_score_label_formats_raw_metrics_and_hides_mixed_units() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    raw_columns = [
        {
            "id": "ds1000:bpb",
            "score_display_format": "raw",
            "score_unit": "bits_per_byte",
            "higher_is_better": False,
        },
        {
            "id": "bigcodebench:bpb",
            "score_display_format": "raw",
            "score_unit": "bits_per_byte",
            "higher_is_better": False,
        },
    ]
    mixed_columns = [
        *raw_columns,
        {
            "id": "gsm8k:olmo3base",
            "score_display_format": "percentage",
            "score_unit": "proportion",
            "higher_is_better": True,
        },
    ]
    model = {
        "task_scores": {
            "ds1000:bpb": 0.41,
            "bigcodebench:bpb": 0.57,
            "gsm8k:olmo3base": 0.80,
        }
    }

    assert viewer_server._model_filter_score_label(model, raw_columns) == "0.5"
    assert viewer_server._model_filter_score_label(model, mixed_columns) == "—"


def test_results_viewer_rejects_removed_plot_format() -> None:
    """Static plot mode has been removed in favor of the viewer."""
    results_cli = importlib.import_module("olmo_eval.cli.results")

    runner = CliRunner()
    result = runner.invoke(results_cli.results, ["viewer", "--format", "plot"])

    assert result.exit_code != 0
    assert "'plot' is not one of" in result.output


def test_results_viewer_rejects_removed_html_export_format() -> None:
    """The viewer no longer exposes a standalone HTML export mode."""
    results_cli = importlib.import_module("olmo_eval.cli.results")

    runner = CliRunner()
    result = runner.invoke(results_cli.results, ["viewer", "--format", "html"])

    assert result.exit_code != 0
    assert "'html' is not one of" in result.output


def test_timed_value_cache_reuses_fresh_entries_and_expires_stale_ones() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    now = [100.0]
    cache = viewer_server._TimedValueCache(ttl_seconds=5.0, clock=lambda: now[0])
    calls = {"count": 0}

    def loader() -> dict[str, int]:
        calls["count"] += 1
        return {"value": calls["count"]}

    first = cache.get_or_set("groups", loader)
    second = cache.get_or_set("groups", loader)
    now[0] += 6.0
    third = cache.get_or_set("groups", loader)

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert third == {"value": 2}
    assert calls["count"] == 2


def test_viewer_scope_pickers_require_explicit_selection() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    groups = [
        {"name": "alpha-benchmark", "models": 2, "tasks": 3},
        {"name": "beta-benchmark", "models": 4, "tasks": 5},
    ]
    group_data = {
        "scope_options": [
            {
                "key": "suite::olmobase:math",
                "kind": "suite",
                "label": "olmobase:math",
                "value": "olmobase:math",
                "task_ids": ["gsm8k:olmo3base"],
            },
            {
                "key": "task::gsm8k:olmo3base",
                "kind": "task",
                "label": "gsm8k",
                "value": "gsm8k:olmo3base",
                "task_ids": ["gsm8k:olmo3base"],
            },
        ]
    }

    assert viewer_server._pick_group(groups, None) is None
    assert viewer_server._pick_group(groups, "alpha") == "alpha-benchmark"
    assert viewer_server._pick_group(groups, "missing") is None

    assert viewer_server._pick_scope(group_data, None) is None
    assert viewer_server._pick_scope(group_data, "suite::olmobase:math") == ("suite::olmobase:math")
    assert viewer_server._pick_scope(group_data, "suite::missing") is None


def test_serialize_viewer_export_supports_csv_and_json() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    filename, content_type, body = viewer_server._serialize_viewer_export(
        kind="instance-results",
        format_name="csv",
        base_name="my-group-my-scope",
        metadata={"group_name": "my-group"},
        rows=[
            {
                "model_display_rank": 1,
                "model_label": "model-a",
                "model_name": "model-a",
                "model_hash": "abc12345",
                "timestamp": "2026-04-20T00:00:00+00:00",
                "task_name": "gsm8k:olmo3base",
                "native_id": "doc-1",
                "task_metric_key": "accuracy:exact_match",
                "raw_score": 1.0,
                "comparison_score": 1.0,
                "score_display_format": "percentage",
                "score_unit": "proportion",
                "score_higher_is_better": True,
            }
        ],
    )

    assert filename == "my-group-my-scope-instance-results.csv"
    assert content_type == "text/csv; charset=utf-8"
    assert body.decode("utf-8").startswith(
        "model_display_rank,model_label,model_name,model_hash,timestamp,"
    )

    filename, content_type, body = viewer_server._serialize_viewer_export(
        kind="stored-files",
        format_name="json",
        base_name="my-group-my-scope",
        metadata={"group_name": "my-group"},
        rows=[
            {
                "model_display_rank": 1,
                "task_name": "gsm8k:olmo3base",
                "predictions_file": "s3://bucket/predictions.jsonl",
            }
        ],
    )

    assert filename == "my-group-my-scope-stored-files.json"
    assert content_type == "application/json; charset=utf-8"
    assert '"predictions_file": "s3://bucket/predictions.jsonl"' in body.decode("utf-8")


def test_render_results_viewer_page_renders_core_viewer_state_and_controls() -> None:
    """The browser page should expose core viewer state without CSS-level snapshot checks."""
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[
            {
                "name": "my-benchmark",
                "models": 4,
                "tasks": 3,
            }
        ],
        selected_group="my-benchmark",
        group_data={
            "summary": {"group_name": "my-benchmark"},
            "scope_options": [
                {
                    "key": "suite::olmobase:math",
                    "kind": "suite",
                    "label": "olmobase:math",
                    "value": "olmobase:math",
                    "task_ids": [
                        "gsm8k:olmo3base",
                        "minerva_math_algebra:olmo3base",
                    ],
                },
                {
                    "key": "task::gsm8k:olmo3base",
                    "kind": "task",
                    "label": "gsm8k",
                    "value": "gsm8k:olmo3base",
                    "task_ids": ["gsm8k:olmo3base"],
                },
            ],
            "results_table": {
                "models": [
                    {
                        "index": 0,
                        "display_label": "Qwen/Qwen3-8B",
                        "model_name": "Qwen/Qwen3-8B",
                        "model_hash": "abc12345",
                        "avg_score": 0.515,
                        "task_scores": {
                            "gsm8k:olmo3base": 0.57,
                            "minerva_math_algebra:olmo3base": 0.46,
                            "truthfulqa:mc:olmo3base": 0.61,
                        },
                    },
                    {
                        "index": 1,
                        "display_label": "Qwen/Qwen2.5-7B",
                        "model_name": "Qwen/Qwen2.5-7B",
                        "model_hash": "def67890",
                        "avg_score": 0.595,
                        "task_scores": {
                            "gsm8k:olmo3base": 0.63,
                            "minerva_math_algebra:olmo3base": 0.54,
                            "truthfulqa:mc:olmo3base": 0.61,
                        },
                    },
                ],
                "task_columns": [
                    {
                        "id": "gsm8k:olmo3base",
                        "label": "gsm8k",
                        "full_label": "gsm8k:olmo3base",
                        "model_count": 2,
                    },
                    {
                        "id": "minerva_math_algebra:olmo3base",
                        "label": "minerva math algebra",
                        "full_label": "minerva_math_algebra:olmo3base",
                        "model_count": 2,
                    },
                    {
                        "id": "truthfulqa:mc:olmo3base",
                        "label": "truthfulqa mc",
                        "full_label": "truthfulqa:mc:olmo3base",
                        "model_count": 2,
                    },
                ],
            },
        },
        selected_scope_key="suite::olmobase:math",
        selected_run_mode="repeated",
        pairwise_data={
            "meta": {
                "scope_label": "olmobase:math",
                "scope_kind": "suite",
                "task_count": 2,
                "shared_n": 6252,
                "mde80": 0.017,
                "mde80_by_alpha": {
                    "0.1": 0.015,
                    "0.05": 0.017,
                    "0.01": 0.022,
                    "0.001": 0.03,
                },
            }
        },
        pairwise_error=None,
    )

    payload = _extract_browser_payload(html)

    assert payload["has_groups"] is True
    assert payload["selected_group"] == "my-benchmark"
    assert payload["selected_scope_key"] == "suite::olmobase:math"
    assert payload["selected_metric"] is None
    assert payload["selected_run_mode"] == "repeated"
    assert payload["pairwise_error"] is None
    assert payload["pairwise_error_details"] is None
    assert payload["pairwise_data"]["meta"]["shared_n"] == 6252
    assert payload["pairwise_data"]["meta"]["mde80"] == 0.017
    assert payload["group_data"]["results_table"]["task_columns"][0]["id"] == "gsm8k:olmo3base"

    assert "<title>olmo-eval results viewer</title>" in html
    assert "Results viewer" in html
    assert 'data-search-select="group"' in html
    assert 'data-search-select="scope"' in html
    assert 'placeholder="search groups..."' in html
    assert 'placeholder="search suites or tasks..."' in html
    assert "olmobase:math (2 tasks) · N=6252" in html
    assert "MDE80" in html
    assert 'id="model-filter-summary"' in html
    assert 'data-action="toggle-model-checkbox"' in html
    assert 'data-model-key="abc12345"' in html

    for action in (
        "export-pairwise-csv",
        "export-pairwise-json",
        "export-pairwise-instance-results",
        "export-pairwise-stored-files",
        "export-pairwise-all",
    ):
        assert f'data-action="{action}"' in html

    assert "paired test" in html
    assert "Δ (row − col)" in html
    assert "P(row > col)" in html
    assert 'scopeForm?.addEventListener("submit"' in html
    assert 'document.body.classList.add("is-page-loading");' in html
    assert 'scopeForm.classList.add("is-loading");' in html
    assert 'id="run-mode-select"' in html
    assert 'name="runs"' in html
    assert '<option value="repeated" selected="selected">' in html
    assert 'id="scope-loading"' not in html
    assert "scope-status" not in html
    assert "scope-spinner" not in html
    assert '<label id="alpha-control"' not in html
    assert "renderDiscovery" not in html
    assert 'id="metric-select"' not in html


def test_render_results_viewer_page_shows_root_default_selection_state() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[{"name": "my-benchmark", "models": 4, "tasks": 3}],
        selected_group=None,
        group_data=None,
        selected_scope_key=None,
        selected_metric=None,
        pairwise_data=None,
        pairwise_error=None,
    )

    payload = _extract_browser_payload(html)

    assert payload["has_groups"] is True
    assert payload["selected_group"] is None
    assert payload["selected_scope_key"] is None
    assert payload["group_data"] is None
    assert payload["pairwise_data"] is None
    assert "select group..." in html
    assert "nothing to compare yet" in html
    assert "pick an experiment group and suite or task" in html
    assert "use the selectors above to choose what you want to compare." in html


def test_render_results_viewer_page_leaves_scope_unselected_without_request() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[{"name": "my-benchmark", "models": 4, "tasks": 3}],
        selected_group="my-benchmark",
        group_data={
            "summary": {"group_name": "my-benchmark"},
            "scope_options": [
                {
                    "key": "suite::olmobase:math",
                    "kind": "suite",
                    "label": "olmobase:math",
                    "value": "olmobase:math",
                    "task_ids": [
                        "gsm8k:olmo3base",
                        "minerva_math_algebra:olmo3base",
                    ],
                },
                {
                    "key": "task::gsm8k:olmo3base",
                    "kind": "task",
                    "label": "gsm8k",
                    "value": "gsm8k:olmo3base",
                    "task_ids": ["gsm8k:olmo3base"],
                },
            ],
            "results_table": {
                "models": [],
                "task_columns": [],
            },
        },
        selected_scope_key=None,
        selected_metric=None,
        pairwise_data=None,
        pairwise_error=None,
    )

    payload = _extract_browser_payload(html)

    assert payload["selected_group"] == "my-benchmark"
    assert payload["selected_scope_key"] is None
    assert payload["group_data"]["summary"]["group_name"] == "my-benchmark"
    assert "select suite or task..." in html
    assert "pick a suite or task to open the paired-test view." in html


def test_render_results_viewer_page_embeds_structured_pairwise_error() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[{"name": "olmo-3-parity-apr5", "models": 8, "tasks": 13}],
        selected_group="olmo-3-parity-apr5",
        group_data={
            "summary": {"group_name": "olmo-3-parity-apr5"},
            "scope_options": [
                {
                    "key": "suite::mmlu:humanities:mc:olmo3base",
                    "kind": "suite",
                    "label": "mmlu:humanities:mc:olmo3base",
                    "value": "mmlu:humanities:mc:olmo3base",
                    "task_ids": ["task-a", "task-b"],
                }
            ],
            "results_table": {
                "models": [
                    {
                        "index": 0,
                        "display_label": "allenai/OLMo-3-1025-7B",
                        "model_name": "allenai/OLMo-3-1025-7B",
                        "model_hash": "abc12345",
                        "task_scores": {"task-a": 0.71},
                    },
                    {
                        "index": 1,
                        "display_label": "allenai/OLMo-2-0425-1B",
                        "model_name": "allenai/OLMo-2-0425-1B",
                        "model_hash": "def67890",
                        "task_scores": {"task-a": 0.49},
                    },
                ],
                "task_columns": [
                    {
                        "id": "task-a",
                        "label": "task a",
                        "full_label": "task-a",
                        "model_count": 2,
                    },
                    {
                        "id": "task-b",
                        "label": "task b",
                        "full_label": "task-b",
                        "model_count": 1,
                    },
                ],
            },
        },
        selected_scope_key="suite::mmlu:humanities:mc:olmo3base",
        pairwise_data=None,
        pairwise_error="Only 1 experiment(s) matched the filters — need at least 2.",
        pairwise_error_details={
            "code": "insufficient_matched_experiments",
            "summary": "Only 1 run matched the paired-test requirements for this scope.",
            "message": "Only 1 experiment(s) matched the filters — need at least 2.",
            "scope_label": "mmlu:humanities:mc:olmo3base",
            "filter_summary": (
                "groups=['olmo-3-parity-apr5'], suite='mmlu:humanities:mc:olmo3base' (13 tasks)"
            ),
            "counts": [
                {"label": "matched runs", "value": 1},
                {"label": "minimum required", "value": 2},
            ],
            "matched_runs": [
                {
                    "label": "allenai/OLMo-3-1025-7B (abc12345)",
                    "timestamp_label": "2026-04-21 08:00",
                }
            ],
            "notes": ["The paired test only uses runs that match the selected scope."],
            "suggestions": ["Broaden the filters or choose a narrower task."],
        },
    )

    payload = _extract_browser_payload(html)

    assert payload["selected_group"] == "olmo-3-parity-apr5"
    assert payload["selected_scope_key"] == "suite::mmlu:humanities:mc:olmo3base"
    assert payload["pairwise_data"] is None
    assert payload["pairwise_error"] == (
        "Only 1 experiment(s) matched the filters — need at least 2."
    )
    assert payload["pairwise_error_details"]["code"] == "insufficient_matched_experiments"
    assert payload["pairwise_error_details"]["matched_runs"] == [
        {
            "label": "allenai/OLMo-3-1025-7B (abc12345)",
            "timestamp_label": "2026-04-21 08:00",
        }
    ]
    assert "runs that matched this scope" in html
    assert "The results tab is broader" in html
    assert "what to do next" in html
    assert "raw detail" not in html


def test_render_results_viewer_page_renders_metric_selector_for_recoverable_scope() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[{"name": "olmo-eval-external", "models": 6, "tasks": 1}],
        selected_group="olmo-eval-external",
        group_data={
            "summary": {"group_name": "olmo-eval-external"},
            "scope_options": [
                {
                    "key": "task::terminal_bench_2",
                    "kind": "task",
                    "label": "terminal_bench_2 · 6 models",
                    "value": "terminal_bench_2",
                    "task_ids": ["terminal_bench_2"],
                    "default_metric": "",
                    "metric_options": [
                        {
                            "value": "pass^1:external",
                            "label": "pass^1:external",
                            "model_count": 6,
                            "meta": "6 models",
                        },
                        {
                            "value": "reward:external",
                            "label": "reward:external",
                            "model_count": 6,
                            "meta": "6 models",
                        },
                    ],
                }
            ],
            "results_table": {"models": [], "task_columns": []},
        },
        selected_scope_key="task::terminal_bench_2",
        selected_metric=None,
        pairwise_data=None,
        pairwise_error="'terminal_bench_2' does not define a default metric for the paired test.",
        pairwise_error_details={
            "code": "missing_primary_metric",
            "summary": "'terminal_bench_2' does not define a default metric for the paired test.",
            "scope_label": "terminal_bench_2",
            "notes": [],
            "suggestions": [],
            "counts": [],
            "matched_runs": [],
            "compared_models": [],
            "dropped_duplicate_runs": [],
            "dropped_partial_coverage_models": [],
            "scored_models": [],
            "unscored_models": [],
            "unsupported_task_metrics": [],
            "per_model_instance_counts": [],
            "filter_summary": None,
            "message": "'terminal_bench_2' does not define a default metric for the paired test.",
        },
    )

    payload = _extract_browser_payload(html)

    assert payload["selected_scope_key"] == "task::terminal_bench_2"
    assert payload["pairwise_error_details"]["code"] == "missing_primary_metric"
    assert 'id="metric-select"' in html
    assert 'name="metric"' in html
    assert "select metric..." in html

    for metric in ("pass^1:external", "reward:external"):
        assert f'value="{metric}"' in html

    assert "choose a metric to retry this paired test" in html
    assert "Use the metric control above to choose a metric and retry the paired test." in html


def test_render_results_viewer_page_shows_scope_readiness_states() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    html = viewer_server.render_results_viewer_page(
        groups=[{"name": "olmo-3-parity-apr5", "models": 8, "tasks": 13}],
        selected_group="olmo-3-parity-apr5",
        group_data={
            "summary": {"group_name": "olmo-3-parity-apr5"},
            "scope_options": [
                {
                    "key": "suite::ready-suite",
                    "kind": "suite",
                    "label": "ready-suite · 13/13",
                    "value": "ready-suite",
                    "task_ids": ["task-a", "task-b"],
                    "status_badge": "ready",
                    "status_tone": "ready",
                    "supporting_text": "paired test ready with 3 latest models",
                    "title_suffix": "paired test ready now with 3 latest models",
                },
                {
                    "key": "suite::needs-coverage",
                    "kind": "suite",
                    "label": "needs-coverage · 11/13",
                    "value": "needs-coverage",
                    "task_ids": ["task-c"],
                    "status_badge": "needs coverage",
                    "status_tone": "limited",
                    "supporting_text": (
                        "needs coverage: 2 suite tasks are still missing in this group"
                    ),
                    "title_suffix": "click to see what still needs to run",
                },
            ],
            "results_table": {"models": [], "task_columns": []},
        },
        selected_scope_key="suite::ready-suite",
        pairwise_data=None,
        pairwise_error=None,
    )

    payload = _extract_browser_payload(html)

    assert payload["selected_scope_key"] == "suite::ready-suite"
    assert payload["group_data"]["scope_options"][0]["status_badge"] == "ready"
    assert payload["group_data"]["scope_options"][1]["status_badge"] == "needs coverage"
    assert 'data-value="suite::ready-suite"' in html
    assert 'data-value="suite::needs-coverage"' in html
    assert "paired test ready with 3 latest models" in html
    assert "needs coverage: 2 suite tasks are still missing in this group" in html


def test_viewer_pairwise_error_payload_rewrites_missing_primary_metric_for_ui() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")

    payload = viewer_server._viewer_pairwise_error_payload(
        ValueError(
            "No primary_metric set for task 'codex_humaneval:olmo3base' — "
            "specify --metric explicitly"
        ),
        selected_group="olmo-3-parity-apr5",
    )

    assert payload["code"] == "missing_primary_metric"
    assert payload["summary"] == (
        "'codex_humaneval:olmo3base' does not define a default metric for the paired test."
    )
    assert "--metric" not in payload["summary"]
    assert any("Choose another task or suite" in item for item in payload["suggestions"])


def test_viewer_pairwise_error_payload_rewrites_structured_cli_suggestions() -> None:
    viewer_server = importlib.import_module("olmo_eval.cli.results.viewer_server")
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")

    error = analysis_pairwise.PairwiseEligibilityError(
        code="insufficient_matched_experiments",
        summary="Only 1 run matched the paired-test requirements for this scope.",
        suggestions=[
            "Broaden the filters or choose a scope that at least two runs completed.",
            "Run `olmo-eval results group foo` to inspect which models are present.",
        ],
    )

    payload = viewer_server._viewer_pairwise_error_payload(
        error,
        selected_group="foo",
    )

    assert payload["summary"] == "Only 1 run matched the paired-test requirements for this scope."
    assert any("Switch to the Results tab" in item for item in payload["suggestions"])
    assert not any("olmo-eval results group" in item for item in payload["suggestions"])

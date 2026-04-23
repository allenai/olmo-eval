"""Tests for the results viewer CLI."""

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


def test_results_viewer_json_blob_forwards_exclude_filters(monkeypatch) -> None:
    """JSON dump mode should stream a blob and thread exclusions into compute_pairwise."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
    pairwise_cli = importlib.import_module("olmo_eval.cli.results.pairwise")

    captured: dict[str, object] = {}

    def fake_compute_pairwise(**kwargs):
        captured.update(kwargs)
        return _build_pairwise_result()

    monkeypatch.setattr(analysis_pairwise, "compute_pairwise", fake_compute_pairwise)
    monkeypatch.setattr(pairwise_cli, "get_database_session", lambda *args: _DummyDB())

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


def test_results_viewer_dump_keep_all_status_uses_actual_flag_name(
    monkeypatch, tmp_path: Path
) -> None:
    """The keep-all summary should print the real CLI flag without spaces."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
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
        results_cli.results,
        [
            "viewer",
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


def test_results_viewer_starts_server(monkeypatch) -> None:
    """`results viewer` should start the local results viewer server."""
    results_cli = importlib.import_module("olmo_eval.cli.results")
    pairwise_cli = importlib.import_module("olmo_eval.cli.results.pairwise")

    captured: dict[str, object] = {}

    def fake_serve(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(pairwise_cli, "_serve_html_browser", fake_serve)

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
            "--all",
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


def test_results_viewer_csv_dump_streams_to_stdout(monkeypatch) -> None:
    """CSV dump mode should still stream pairwise rows from `results viewer`."""
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")
    results_cli = importlib.import_module("olmo_eval.cli.results")
    pairwise_cli = importlib.import_module("olmo_eval.cli.results.pairwise")

    monkeypatch.setattr(
        analysis_pairwise,
        "compute_pairwise",
        lambda **kwargs: _build_pairwise_result(),
    )
    monkeypatch.setattr(pairwise_cli, "get_database_session", lambda *args: _DummyDB())

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
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    now = [100.0]
    cache = pairwise_server._TimedValueCache(ttl_seconds=5.0, clock=lambda: now[0])
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


def test_render_pairwise_browser_page_uses_dimming_only_loading_state() -> None:
    """The browser page should dim during scope changes without a separate status pill."""
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    html = pairwise_server.render_pairwise_browser_page(
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

    assert 'scopeForm?.addEventListener("submit"' in html
    assert "function showScopeLoading()" in html
    assert 'document.body.classList.add("is-page-loading");' in html
    assert 'scopeForm.classList.add("is-loading");' in html
    assert 'id="scope-loading"' not in html
    assert "scope-status" not in html
    assert "scope-spinner" not in html
    assert "function filterSearchSelect(control, query)" in html
    assert "function orderedVisibleSearchOptions(control)" in html
    assert "function moveActiveSearchOption(control, step)" in html
    assert "function renderPairwiseError(errorDetails, errorMessage, groupData)" in html
    assert "function renderDiagnosticRunSection(title, items)" in html
    assert "function renderDiagnosticInstanceCounts(title, items)" in html
    assert 'const modelFilterDetails = document.getElementById("model-filter-details");' in html
    assert 'data-search-select="group"' in html
    assert 'data-search-select="scope"' in html
    assert 'data-role="search-select-filter"' in html
    assert 'data-action="select-search-option"' in html
    assert 'data-option-index="0"' in html
    assert 'placeholder="search groups..."' in html
    assert 'placeholder="search suites or tasks..."' in html
    assert 'type="hidden" name="group"' in html
    assert 'type="hidden" name="scope"' in html
    assert 'scopeForm?.querySelectorAll("[data-search-select]")' in html
    assert 'querySelectorAll("select[data-loading-label]")' not in html
    assert 'onchange="this.form.submit()"' not in html
    assert "olmobase:math (2 tasks) · N=6252" in html
    assert 'title="olmobase:math (2 tasks) · N=6252"' in html
    assert "MDE80" in html
    assert "legend-metric-value" in html
    assert "<title>olmo-eval results viewer</title>" in html
    assert "olmo-eval" in html
    assert "Results viewer" in html
    assert '<span class="legend-title legend-title-alpha">α</span>' in html
    assert ".legend-title-alpha {" in html
    assert "text-transform: none;" in html
    assert (
        html.index('<span class="legend-title">intensity</span>')
        < html.index("${alphaLegend()}")
        < html.index('<span class="legend-title">MDE80</span>')
    )
    assert "--app-gutter: clamp(18px, 3vw, 42px);" in html
    assert "--app-max: 1880px;" in html
    assert "max-width: min(100%, var(--app-max));" in html
    assert "margin-inline: auto;" in html
    assert "padding-inline: var(--app-gutter);" in html
    assert 'grid-template-areas: "brand scope filters";' in html
    assert "grid-template-columns: auto minmax(0, 1.45fr) minmax(0, 0.95fr);" in html
    assert (
        "grid-template-columns: minmax(240px, 1fr) minmax(360px, 1.45fr) "
        "minmax(300px, 1.1fr);" in html
    )
    assert "min-width: min(300px, 100%);" in html
    assert "@media (max-width: 1880px)" in html
    assert 'grid-template-areas:\n          "brand filters"\n          "scope scope";' in html
    assert "grid-template-columns: auto minmax(420px, 1fr);" in html
    assert "max-width: 560px;" in html
    assert "@media (max-width: 1380px)" in html
    assert (
        'grid-template-areas:\n          "brand"\n          "scope"\n          "filters";' in html
    )
    assert "grid-area: filters;" in html
    assert "max-width: none;" in html
    assert "@media (max-width: 820px)" in html
    assert "width: auto;" in html
    assert "min-width: max-content;" in html
    assert '<label id="alpha-control"' not in html
    assert 'const alphaControl = document.getElementById("alpha-control");' not in html
    assert 'document.getElementById("alpha-select")?.addEventListener("change"' not in html
    assert ".results-table th.th-task:last-child," in html
    assert ">group<" in html
    assert ">suite / task<" in html
    assert ">filter<" in html
    assert "filter model names..." in html
    assert "included models" in html
    assert 'id="model-filter-summary"' in html
    assert 'data-action="toggle-model-checkbox"' in html
    assert 'data-model-key="abc12345"' in html
    assert ".search-select-menu {" in html
    assert "left: 0;" in html
    assert "right: auto;" in html
    assert "width: max(100%, var(--search-select-menu-width, 320px));" in html
    assert "max-width: min(620px, calc(100vw - (2 * var(--app-gutter))));" in html
    assert ".group-select .search-select-menu," in html
    assert ".scope-select .search-select-menu {" in html
    assert "max-width: calc(100vw - (2 * var(--app-gutter)));" in html
    assert "function syncSearchSelectMenuWidth(control)" in html
    assert "function queueSearchSelectMenuWidthSync()" in html
    assert 'control.style.setProperty("--search-select-menu-width"' in html
    assert ".search-select-option {" in html
    assert "grid-template-columns: minmax(0, 1fr) auto;" in html
    assert ".search-select-option-main {" in html
    assert "overflow-wrap: anywhere;" in html
    assert "white-space: normal;" in html
    assert ".search-select-option-aside {" in html
    assert "flex-wrap: wrap;" in html
    assert ".group-select .search-select-option," in html
    assert ".scope-select .search-select-option {" in html
    assert ".group-select .search-select-option-aside," in html
    assert ".scope-select .search-select-option-aside {" in html
    assert "${colsSvg()} columns" in html
    assert "${downloadSvg()} csv" in html
    assert "${downloadSvg()} export" in html
    assert 'data-action="export-pairwise-csv"' in html
    assert 'data-action="export-pairwise-json"' in html
    assert ">data<" in html
    assert ">csv<" in html
    assert ">json<" in html
    assert "pairwise csv" not in html
    assert "pairwise json" not in html
    assert "pairwise data" not in html
    assert '"selected_scope_key":"suite::olmobase:math"' in html
    assert '"task_ids":["gsm8k:olmo3base","minerva_math_algebra:olmo3base"]' in html
    assert "toggle-row-checkbox" not in html
    assert html.index('<button id="view-matrix" class="tab">paired test</button>') < html.index(
        '<button id="view-table" class="tab">results</button>'
    )
    assert "renderDiscovery" not in html
    assert "selected group" not in html
    assert "suite coverage" not in html
    assert "pairwise browser" not in html
    assert "point the browser at a populated results database." not in html
    assert "select a suite or task to load a pairwise comparison." not in html
    assert "function scopedTaskColumns(resultsData)" in html
    assert "function showAverageColumn(columns)" in html
    assert "const scopedColumns = scopedTaskColumns(resultsData);" in html
    assert "${column.model_count} models ${sortArrow(column.id)}" not in html
    assert "const header = showAverage" in html
    assert "task ${sortArrow(column.id)}" not in html
    assert "left: calc(var(--table-idx-w) + var(--table-name-w));" in html
    assert ".results-table thead th.th-name.sortable:hover," in html
    assert ".results-table thead th.th-name::after," in html
    assert ".results-table tbody td.td-name:hover," in html
    assert 'class="td-name-main"' in html
    assert ".results-table tbody td.td-name:focus-within .td-name-main {" in html
    assert "--td-name-hide-space: 30px;" in html
    assert "height: 100%;" in html
    assert "top: 0;" in html
    assert "bottom: 0;" in html
    assert "left: 0;" in html
    assert "min-width: 100%;" in html
    assert "width: fit-content;" in html
    assert "padding: 7px 12px 7px calc(12px + var(--td-name-hide-space));" in html
    assert "pointer-events: none;" in html
    assert 'class="row-hdr-main"' in html
    assert ".row-hdr:hover .row-hdr-name," in html
    assert ".row-hdr:focus-within .row-hdr-name {" in html
    assert "--row-hdr-hide-space: 24px;" in html
    assert ".row-hdr:hover .row-hdr-main," in html
    assert "height: 100%;" in html
    assert "top: 0;" in html
    assert "bottom: 0;" in html
    assert "left: 0;" in html
    assert "min-width: 100%;" in html
    assert "padding: 0 12px 0 6px;" in html
    assert ".row-hdr:hover .row-hdr-score," in html
    assert ".row-hdr:hover .row-hdr-hide," in html
    assert ".results-table tbody tr:hover .td-name-hide," in html
    assert ".td-name-hide:hover," in html
    assert ".results-table tbody tr:hover td.td-idx," in html
    assert ".results-table tbody tr:hover td.td-name {" in html
    assert ".results-table tbody td.td-name:focus-within {" in html
    assert ".matrix-hdr-hide {" in html
    assert "opacity: 0.46;" in html
    assert "opacity: 0.58;" in html
    assert "color: var(--c-ink-40);" in html
    assert "background: transparent;" in html
    assert "max-width: min(72vw, 880px);" in html
    assert "pointer-events: none;" in html
    assert "align-items: flex-end;" in html
    assert "min-height: 18px;" in html
    assert 'class="th-inline"' in html
    assert ".results-table thead .th-inline," in html
    assert "z-index: 1;" in html
    assert "function sortSvg() {" in html
    assert 'class="sort-glyph ${stateClass}"' in html
    assert 'class="th-task sortable ${sortState.key === column.id ? "active" : ""}"' in html
    assert 'data-action="table-sort"' in html
    assert "visible mean" not in html
    assert 'title="mean across visible task columns"' in html
    assert "overflow-wrap: anywhere;" in html
    assert "scrollbar-gutter: stable both-edges;" in html
    assert "scrollbar-width: thin;" in html
    assert "scrollbar-color: var(--c-ink-40) var(--c-paper-2);" in html
    assert ".table-scroll::-webkit-scrollbar {" in html
    assert "height: 13px;" in html
    assert ".table-scroll::-webkit-scrollbar-thumb {" in html
    assert "border: 3px solid var(--c-paper-2);" in html
    assert 'class="table-scroll-shell"' in html
    assert 'data-role="results-scroll-region"' in html
    assert 'data-role="results-scrollbar-track"' in html
    assert 'class="table-xbar-label">scroll</span>' in html
    assert ".table-xbar[hidden] {" in html
    assert "display: none !important;" in html
    assert ".table-xbar-track {" in html
    assert ".table-xbar-thumb {" in html
    assert "function bindResultsScrollbars()" in html
    assert "const resultsScrollbarObservers = new WeakMap();" in html
    assert "let resultsScrollSyncQueued = false;" in html
    assert "function queueResultsScrollSync()" in html
    assert (
        'if (table && "ResizeObserver" in window && !resultsScrollbarObservers.has(shell)) {'
        in html
    )
    assert "const observer = new ResizeObserver(() => {" in html
    assert "function measureResultsScrollMetrics(region)" in html
    assert "function syncResultsScrollbars()" in html
    assert "const actualScrollWidth = region.scrollWidth || 0;" in html
    assert "const fallbackWidth =" in html
    assert "const contentWidth = actualScrollWidth || fallbackWidth;" in html
    assert "actualScrollWidth > 0" in html
    assert "if (maxScroll <= 1) {" in html
    assert "rail.hidden = false;" in html
    assert "Math.round(track.getBoundingClientRect().width) ||" in html
    assert "if (trackWidth <= 0) {" in html
    assert "const scrollLeft = clamp(region.scrollLeft, 0, maxScroll);" in html
    assert "window.setTimeout(queueResultsScrollSync, 80);" in html
    assert 'window.addEventListener("resize", queueResultsScrollSync);' in html
    assert "document.fonts?.ready?.then(() => {" in html
    assert 'document.body.classList.add("is-dragging-table-xbar");' in html
    assert ".search-select-option.is-active {" in html
    assert ".diag-card {" in html
    assert ".diag-count-grid {" in html
    assert 'event.key === "ArrowDown" || event.key === "ArrowUp"' in html
    assert (
        "option.style.order = String((needle ? matchRank * 1000 : 0) + "
        "searchOptionIndex(option));" in html
    )
    assert "activeSearchOption(control)" in html
    assert "if (modelFilterDetails?.open && !modelFilterDetails.contains(event.target)) {" in html
    assert "renderPairwiseMeta" not in html
    assert "paired test" in html
    assert "Δ (row − col)" in html
    assert "P(row > col)" in html
    assert "function cellSignalLevel" in html
    assert 'return `<span class="cell-signal sig-${level}" aria-hidden="true"></span>`;' in html
    assert ".cell-signal.sig-3 {" in html
    assert ".cell-signal-bar" not in html
    assert "alpha: 0.05," in html
    assert 'loadState("alpha", "0.05")' not in html
    assert 'storageBase + "alpha"' not in html
    assert "col-hdr-idx" not in html
    assert 'class="matrix-hdr-hide row-hdr-hide"' in html
    assert 'class="matrix-hdr-hide col-hdr-hide"' in html
    assert "background: currentColor;" in html
    assert 'const fg = lightness < 0.72 ? "var(--c-paper)" : "var(--c-ink-70)";' in html
    assert "fmtCellMeta" not in html
    assert "sigStars" not in html
    assert "value < 1e-4" in html
    assert 'toExponential(0).replace("e+", "e")' in html
    assert 'toLocaleString("en-US"' in html
    assert ".tt-menu-action {" in html
    assert "const hiddenCount = Math.max(0, scopedColumns.length - visibleCount);" in html
    assert 'class="tt-dd tt-cols-menu"' in html
    assert 'class="tt-menu-clear"' in html
    assert 'class="tt-menu-name-btn"' in html
    assert 'data-action="solo-col"' in html
    assert 'data-action="reset-cols"' in html
    assert "function bindColumnsMenu()" in html
    assert "state.columnsMenuOpen = true;" in html
    assert "if (colsMenu?.open && !colsMenu.contains(event.target)) {" in html
    assert "function buildPairwiseExportPayload(pairwiseData)" in html
    assert "function exportPairwiseCsv()" in html
    assert "function exportPairwiseJson()" in html
    assert "shared_instance_mean_score" in html
    assert "mean_task_score" in html
    assert "bt_elo" in html
    assert "pairwise_comparisons" in html
    assert "pairwise_matrices" in html
    assert '"pairwise_error_details":null' in html
    assert "score_difference_row_minus_column" in html
    assert "score_difference_display_format" in html
    assert "score_diff_row_minus_column" not in html
    assert "score_diff_display_format" not in html


def test_render_pairwise_browser_page_embeds_structured_pairwise_error() -> None:
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    html = pairwise_server.render_pairwise_browser_page(
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

    assert '"pairwise_error_details":{"code":"insufficient_matched_experiments"' in html
    assert '"matched_runs":[{"label":"allenai/OLMo-3-1025-7B (abc12345)"' in html
    assert "runs that matched this scope" in html
    assert "The results tab is broader" in html
    assert "what to do next" in html
    assert "raw detail" not in html


def test_render_pairwise_browser_page_renders_metric_selector_for_recoverable_scope() -> None:
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    html = pairwise_server.render_pairwise_browser_page(
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

    assert 'id="metric-select"' in html
    assert 'name="metric"' in html
    assert ">metric<" in html
    assert "select metric..." in html
    assert 'value="pass^1:external"' in html
    assert 'value="reward:external"' in html
    assert "choose a metric to retry this paired test" in html
    assert "Use the metric control above to choose a metric and retry the paired test." in html


def test_render_pairwise_browser_page_shows_scope_readiness_states() -> None:
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    html = pairwise_server.render_pairwise_browser_page(
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

    assert "search-select-option-state is-ready" in html
    assert "search-select-option-state is-limited" in html
    assert "search-select-option-sub" in html
    assert "paired test ready with 3 latest models" in html
    assert "needs coverage: 2 suite tasks are still missing in this group" in html


def test_viewer_pairwise_error_payload_rewrites_missing_primary_metric_for_ui() -> None:
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")

    payload = pairwise_server._viewer_pairwise_error_payload(
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
    pairwise_server = importlib.import_module("olmo_eval.cli.results.pairwise_server")
    analysis_pairwise = importlib.import_module("olmo_eval.analysis.pairwise")

    error = analysis_pairwise.PairwiseEligibilityError(
        code="insufficient_matched_experiments",
        summary="Only 1 run matched the paired-test requirements for this scope.",
        suggestions=[
            "Broaden the filters or choose a scope that at least two runs completed.",
            "Run `olmo-eval results group foo` to inspect which models are present.",
        ],
    )

    payload = pairwise_server._viewer_pairwise_error_payload(
        error,
        selected_group="foo",
    )

    assert payload["summary"] == "Only 1 run matched the paired-test requirements for this scope."
    assert any("Switch to the Results tab" in item for item in payload["suggestions"])
    assert not any("olmo-eval results group" in item for item in payload["suggestions"])

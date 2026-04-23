"""Local server for the results viewer UI."""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from textwrap import dedent
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import load_only, noload

from olmo_eval.analysis.pairwise import compute_pairwise, get_task_metric_profile
from olmo_eval.analysis.pairwise_html import build_pairwise_html_payload
from olmo_eval.analysis.pairwise_metrics import _build_task_label_lookup
from olmo_eval.analysis.pairwise_viewer.assets import (
    browser_css_text,
    browser_js_text,
    render_template,
    shared_css_text,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


_GROUP_LIST_CACHE_TTL_SECONDS = 15.0
_GROUP_BROWSER_CACHE_TTL_SECONDS = 15.0
_GROUP_BROWSER_CACHE_MAX_ENTRIES = 64


@dataclass(slots=True)
class _TimedCacheEntry:
    created_at: float
    value: Any


class _TimedValueCache:
    """Small in-process TTL cache for browser response building."""

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int | None = None,
        clock: Any = monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: dict[Any, _TimedCacheEntry] = {}
        self._lock = RLock()

    def get_or_set(self, key: Any, factory: Any) -> Any:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and now - entry.created_at <= self._ttl_seconds:
                return entry.value

        value = factory()

        with self._lock:
            self._entries[key] = _TimedCacheEntry(created_at=self._clock(), value=value)
            self._prune_locked()
            return value

    def _prune_locked(self) -> None:
        if not self._entries:
            return

        now = self._clock()
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if now - entry.created_at > self._ttl_seconds
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

        if self._max_entries is None or len(self._entries) <= self._max_entries:
            return

        overflow = len(self._entries) - self._max_entries
        oldest_keys = sorted(
            self._entries,
            key=lambda key: self._entries[key].created_at,
        )[:overflow]
        for key in oldest_keys:
            self._entries.pop(key, None)


def _make_scope_key(kind: str, value: str) -> str:
    return f"{kind}::{value}"


def _parse_scope_key(scope_key: str | None) -> tuple[str | None, str | None]:
    if not scope_key or "::" not in scope_key:
        return None, None
    kind, value = scope_key.split("::", 1)
    if kind not in {"suite", "task"} or not value:
        return None, None
    return kind, value


def _pick_group(groups: list[dict[str, Any]], requested: str | None) -> str | None:
    if not groups:
        return None
    if requested:
        exact = next((group["name"] for group in groups if group["name"] == requested), None)
        if exact:
            return exact
        prefix = next(
            (group["name"] for group in groups if group["name"].startswith(requested)),
            None,
        )
        if prefix:
            return prefix
    return groups[0]["name"]


def _pick_scope(group_data: dict[str, Any], requested: str | None) -> str | None:
    scope_options = group_data.get("scope_options", [])
    if not scope_options:
        return None
    if requested and any(option["key"] == requested for option in scope_options):
        return requested
    preferred_suite = next(
        (option["key"] for option in scope_options if option["kind"] == "suite"),
        None,
    )
    return preferred_suite or scope_options[0]["key"]


def _list_groups(session: Session, *, limit: int = 500) -> list[dict[str, Any]]:
    from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult

    group_rows = session.execute(
        select(
            Experiment.experiment_group,
            func.count(distinct(Experiment.id)).label("experiments"),
            func.count(distinct(Experiment.model_hash)).label("models"),
            func.max(Experiment.timestamp).label("most_recent"),
        )
        .group_by(Experiment.experiment_group)
        .order_by(func.max(Experiment.timestamp).desc())
        .limit(limit)
    ).all()

    task_count_map = {
        group_name: count
        for group_name, count in session.execute(
            select(
                Experiment.experiment_group,
                func.count(distinct(TaskResult.task_name)).label("tasks"),
            )
            .join(TaskResult, Experiment.id == TaskResult.experiment_pk)
            .group_by(Experiment.experiment_group)
        ).all()
    }

    return [
        {
            "name": group_name,
            "experiments": int(experiments or 0),
            "models": int(models or 0),
            "tasks": int(task_count_map.get(group_name, 0)),
            "most_recent": most_recent.isoformat() if most_recent is not None else None,
            "most_recent_label": most_recent.strftime("%Y-%m-%d %H:%M") if most_recent else "",
        }
        for group_name, experiments, models, most_recent in group_rows
    ]


def _latest_group_experiments(session: Session, group_name: str) -> list[Any]:
    from olmo_eval.storage.backends.postgres.models import Experiment

    experiments = (
        session.execute(
            select(Experiment)
            .options(
                load_only(
                    Experiment.id,
                    Experiment.model_name,
                    Experiment.model_hash,
                    Experiment.timestamp,
                ),
                noload(Experiment.task_results),
                noload(Experiment.instance_predictions),
            )
            .where(Experiment.experiment_group == group_name)
            .distinct(Experiment.model_hash)
            .order_by(Experiment.model_hash, Experiment.timestamp.desc())
        )
        .scalars()
        .all()
    )
    return sorted(experiments, key=lambda experiment: experiment.timestamp, reverse=True)


def _build_results_table(session: Session, group_name: str) -> dict[str, Any]:
    from olmo_eval.runners.processing.utils import extract_score_from_metrics
    from olmo_eval.storage.backends.postgres.models import TaskResult

    experiments = _latest_group_experiments(session, group_name)
    if not experiments:
        return {"models": [], "task_columns": []}

    experiments.sort(key=lambda experiment: experiment.timestamp, reverse=True)
    selected_pks = [experiment.id for experiment in experiments]
    task_rows = session.execute(
        select(
            TaskResult.experiment_pk,
            TaskResult.task_name,
            TaskResult.metrics,
            TaskResult.primary_metric,
        ).where(TaskResult.experiment_pk.in_(selected_pks))
    ).all()

    label_counts = Counter(experiment.model_name for experiment in experiments)
    experiment_labels = {
        experiment.id: (
            f"{experiment.model_name} ({experiment.model_hash[:8]})"
            if label_counts[experiment.model_name] > 1
            else experiment.model_name
        )
        for experiment in experiments
    }

    task_metric_by_name: dict[str, str | None] = {}
    task_profile_by_name: dict[str, Any] = {}
    model_count_by_task: dict[str, int] = {}
    task_scores_by_pk: dict[int, dict[str, float | None]] = {pk: {} for pk in selected_pks}
    for experiment_pk, task_name, metrics, primary_metric in task_rows:
        if primary_metric:
            task_metric_by_name.setdefault(task_name, primary_metric)
            task_profile_by_name.setdefault(
                task_name,
                get_task_metric_profile(task_name, primary_metric),
            )
        score = extract_score_from_metrics(metrics, primary_metric) if primary_metric else None
        task_scores_by_pk.setdefault(experiment_pk, {})[task_name] = score
        if task_name not in model_count_by_task:
            model_count_by_task[task_name] = 0
        if score is not None:
            model_count_by_task[task_name] += 1

    task_names = tuple(sorted({task_name for _, task_name, _, _ in task_rows}))
    task_label_lookup = _build_task_label_lookup(task_names)

    task_columns = [
        {
            "id": task_name,
            "label": task_label_lookup.get(task_name, task_name),
            "full_label": task_name,
            "metric": task_metric_by_name.get(task_name) or "",
            "score_display_format": (
                task_profile_by_name[task_name].display_format
                if task_name in task_profile_by_name and task_profile_by_name[task_name] is not None
                else "raw"
            ),
            "score_unit": (
                task_profile_by_name[task_name].unit
                if task_name in task_profile_by_name and task_profile_by_name[task_name] is not None
                else task_metric_by_name.get(task_name) or ""
            ),
            "higher_is_better": (
                task_profile_by_name[task_name].higher_is_better
                if task_name in task_profile_by_name and task_profile_by_name[task_name] is not None
                else True
            ),
            "model_count": int(model_count_by_task.get(task_name, 0)),
        }
        for task_name in task_names
    ]

    models: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        task_scores = task_scores_by_pk.get(experiment.id, {})
        scored_values = [score for score in task_scores.values() if score is not None]
        avg_score = sum(scored_values) / len(scored_values) if scored_values else None
        models.append(
            {
                "index": index,
                "display_label": experiment_labels[experiment.id],
                "model_name": experiment.model_name,
                "model_hash": experiment.model_hash,
                "timestamp": experiment.timestamp.isoformat(),
                "avg_score": avg_score,
                "task_scores": {task_name: task_scores.get(task_name) for task_name in task_names},
            }
        )

    return {
        "models": models,
        "task_columns": task_columns,
    }


def _build_group_browser_data(session: Session, group_name: str) -> dict[str, Any]:
    from olmo_eval.evals.suites.registry import get_suite, list_suites
    from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult

    experiments, models, first_ts, last_ts = session.execute(
        select(
            func.count(distinct(Experiment.id)).label("experiments"),
            func.count(distinct(Experiment.model_hash)).label("models"),
            func.min(Experiment.timestamp).label("first_ts"),
            func.max(Experiment.timestamp).label("last_ts"),
        ).where(Experiment.experiment_group == group_name)
    ).one()

    task_rows = [
        {
            "name": task_name,
            "models": int(model_count or 0),
            "metric": metric or "",
        }
        for task_name, model_count, metric in session.execute(
            select(
                TaskResult.task_name,
                func.count(distinct(Experiment.model_hash)).label("models"),
                func.max(TaskResult.primary_metric).label("metric"),
            )
            .join(Experiment, Experiment.id == TaskResult.experiment_pk)
            .where(Experiment.experiment_group == group_name)
            .group_by(TaskResult.task_name)
            .order_by(TaskResult.task_name)
        ).all()
    ]
    present_tasks = {row["name"] for row in task_rows}

    suite_rows: list[dict[str, Any]] = []
    for suite_name in list_suites():
        expanded_tasks = get_suite(suite_name).expanded_tasks
        visible_tasks = list(
            dict.fromkeys(task_name for task_name in expanded_tasks if task_name in present_tasks)
        )
        total = len(expanded_tasks)
        covered = len(visible_tasks)
        if covered == 0:
            continue
        ratio = covered / total if total else 0.0
        suite_rows.append(
            {
                "name": suite_name,
                "covered": covered,
                "total": total,
                "ratio": ratio,
                "task_ids": visible_tasks,
            }
        )
    suite_rows.sort(key=lambda row: (-row["ratio"], row["name"]))

    scope_options = [
        {
            "key": _make_scope_key("suite", suite_row["name"]),
            "kind": "suite",
            "label": f"{suite_row['name']} · {suite_row['covered']}/{suite_row['total']}",
            "value": suite_row["name"],
            "task_ids": list(suite_row["task_ids"]),
        }
        for suite_row in suite_rows
    ]
    scope_options.extend(
        {
            "key": _make_scope_key("task", task_row["name"]),
            "kind": "task",
            "label": f"{task_row['name']} · {task_row['models']} models",
            "value": task_row["name"],
            "task_ids": [task_row["name"]],
        }
        for task_row in task_rows
    )

    return {
        "summary": {
            "group_name": group_name,
            "experiments": int(experiments or 0),
            "models": int(models or 0),
            "tasks": len(task_rows),
            "first_ts": first_ts.isoformat() if first_ts is not None else None,
            "last_ts": last_ts.isoformat() if last_ts is not None else None,
            "first_label": first_ts.strftime("%Y-%m-%d") if first_ts else "",
            "last_label": last_ts.strftime("%Y-%m-%d") if last_ts else "",
        },
        "task_rows": task_rows,
        "suite_rows": suite_rows,
        "scope_options": scope_options,
        "results_table": _build_results_table(session, group_name),
    }


_PAIRWISE_SHARED_CSS = shared_css_text()


_BROWSER_EXTRA_CSS = browser_css_text()


_BROWSER_JS = browser_js_text()


def _clean_inline_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _render_search_select(
    *,
    control_name: str,
    label: str,
    field_name: str,
    selected_value: str | None,
    selected_label: str,
    placeholder: str,
    options: list[dict[str, str]],
    empty_label: str,
    disabled_label: str,
) -> str:
    if not options:
        return dedent(
            f"""
            <div class="select search-select {html.escape(control_name)}-select">
              <span class="select-label">{html.escape(label)}</span>
              <div class="control-summary is-disabled" title="{html.escape(disabled_label)}">
                <span class="control-summary-text">{html.escape(disabled_label)}</span>
              </div>
            </div>
            """
        ).strip()

    resolved_selected_value = _clean_inline_text(selected_value or options[0]["value"])
    resolved_selected_label = _clean_inline_text(
        selected_label
        or next(
            (
                option["summary_text"]
                for option in options
                if option["value"] == resolved_selected_value
            ),
            options[0]["summary_text"],
        )
    )
    control_name_html = html.escape(control_name)
    label_html = html.escape(label)
    field_name_html = html.escape(field_name)
    selected_value_html = html.escape(resolved_selected_value)
    selected_label_html = html.escape(resolved_selected_label)
    placeholder_html = html.escape(placeholder)
    empty_label_html = html.escape(empty_label)

    def _render_search_select_option(index: int, option: dict[str, str]) -> str:
        selected_class = " is-selected" if option["value"] == resolved_selected_value else ""
        meta_markup = ""
        if option["meta"]:
            meta_markup = (
                f'<span class="search-select-option-meta">{html.escape(option["meta"])}</span>'
            )
        return dedent(
            f"""
            <button
              type="button"
              class="search-select-option{selected_class}"
              data-action="select-search-option"
              data-role="search-select-option"
              data-value="{html.escape(option["value"])}"
              data-summary-text="{html.escape(option["summary_text"])}"
              data-filter-text="{html.escape(option["filter_text"])}"
              data-option-index="{index}"
              title="{html.escape(option["title"])}"
            >
              <span class="search-select-option-main">{html.escape(option["label"])}</span>
              {meta_markup}
            </button>
            """
        ).strip()

    rendered_options = "".join(
        _render_search_select_option(index, option) for index, option in enumerate(options)
    )

    return dedent(
        f"""
        <div
          class="select search-select {control_name_html}-select"
          data-search-select="{control_name_html}"
        >
          <span class="select-label">{label_html}</span>
          <input type="hidden" name="{field_name_html}" value="{selected_value_html}" />
          <details class="tt-dd control-dd search-select-dd">
            <summary
              class="control-summary search-select-summary"
              title="{selected_label_html}"
            >
              <span class="control-summary-text">{selected_label_html}</span>
            </summary>
            <div class="tt-menu search-select-menu">
              <div class="search-select-search">
                <input
                  type="search"
                  class="search-select-filter"
                  data-role="search-select-filter"
                  placeholder="{placeholder_html}"
                  aria-label="{label_html}"
                  autocomplete="off"
                  spellcheck="false"
                />
              </div>
              <div class="search-select-body">
                {rendered_options}
              </div>
              <div class="search-select-empty" data-role="search-select-empty" hidden>
                {empty_label_html}
              </div>
            </div>
          </details>
        </div>
        """
    ).strip()


def _scope_option_label(
    option: dict[str, Any],
    *,
    selected_scope_key: str | None,
    pairwise_data: dict[str, Any] | None,
) -> str:
    if option["key"] != selected_scope_key or pairwise_data is None:
        return str(option["label"])

    meta = pairwise_data.get("meta", {})
    scope_label = str(meta.get("scope_label") or option["value"])
    if meta.get("scope_kind") == "suite":
        task_count = int(meta.get("task_count") or 0)
        if task_count > 0:
            task_label = "task" if task_count == 1 else "tasks"
            scope_label = f"{scope_label} ({task_count} {task_label})"
    shared_n = meta.get("shared_n")
    if shared_n is not None:
        scope_label = f"{scope_label} · N={shared_n}"
    return scope_label


def _model_key(model: dict[str, Any]) -> str:
    return str(
        model.get("model_hash")
        or model.get("model_name")
        or model.get("display_label")
        or model.get("index")
        or ""
    )


def _format_model_avg(avg_score: Any) -> str:
    if isinstance(avg_score, (int, float)):
        return f"{float(avg_score) * 100:.1f}%"
    return "—"


def render_pairwise_browser_page(
    *,
    groups: list[dict[str, Any]],
    selected_group: str | None,
    group_data: dict[str, Any] | None,
    selected_scope_key: str | None,
    pairwise_data: dict[str, Any] | None,
    pairwise_error: str | None,
) -> str:
    """Render the viewer page with server-populated selectors and payload."""
    browser_payload = {
        "group_data": group_data,
        "selected_scope_key": selected_scope_key,
        "pairwise_data": pairwise_data,
        "pairwise_error": pairwise_error,
    }
    payload_json = json.dumps(browser_payload, separators=(",", ":")).replace("</", "<\\/")

    group_select_options = [
        {
            "value": _clean_inline_text(group["name"]),
            "label": _clean_inline_text(group["name"]),
            "summary_text": _clean_inline_text(group["name"]),
            "filter_text": _clean_inline_text(
                f"{group['name']} {group['models']} models {group.get('tasks', 0)} tasks"
            ),
            "meta": _clean_inline_text(
                f"{group['models']} models"
                + (f" · {group.get('tasks', 0)} tasks" if group.get("tasks") is not None else "")
            ),
            "title": _clean_inline_text(
                f"{group['name']} · {group['models']} models"
                + (f" · {group.get('tasks', 0)} tasks" if group.get("tasks") is not None else "")
            ),
        }
        for group in groups
    ]

    group_select = _render_search_select(
        control_name="group",
        label="group",
        field_name="group",
        selected_value=selected_group,
        selected_label=_clean_inline_text(selected_group or ""),
        placeholder="search groups...",
        options=group_select_options,
        empty_label="No groups match.",
        disabled_label="no groups found",
    )

    scope_select_options: list[dict[str, str]] = []
    selected_scope_label = ""
    if group_data:
        selected_scope_label = next(
            (
                _scope_option_label(
                    option,
                    selected_scope_key=selected_scope_key,
                    pairwise_data=pairwise_data,
                )
                for option in group_data["scope_options"]
                if option["key"] == selected_scope_key
            ),
            "",
        )
        scope_select_options = [
            {
                "value": _clean_inline_text(option["key"]),
                "label": _clean_inline_text(option["label"]),
                "summary_text": _clean_inline_text(
                    _scope_option_label(
                        option,
                        selected_scope_key=selected_scope_key,
                        pairwise_data=pairwise_data,
                    )
                    if option["key"] == selected_scope_key
                    else option["label"]
                ),
                "filter_text": _clean_inline_text(
                    f"{option['value']} {option['label']} {option['kind']}"
                ),
                "meta": _clean_inline_text(option["kind"]),
                "title": _clean_inline_text(
                    _scope_option_label(
                        option,
                        selected_scope_key=selected_scope_key,
                        pairwise_data=pairwise_data,
                    )
                    if option["key"] == selected_scope_key
                    else option["label"]
                ),
            }
            for option in group_data["scope_options"]
        ]

    model_filter_models: list[dict[str, Any]] = []
    if group_data and group_data.get("results_table"):
        model_filter_models = sorted(
            list(group_data["results_table"].get("models", [])),
            key=lambda model: str(model.get("display_label") or "").lower(),
        )

    model_filter_options = "".join(
        (
            '<label class="tt-menu-row">'
            f'<input type="checkbox" data-action="toggle-model-checkbox" '
            f'data-model-key="{html.escape(_model_key(model))}" checked />'
            f'<span class="tt-menu-name">'
            f"{html.escape(str(model.get('display_label') or '').replace(chr(10), ' '))}"
            "</span>"
            f'<span class="tt-menu-n">'
            f"{html.escape(_format_model_avg(model.get('avg_score')))}"
            "</span>"
            "</label>"
        )
        for model in model_filter_models
    )

    model_filter_total = len(model_filter_models)
    if model_filter_models:
        model_filter_control = f"""
        <div class="filter-block filter-block-models">
          <span class="select-label">models</span>
          <details id="model-filter-details" class="tt-dd control-dd model-filter">
            <summary class="control-summary">
              <span id="model-filter-summary" class="control-summary-text">all models</span>
              <span id="model-filter-count" class="tt-pill">{model_filter_total}</span>
            </summary>
            <div class="tt-menu tt-menu-models">
              <div class="tt-menu-head">
                <span>included models</span>
                <button
                  id="model-filter-reset"
                  type="button"
                  class="tt-menu-clear"
                  hidden
                >reset</button>
              </div>
              <div class="tt-menu-body">
                {model_filter_options}
              </div>
            </div>
          </details>
        </div>
        """
    else:
        model_filter_control = """
        <div class="filter-block filter-block-models">
          <span class="select-label">models</span>
          <div class="control-summary is-disabled">
            <span id="model-filter-summary" class="control-summary-text">none available</span>
            <span id="model-filter-count" class="tt-pill">0</span>
          </div>
        </div>
        """

    scope_select = _render_search_select(
        control_name="scope",
        label="suite / task",
        field_name="scope",
        selected_value=selected_scope_key,
        selected_label=_clean_inline_text(selected_scope_label),
        placeholder="search suites or tasks...",
        options=scope_select_options,
        empty_label="No suites or tasks match.",
        disabled_label="nothing to compare yet",
    )

    return (
        render_template(
            "browser.html",
            page_title="olmo-eval results viewer",
            styles=_PAIRWISE_SHARED_CSS + "\n" + _BROWSER_EXTRA_CSS,
            group_select=group_select,
            scope_select=scope_select,
            model_filter_control=model_filter_control,
            payload_json=payload_json,
            script=_BROWSER_JS,
        ).strip()
        + "\n"
    )


def serve_pairwise_browser(
    *,
    db: Any,
    host: str,
    port: int,
    initial_group: str | None,
    initial_scope_key: str | None,
    margin: float,
    keep_all: bool,
    require_full_coverage: bool,
) -> int:
    """Start the local results viewer server and block until interrupted."""
    groups_cache = _TimedValueCache(ttl_seconds=_GROUP_LIST_CACHE_TTL_SECONDS, max_entries=1)
    group_browser_cache = _TimedValueCache(
        ttl_seconds=_GROUP_BROWSER_CACHE_TTL_SECONDS,
        max_entries=_GROUP_BROWSER_CACHE_MAX_ENTRIES,
    )

    class PairwiseBrowserHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if parsed.path not in {"", "/"}:
                self.send_response(404)
                self.end_headers()
                return

            params = parse_qs(parsed.query)
            requested_group = params.get("group", [initial_group or ""])[0] or None
            requested_scope = params.get("scope", [initial_scope_key or ""])[0] or None

            with db.session() as session:
                groups = groups_cache.get_or_set(
                    ("groups", 500),
                    lambda: _list_groups(session),
                )
                selected_group = _pick_group(groups, requested_group)
                group_data = (
                    group_browser_cache.get_or_set(
                        selected_group,
                        lambda: _build_group_browser_data(session, selected_group),
                    )
                    if selected_group is not None
                    else None
                )
                selected_scope_key = (
                    _pick_scope(group_data, requested_scope) if group_data is not None else None
                )
                pairwise_data: dict[str, Any] | None = None
                pairwise_error: str | None = None

                scope_kind, scope_value = _parse_scope_key(selected_scope_key)
                if (
                    selected_group is not None
                    and scope_kind is not None
                    and scope_value is not None
                ):
                    try:
                        result = compute_pairwise(
                            session=session,
                            task_name=scope_value if scope_kind == "task" else None,
                            suite_name=scope_value if scope_kind == "suite" else None,
                            margin=margin,
                            experiment_groups=[selected_group],
                            keep_all=keep_all,
                            require_full_coverage=require_full_coverage,
                        )
                        pairwise_data = build_pairwise_html_payload(result)
                    except ValueError as error:
                        pairwise_error = str(error)

            page = render_pairwise_browser_page(
                groups=groups,
                selected_group=selected_group,
                group_data=group_data,
                selected_scope_key=selected_scope_key,
                pairwise_data=pairwise_data,
                pairwise_error=pairwise_error,
            )
            encoded = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), PairwiseBrowserHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return server.server_port

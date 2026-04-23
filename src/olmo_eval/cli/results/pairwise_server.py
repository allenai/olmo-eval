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

from olmo_eval.analysis.pairwise import (
    PairwiseEligibilityError,
    compute_pairwise,
    get_task_metric_profile,
)
from olmo_eval.analysis.pairwise_metrics import _build_task_label_lookup
from olmo_eval.analysis.pairwise_viewer.assets import (
    browser_css_text,
    browser_js_text,
    render_template,
    shared_css_text,
)
from olmo_eval.analysis.pairwise_viewer_payload import build_pairwise_viewer_payload

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
    task_metric_options_by_name: dict[str, Counter[str]] = {}
    model_count_by_task: dict[str, int] = {}
    task_scores_by_pk: dict[int, dict[str, float | None]] = {pk: {} for pk in selected_pks}
    for experiment_pk, task_name, metrics, primary_metric in task_rows:
        if primary_metric:
            task_metric_by_name.setdefault(task_name, primary_metric)
            task_profile_by_name.setdefault(
                task_name,
                get_task_metric_profile(task_name, primary_metric),
            )
        metric_counter = task_metric_options_by_name.setdefault(task_name, Counter())
        for metric_key in _available_metric_keys(metrics):
            metric_counter[metric_key] += 1
        if primary_metric:
            metric_counter[primary_metric] += 1
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
            "metric_options": _serialize_metric_options(
                task_metric_options_by_name.get(task_name, Counter()),
            ),
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


def _is_numeric_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _available_metric_keys(metrics: Any) -> set[str]:
    keys: set[str] = set()
    if not isinstance(metrics, dict):
        return keys
    for metric_name, scorer_values in metrics.items():
        if not isinstance(metric_name, str) or not isinstance(scorer_values, dict):
            continue
        for scorer_name in scorer_values:
            if isinstance(scorer_name, str):
                keys.add(f"{metric_name}:{scorer_name}")
    return keys


def _serialize_metric_options(metric_counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            "value": metric_key,
            "label": metric_key,
            "model_count": int(count),
            "meta": f"{int(count)} model" + ("" if int(count) == 1 else "s"),
        }
        for metric_key, count in sorted(
            metric_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if metric_key
    ]


def _common_metric_options(
    task_ids: list[str],
    task_column_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    common_keys: set[str] | None = None
    model_count_by_metric: dict[str, int] = {}
    task_count = 0

    for task_id in task_ids:
        task_column = task_column_by_id.get(task_id) or {}
        metric_options = task_column.get("metric_options") or []
        task_metric_counts = {
            str(option.get("value")): int(option.get("model_count") or 0)
            for option in metric_options
            if option.get("value")
        }
        if not task_metric_counts:
            return []
        task_keys = set(task_metric_counts)
        common_keys = task_keys if common_keys is None else common_keys & task_keys
        model_count_by_metric.update(
            {
                metric_key: (
                    task_metric_counts[metric_key]
                    if metric_key not in model_count_by_metric
                    else min(model_count_by_metric[metric_key], task_metric_counts[metric_key])
                )
                for metric_key in task_metric_counts
            }
        )
        task_count += 1
        if common_keys == set():
            return []

    if not common_keys:
        return []

    return [
        {
            "value": metric_key,
            "label": metric_key,
            "model_count": int(model_count_by_metric.get(metric_key, 0)),
            "meta": (
                f"all {task_count} tasks"
                + (
                    f" · {int(model_count_by_metric.get(metric_key, 0))} model"
                    + ("" if int(model_count_by_metric.get(metric_key, 0)) == 1 else "s")
                )
            ),
        }
        for metric_key in sorted(common_keys)
    ]


def _count_models_with_task_scores(
    models: list[dict[str, Any]],
    task_ids: list[str],
    *,
    require_all: bool,
) -> int:
    if not task_ids:
        return 0

    count = 0
    for model in models:
        task_scores = model.get("task_scores", {})
        if require_all:
            if all(_is_numeric_score(task_scores.get(task_id)) for task_id in task_ids):
                count += 1
        elif any(_is_numeric_score(task_scores.get(task_id)) for task_id in task_ids):
            count += 1
    return count


def _task_scope_availability(
    *,
    task_name: str,
    group_model_count: int,
    latest_model_count: int,
) -> dict[str, Any]:
    ready = latest_model_count >= 2
    model_label = "model" if latest_model_count == 1 else "models"
    if ready:
        supporting_text = f"paired test ready with {latest_model_count} latest {model_label}"
    elif latest_model_count == 1:
        supporting_text = "needs coverage: only 1 latest model has a score"
    else:
        supporting_text = "needs coverage: no latest models have a score"

    return {
        "ready": ready,
        "status_badge": "ready" if ready else "needs coverage",
        "status_tone": "ready" if ready else "limited",
        "supporting_text": supporting_text,
        "sort_priority": 0 if ready else 1,
        "title_suffix": (
            f"paired test ready now with {latest_model_count} latest {model_label}; "
            f"{group_model_count} group-level runs scored this task"
            if ready
            else f"only {latest_model_count} latest {model_label} scored this task; "
            "click to see what is missing"
        ),
    }


def _suite_scope_availability(
    *,
    suite_name: str,
    all_task_ids: list[str],
    covered_tasks: int,
    total_tasks: int,
    latest_models: list[dict[str, Any]],
    require_full_coverage: bool,
) -> dict[str, Any]:
    full_model_count = _count_models_with_task_scores(
        latest_models,
        all_task_ids,
        require_all=True,
    )
    partial_model_count = _count_models_with_task_scores(
        latest_models,
        all_task_ids,
        require_all=False,
    )

    if require_full_coverage:
        ready = covered_tasks == total_tasks and full_model_count >= 2
        if covered_tasks < total_tasks:
            missing = total_tasks - covered_tasks
            supporting_text = (
                f"needs coverage: {missing} suite task(s) are still missing in this group"
            )
            title_suffix = (
                f"only {covered_tasks}/{total_tasks} suite tasks appear in the group; "
                "click to see what still needs to run"
            )
        elif full_model_count >= 2:
            model_label = "model" if full_model_count == 1 else "models"
            supporting_text = f"paired test ready with {full_model_count} latest {model_label}"
            title_suffix = (
                f"paired test ready now with {full_model_count} latest models covering all "
                f"{total_tasks} suite tasks"
            )
        elif full_model_count == 1:
            supporting_text = "needs coverage: only 1 latest model covers the full suite"
            title_suffix = (
                f"all {total_tasks} suite tasks exist, but only 1 latest model covers them all; "
                "click to see what still needs to run"
            )
        else:
            supporting_text = "needs coverage: no latest model covers the full suite"
            title_suffix = (
                f"all {total_tasks} suite tasks exist, but no latest models cover them all; "
                "click to see what still needs to run"
            )
    else:
        ready = partial_model_count >= 2
        model_label = "model" if partial_model_count == 1 else "models"
        if ready:
            supporting_text = f"likely ready with {partial_model_count} latest {model_label}"
            title_suffix = (
                f"at least {partial_model_count} latest models have scores on this suite; "
                "pairwise still depends on shared instance overlap"
            )
        elif partial_model_count == 1:
            supporting_text = "needs coverage: only 1 latest model has suite scores"
            title_suffix = "only 1 latest model has scores in this suite"
        else:
            supporting_text = "needs coverage: no latest models have suite scores"
            title_suffix = "no latest models have scores in this suite"

    return {
        "ready": ready,
        "status_badge": "ready" if ready else "needs coverage",
        "status_tone": "ready" if ready else "limited",
        "supporting_text": supporting_text,
        "sort_priority": 0 if ready else 1,
        "title_suffix": title_suffix,
    }


def _build_group_browser_data(
    session: Session,
    group_name: str,
    *,
    require_full_coverage: bool,
) -> dict[str, Any]:
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
    results_table = _build_results_table(session, group_name)
    latest_models = list(results_table.get("models", []))
    task_column_by_id = {
        str(column["id"]): column for column in results_table.get("task_columns", [])
    }
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
                "task_ids": list(dict.fromkeys(expanded_tasks)),
                "visible_task_ids": visible_tasks,
            }
        )
    for suite_row in suite_rows:
        default_metrics = {
            str(task_column_by_id.get(task_id, {}).get("metric") or "")
            for task_id in suite_row["visible_task_ids"]
            if task_column_by_id.get(task_id, {}).get("metric")
        }
        suite_row["availability"] = _suite_scope_availability(
            suite_name=suite_row["name"],
            all_task_ids=list(suite_row["task_ids"]),
            covered_tasks=int(suite_row["covered"]),
            total_tasks=int(suite_row["total"]),
            latest_models=latest_models,
            require_full_coverage=require_full_coverage,
        )
        suite_row["default_metric"] = (
            next(iter(default_metrics)) if len(default_metrics) == 1 else ""
        )
        suite_row["metric_options"] = _common_metric_options(
            list(suite_row["visible_task_ids"]),
            task_column_by_id,
        )
    suite_rows.sort(
        key=lambda row: (
            int(row["availability"]["sort_priority"]),
            -float(row["ratio"]),
            row["name"],
        )
    )

    for task_row in task_rows:
        latest_model_count = int(task_column_by_id.get(task_row["name"], {}).get("model_count", 0))
        task_row["latest_models"] = latest_model_count
        task_row["metric_options"] = list(
            task_column_by_id.get(task_row["name"], {}).get("metric_options", []),
        )
        task_row["default_metric"] = str(
            task_column_by_id.get(task_row["name"], {}).get("metric") or ""
        )
        task_row["availability"] = _task_scope_availability(
            task_name=str(task_row["name"]),
            group_model_count=int(task_row["models"]),
            latest_model_count=latest_model_count,
        )
    task_rows.sort(
        key=lambda row: (
            int(row["availability"]["sort_priority"]),
            -int(row["latest_models"]),
            str(row["name"]),
        )
    )

    scope_options = [
        {
            "key": _make_scope_key("suite", suite_row["name"]),
            "kind": "suite",
            "label": f"{suite_row['name']} · {suite_row['covered']}/{suite_row['total']}",
            "value": suite_row["name"],
            "task_ids": list(suite_row["task_ids"]),
            "default_metric": suite_row["default_metric"],
            "metric_options": list(suite_row["metric_options"]),
            "status_badge": suite_row["availability"]["status_badge"],
            "status_tone": suite_row["availability"]["status_tone"],
            "supporting_text": suite_row["availability"]["supporting_text"],
            "ready": bool(suite_row["availability"]["ready"]),
            "sort_priority": int(suite_row["availability"]["sort_priority"]),
            "title_suffix": suite_row["availability"]["title_suffix"],
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
            "default_metric": task_row["default_metric"],
            "metric_options": list(task_row["metric_options"]),
            "status_badge": task_row["availability"]["status_badge"],
            "status_tone": task_row["availability"]["status_tone"],
            "supporting_text": task_row["availability"]["supporting_text"],
            "ready": bool(task_row["availability"]["ready"]),
            "sort_priority": int(task_row["availability"]["sort_priority"]),
            "title_suffix": task_row["availability"]["title_suffix"],
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
        "results_table": results_table,
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
    options: list[dict[str, Any]],
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

    def _render_search_select_option(index: int, option: dict[str, Any]) -> str:
        selected_class = " is-selected" if option["value"] == resolved_selected_value else ""
        tone_class = (
            f" is-{html.escape(str(option['status_tone']))}" if option.get("status_tone") else ""
        )
        meta_markup = ""
        if option["meta"]:
            meta_markup = (
                f'<span class="search-select-option-meta">{html.escape(option["meta"])}</span>'
            )
        status_markup = ""
        if option.get("status_badge"):
            status_tone = html.escape(str(option.get("status_tone") or "neutral"))
            status_markup = (
                '<span class="search-select-option-state '
                f'is-{status_tone}">{html.escape(str(option["status_badge"]))}</span>'
            )
        supporting_markup = ""
        if option.get("supporting_text"):
            supporting_markup = (
                '<span class="search-select-option-sub">'
                f"{html.escape(str(option['supporting_text']))}"
                "</span>"
            )
        aside_markup = ""
        if meta_markup or status_markup:
            aside_markup = (
                f'<span class="search-select-option-aside">{meta_markup}{status_markup}</span>'
            )
        return dedent(
            f"""
            <button
              type="button"
              class="search-select-option{selected_class}{tone_class}"
              data-action="select-search-option"
              data-role="search-select-option"
              data-value="{html.escape(option["value"])}"
              data-summary-text="{html.escape(option["summary_text"])}"
              data-filter-text="{html.escape(option["filter_text"])}"
              data-option-index="{index}"
              title="{html.escape(option["title"])}"
            >
              <span class="search-select-option-copy">
                <span class="search-select-option-main">{html.escape(option["label"])}</span>
                {supporting_markup}
              </span>
              {aside_markup}
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


def _selected_scope_option(
    group_data: dict[str, Any] | None,
    selected_scope_key: str | None,
) -> dict[str, Any] | None:
    if group_data is None or selected_scope_key is None:
        return None
    return next(
        (
            option
            for option in group_data.get("scope_options", [])
            if option.get("key") == selected_scope_key
        ),
        None,
    )


def _pick_metric_for_scope(
    group_data: dict[str, Any] | None,
    selected_scope_key: str | None,
    requested_metric: str | None,
) -> str | None:
    if not requested_metric:
        return None
    scope_option = _selected_scope_option(group_data, selected_scope_key)
    if scope_option is None:
        return None
    valid_metrics = {
        str(option.get("value") or "")
        for option in scope_option.get("metric_options", [])
        if option.get("value")
    }
    return requested_metric if requested_metric in valid_metrics else None


def _render_metric_control(
    *,
    scope_option: dict[str, Any] | None,
    selected_metric: str | None,
    pairwise_error_details: dict[str, Any] | None,
) -> str:
    metric_options = list((scope_option or {}).get("metric_options") or [])
    if not metric_options:
        return ""

    default_metric = str((scope_option or {}).get("default_metric") or "")
    error_code = str((pairwise_error_details or {}).get("code") or "")
    should_render = bool(
        selected_metric
        or len(metric_options) > 1
        or not default_metric
        or error_code in {"missing_primary_metric", "insufficient_extractable_instance_scores"}
    )
    if not should_render:
        return ""

    current_value = selected_metric or ""
    default_label = default_metric if default_metric else "select metric..."
    help_text = (
        "choose a metric to retry this paired test"
        if error_code in {"missing_primary_metric", "insufficient_extractable_instance_scores"}
        and not selected_metric
        else ""
    )
    options_html = "".join(
        (
            f'<option value="{html.escape(str(option["value"]))}"'
            + (' selected="selected"' if str(option["value"]) == current_value else "")
            + f">{html.escape(str(option['label']))}</option>"
        )
        for option in metric_options
    )
    default_selected_attr = ' selected="selected"' if current_value == "" else ""
    help_markup = f'<span class="control-help">{html.escape(help_text)}</span>' if help_text else ""
    return dedent(
        f"""
        <label class="select metric-select-control">
          <span class="select-label">metric</span>
          <div class="select-wrap">
            <select
              id="metric-select"
              name="metric"
              class="control-select"
              aria-label="metric"
            >
              <option value=""{default_selected_attr}>
                {html.escape(default_label)}
              </option>
              {options_html}
            </select>
          </div>
          {help_markup}
        </label>
        """
    ).strip()


def _viewer_pairwise_error_payload(
    error: Exception,
    *,
    selected_group: str | None,
) -> dict[str, Any]:
    if isinstance(error, PairwiseEligibilityError):
        payload = error.to_payload()
        code = str(payload.get("code") or "")
        notes = [
            note
            for note in list(payload.get("notes") or [])
            if "`olmo-eval" not in note and "--" not in note
        ]
        suggestions_by_code = {
            "insufficient_matched_experiments": [
                "Broaden the current group or scope so at least two runs remain for comparison.",
                "Switch to the Results tab to inspect which models and tasks are available "
                "in this group.",
            ],
            "insufficient_full_coverage_models": [
                "Choose a narrower task or suite that more models completed end to end.",
                "Run the missing suite tasks for the dropped model hashes so at least two "
                "latest models cover the full scope.",
            ],
            "insufficient_unique_models_after_dedupe": [
                "Broaden the scope to include more distinct model hashes.",
                "The viewer keeps the latest run per model hash, so repeated runs of the "
                "same checkpoint do not create extra heatmap rows.",
            ],
            "insufficient_compared_models": [
                "Broaden the current filters or switch to a scope with more comparable models.",
            ],
            "missing_task_rows": [
                "Choose another suite or task from the selector.",
                "Switch to the Results tab to inspect which scopes the retained runs actually "
                "completed.",
            ],
            "insufficient_extractable_instance_scores": [
                "Choose another scope that already has per-instance paired-test data.",
                "Re-run the missing tasks with the per-instance metric stored so the viewer "
                "can align instances across models.",
            ],
            "no_shared_instances": [
                "Choose a narrower suite or a single task that the retained models all ran.",
                "If the same models should be comparable here, run the missing tasks so the "
                "models overlap on the same instances.",
            ],
        }
        payload["notes"] = notes
        payload["suggestions"] = suggestions_by_code.get(
            code,
            [
                "Choose another suite or task from the selector.",
                "Switch to the Results tab to inspect what data is available for this group.",
            ],
        )
        payload["message"] = payload.get("summary") or str(error)
        return payload

    message = str(error).strip()

    if message.startswith("No primary_metric set for task '"):
        task_name = message.split("task '", 1)[1].split("'", 1)[0]
        return {
            "code": "missing_primary_metric",
            "summary": f"'{task_name}' does not define a default metric for the paired test.",
            "scope_label": task_name,
            "notes": [
                "The paired-test view needs one metric per task so it knows which "
                "per-instance scores to compare.",
            ],
            "suggestions": [
                "Choose another task or suite that already has a default metric.",
                "If this task should support paired comparison, set a primary metric in the "
                "stored task results and rerun it.",
            ],
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
            "message": message,
        }

    if message.startswith("Suite '") and " not found" in message:
        suite_name = message.split("Suite '", 1)[1].split("'", 1)[0]
        suggestions = ["Choose a suite directly from the suite / task selector."]
        if "Did you mean:" in message:
            hint = message.split("Did you mean:", 1)[1].strip().rstrip("?")
            if hint:
                suggestions.append(f"Nearby suite names: {hint}.")
        return {
            "code": "unknown_suite",
            "summary": f"'{suite_name}' is not a valid suite in this viewer.",
            "scope_label": suite_name,
            "notes": [
                "This usually means the page was opened with a stale or mistyped suite name."
            ],
            "suggestions": suggestions,
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
            "message": message,
        }

    if message.startswith("Suite '") and "resolved to zero tasks" in message:
        suite_name = message.split("Suite '", 1)[1].split("'", 1)[0]
        return {
            "code": "empty_suite_scope",
            "summary": f"'{suite_name}' currently has no tasks available for paired comparison.",
            "scope_label": suite_name,
            "notes": [
                "The suite selector landed on a scope that does not currently expand to any "
                "tasks the viewer can compare."
            ],
            "suggestions": [
                "Choose another suite or a single task from the selector.",
                "Switch to the Results tab to inspect which tasks are present in this group.",
            ],
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
            "message": message,
        }

    if message.startswith("--task-hash prefix '"):
        task_hash = message.split("prefix '", 1)[1].split("'", 1)[0]
        return {
            "code": "ambiguous_task_scope",
            "summary": f"The saved task link '{task_hash}' is ambiguous in this viewer.",
            "scope_label": task_hash,
            "notes": [
                "This scope matches more than one task, so the viewer cannot decide which "
                "single task to compare."
            ],
            "suggestions": [
                "Pick the specific task from the suite / task selector instead of using this "
                "ambiguous saved URL.",
            ],
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
            "message": message,
        }

    if message.startswith("Task '") and "excluded" in message:
        task_name = message.split("Task '", 1)[1].split("'", 1)[0]
        return {
            "code": "excluded_task_scope",
            "summary": f"'{task_name}' is not available in this viewer scope.",
            "scope_label": task_name,
            "notes": [
                "The current paired-test request points at a task that is not available "
                "for comparison."
            ],
            "suggestions": ["Choose another task from the suite / task selector."],
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
            "message": message,
        }

    return {
        "code": "viewer_pairwise_error",
        "summary": "The viewer could not render this paired test.",
        "scope_label": None,
        "notes": [
            "This scope hit a paired-test configuration or data issue that the viewer "
            "could not resolve automatically."
        ],
        "suggestions": [
            "Choose another suite or task from the selector.",
            *(
                ["Switch to the Results tab to inspect what data is available in this group."]
                if selected_group
                else []
            ),
        ],
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
        "message": message,
    }


def render_pairwise_browser_page(
    *,
    groups: list[dict[str, Any]],
    selected_group: str | None,
    group_data: dict[str, Any] | None,
    selected_scope_key: str | None,
    pairwise_data: dict[str, Any] | None,
    pairwise_error: str | None,
    pairwise_error_details: dict[str, Any] | None = None,
    selected_metric: str | None = None,
) -> str:
    """Render the viewer page with server-populated selectors and payload."""
    browser_payload = {
        "group_data": group_data,
        "selected_scope_key": selected_scope_key,
        "selected_metric": selected_metric,
        "pairwise_data": pairwise_data,
        "pairwise_error": pairwise_error,
        "pairwise_error_details": pairwise_error_details,
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
                    " ".join(
                        part
                        for part in (
                            option["value"],
                            option["label"],
                            option["kind"],
                            option.get("supporting_text"),
                            option.get("status_badge"),
                        )
                        if part
                    )
                ),
                "meta": _clean_inline_text(option["kind"]),
                "supporting_text": _clean_inline_text(option.get("supporting_text", "")),
                "status_badge": _clean_inline_text(option.get("status_badge", "")),
                "status_tone": _clean_inline_text(option.get("status_tone", "")),
                "title": _clean_inline_text(
                    " · ".join(
                        part
                        for part in (
                            _scope_option_label(
                                option,
                                selected_scope_key=selected_scope_key,
                                pairwise_data=pairwise_data,
                            )
                            if option["key"] == selected_scope_key
                            else option["label"],
                            option.get("supporting_text"),
                            option.get("title_suffix"),
                        )
                        if part
                    )
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

    selected_scope_option = _selected_scope_option(group_data, selected_scope_key)
    metric_control = _render_metric_control(
        scope_option=selected_scope_option,
        selected_metric=selected_metric,
        pairwise_error_details=pairwise_error_details,
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
            metric_control=metric_control,
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
            requested_metric = params.get("metric", [""])[0] or None

            with db.session() as session:
                groups = groups_cache.get_or_set(
                    ("groups", 500),
                    lambda: _list_groups(session),
                )
                selected_group = _pick_group(groups, requested_group)
                group_data = (
                    group_browser_cache.get_or_set(
                        (selected_group, require_full_coverage),
                        lambda: _build_group_browser_data(
                            session,
                            selected_group,
                            require_full_coverage=require_full_coverage,
                        ),
                    )
                    if selected_group is not None
                    else None
                )
                selected_scope_key = (
                    _pick_scope(group_data, requested_scope) if group_data is not None else None
                )
                selected_metric = _pick_metric_for_scope(
                    group_data,
                    selected_scope_key,
                    requested_metric,
                )
                pairwise_data: dict[str, Any] | None = None
                pairwise_error: str | None = None
                pairwise_error_details: dict[str, Any] | None = None

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
                            metric=selected_metric,
                            keep_all=keep_all,
                            require_full_coverage=require_full_coverage,
                        )
                        pairwise_data = build_pairwise_viewer_payload(result)
                    except PairwiseEligibilityError as error:
                        pairwise_error_details = _viewer_pairwise_error_payload(
                            error,
                            selected_group=selected_group,
                        )
                        pairwise_error = str(pairwise_error_details.get("summary") or error)
                    except ValueError as error:
                        pairwise_error_details = _viewer_pairwise_error_payload(
                            error,
                            selected_group=selected_group,
                        )
                        pairwise_error = str(pairwise_error_details.get("summary") or error)

            page = render_pairwise_browser_page(
                groups=groups,
                selected_group=selected_group,
                group_data=group_data,
                selected_scope_key=selected_scope_key,
                selected_metric=selected_metric,
                pairwise_data=pairwise_data,
                pairwise_error=pairwise_error,
                pairwise_error_details=pairwise_error_details,
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

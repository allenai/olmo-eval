"""Local server for the results viewer UI."""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
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
    _build_pairwise_score_sql_expr,
    _comparison_score,
    _format_experiment_label,
    compute_pairwise,
    get_task_metric_profile,
)
from olmo_eval.analysis.pairwise_metrics import TaskDisplayEntry, build_task_display_entries
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
_MIN_PAIRWISE_READY_MODELS = 2
_VIEWER_ERROR_EMPTY_COLLECTION_FIELDS = (
    "counts",
    "matched_runs",
    "compared_models",
    "dropped_duplicate_runs",
    "dropped_partial_coverage_models",
    "scored_models",
    "unscored_models",
    "unsupported_task_metrics",
    "per_model_instance_counts",
)


@dataclass(slots=True)
class _TimedCacheEntry:
    created_at: float
    value: Any


@dataclass(slots=True, frozen=True)
class _ScopeTarget:
    task_name: str | None = None
    task_hash: str | None = None
    suite_name: str | None = None


@dataclass(slots=True)
class _TaskColumnState:
    task_name: str
    task_hash: str | None
    metric: str | None = None
    profile: Any = None
    metric_options: Counter[str] = field(default_factory=Counter)
    model_count: int = 0

    @property
    def task_id(self) -> str:
        return str(self.task_hash or self.task_name)


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
    if kind not in {"suite", "task", "task-hash"} or not value:
        return None, None
    return kind, value


def _pluralized_label(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _scope_target(scope_kind: str | None, scope_value: str | None) -> _ScopeTarget:
    return _ScopeTarget(
        task_name=scope_value if scope_kind == "task" else None,
        task_hash=scope_value if scope_kind == "task-hash" else None,
        suite_name=scope_value if scope_kind == "suite" else None,
    )


def _result_scope_name(result: Any) -> str | None:
    return result.suite_name or result.task_name


def _result_scope_kind(result: Any) -> str:
    return "suite" if result.suite_name else "task"


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
    return None


def _pick_scope(group_data: dict[str, Any], requested: str | None) -> str | None:
    scope_options = group_data.get("scope_options", [])
    if not scope_options:
        return None
    if requested and any(option["key"] == requested for option in scope_options):
        return requested
    requested_kind, requested_value = _parse_scope_key(requested)
    if requested_kind == "task" and requested_value:
        matching_tasks = [
            str(option["key"])
            for option in scope_options
            if str(option.get("kind") or "") == "task"
            and str(option.get("task_name") or option.get("value") or "") == requested_value
        ]
        if len(matching_tasks) == 1:
            return matching_tasks[0]
    return None


def _serialize_run_mode(keep_all: bool) -> str:
    return "repeated" if keep_all else "latest"


def _resolve_run_mode(
    requested: str | None,
    *,
    default_keep_all: bool,
) -> tuple[str, bool]:
    normalized = str(requested or "").strip().lower()
    if normalized == "repeated":
        return "repeated", True
    if normalized == "latest":
        return "latest", False
    return _serialize_run_mode(default_keep_all), default_keep_all


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


def _all_group_experiments(session: Session, group_name: str) -> list[Any]:
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
            .order_by(Experiment.timestamp.desc(), Experiment.model_hash, Experiment.id.desc())
        )
        .scalars()
        .all()
    )
    return sorted(experiments, key=lambda experiment: experiment.timestamp, reverse=True)


def _group_experiments(session: Session, group_name: str, *, keep_all: bool) -> list[Any]:
    if keep_all:
        return _all_group_experiments(session, group_name)
    return _latest_group_experiments(session, group_name)


def _task_scope_id(task_name: Any, task_hash: Any) -> str:
    return str(task_hash or task_name or "")


def _record_task_column_state(
    task_states_by_id: dict[str, _TaskColumnState],
    *,
    task_name: Any,
    task_hash: Any,
    metrics: Any,
    primary_metric: Any,
) -> _TaskColumnState:
    resolved_name = str(task_name or "")
    resolved_hash = str(task_hash) if task_hash else None
    task_id = _task_scope_id(resolved_name, resolved_hash)
    task_state = task_states_by_id.setdefault(
        task_id,
        _TaskColumnState(task_name=resolved_name, task_hash=resolved_hash),
    )

    metric_key = str(primary_metric) if primary_metric else None
    if metric_key is not None and task_state.metric is None:
        task_state.metric = metric_key
        task_state.profile = get_task_metric_profile(resolved_name, metric_key)

    available_metric_keys = _available_metric_keys(metrics)
    task_state.metric_options.update(available_metric_keys)
    if metric_key and metric_key not in available_metric_keys:
        task_state.metric_options[metric_key] += 1
    return task_state


def _ordered_task_columns(
    task_states: list[_TaskColumnState],
) -> list[tuple[_TaskColumnState, TaskDisplayEntry]]:
    ordered_states = sorted(
        task_states,
        key=lambda task_state: (task_state.task_name, task_state.task_hash or ""),
    )
    task_entries = build_task_display_entries(
        tuple(task_state.task_name for task_state in ordered_states),
        tuple(task_state.task_hash or "" for task_state in ordered_states),
    )
    return list(zip(ordered_states, task_entries, strict=False))


def _serialize_task_column(
    task_state: _TaskColumnState,
    task_entry: TaskDisplayEntry,
) -> dict[str, Any]:
    profile = task_state.profile
    metric = task_state.metric or ""
    return {
        "id": task_entry.id,
        "task_name": task_entry.task_name,
        "task_hash": task_entry.task_hash,
        "label": task_entry.label,
        "full_label": task_entry.full_label,
        "metric": metric,
        "metric_options": _serialize_metric_options(task_state.metric_options),
        "score_display_format": profile.display_format if profile is not None else "raw",
        "score_unit": profile.unit if profile is not None else metric,
        "higher_is_better": profile.higher_is_better if profile is not None else True,
        "model_count": int(task_state.model_count),
    }


def _annotate_task_variants(
    raw_task_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    task_entries = build_task_display_entries(
        tuple(str(row["name"]) for row in raw_task_rows),
        tuple(str(row.get("task_hash") or "") for row in raw_task_rows),
    )
    task_variant_count_by_name = Counter(str(row["name"]) for row in raw_task_rows)

    task_rows: list[dict[str, Any]] = []
    task_ids_by_name: dict[str, list[str]] = {}
    for raw_row, task_entry in zip(raw_task_rows, task_entries, strict=False):
        task_rows.append(
            {
                **raw_row,
                "id": task_entry.id,
                "label": task_entry.label,
                "full_label": task_entry.full_label,
                "hash_qualified": task_variant_count_by_name[str(raw_row["name"])] > 1,
            }
        )
        task_ids_by_name.setdefault(str(raw_row["name"]), []).append(task_entry.id)

    return task_rows, task_ids_by_name


def _build_results_table(session: Session, group_name: str, *, keep_all: bool) -> dict[str, Any]:
    from olmo_eval.runners.processing.utils import extract_score_from_metrics
    from olmo_eval.storage.backends.postgres.models import TaskResult

    experiments = _group_experiments(session, group_name, keep_all=keep_all)
    if not experiments:
        return {"models": [], "task_columns": []}

    experiments.sort(key=lambda experiment: experiment.timestamp, reverse=True)
    selected_pks = [experiment.id for experiment in experiments]
    task_rows = session.execute(
        select(
            TaskResult.experiment_pk,
            TaskResult.task_name,
            TaskResult.task_hash,
            TaskResult.metrics,
            TaskResult.primary_metric,
        ).where(TaskResult.experiment_pk.in_(selected_pks))
    ).all()

    if keep_all:
        experiment_labels = {
            experiment.id: _format_experiment_label(
                experiment.model_name,
                experiment.model_hash,
                experiment.timestamp,
                keep_all=True,
            )
            for experiment in experiments
        }
    else:
        label_counts = Counter(experiment.model_name for experiment in experiments)
        experiment_labels = {
            experiment.id: (
                f"{experiment.model_name} ({experiment.model_hash[:8]})"
                if label_counts[experiment.model_name] > 1
                else experiment.model_name
            )
            for experiment in experiments
        }

    task_states_by_id: dict[str, _TaskColumnState] = {}
    task_scores_by_pk: dict[int, dict[str, float | None]] = {pk: {} for pk in selected_pks}
    for experiment_pk, task_name, task_hash, metrics, primary_metric in task_rows:
        task_state = _record_task_column_state(
            task_states_by_id,
            task_name=task_name,
            task_hash=task_hash,
            metrics=metrics,
            primary_metric=primary_metric,
        )
        task_id = task_state.task_id
        metric_key = str(primary_metric) if primary_metric else None
        score = extract_score_from_metrics(metrics, metric_key) if metric_key else None
        task_scores_by_pk.setdefault(experiment_pk, {})[task_id] = score
        if score is not None:
            task_state.model_count += 1

    ordered_task_columns = _ordered_task_columns(list(task_states_by_id.values()))
    ordered_task_ids = [task_entry.id for _, task_entry in ordered_task_columns]
    task_columns = [
        _serialize_task_column(task_state, task_entry)
        for task_state, task_entry in ordered_task_columns
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
                "task_scores": {task_id: task_scores.get(task_id) for task_id in ordered_task_ids},
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


def _score_display_format(meta: dict[str, Any] | None) -> str:
    if meta is None:
        return "percentage"
    return str(meta.get("score_display_format") or "percentage")


def _score_unit(meta: dict[str, Any] | None) -> Any:
    if meta is None:
        return None
    return meta.get("score_unit")


def _score_higher_is_better(meta: dict[str, Any] | None) -> bool:
    return not (meta is not None and meta.get("higher_is_better") is False)


def _columns_comparable(columns: list[dict[str, Any]]) -> bool:
    if not columns:
        return False
    reference = columns[0]
    return all(
        _score_display_format(column) == _score_display_format(reference)
        and _score_unit(column) == _score_unit(reference)
        and _score_higher_is_better(column) == _score_higher_is_better(reference)
        for column in columns
    )


def _aggregate_column_meta(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _columns_comparable(columns):
        return None
    return {
        "score_display_format": _score_display_format(columns[0]),
        "score_unit": _score_unit(columns[0]),
        "higher_is_better": _score_higher_is_better(columns[0]),
    }


def _scoped_task_columns(
    results_table: dict[str, Any] | None,
    selected_scope_option: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    task_columns = list((results_table or {}).get("task_columns", []))
    task_ids = list((selected_scope_option or {}).get("task_ids") or [])
    if not task_ids:
        return task_columns
    allowed_task_ids = {str(task_id) for task_id in task_ids}
    return [column for column in task_columns if str(column.get("id") or "") in allowed_task_ids]


def _average_model_scope_score(
    model: dict[str, Any],
    columns: list[dict[str, Any]],
) -> float | None:
    if not _columns_comparable(columns):
        return None
    task_scores = model.get("task_scores", {})
    scores = [
        float(task_scores[column["id"]])
        for column in columns
        if _is_numeric_score(task_scores.get(column["id"]))
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _format_score_value(value: Any, meta: dict[str, Any] | None) -> str:
    if not _is_numeric_score(value) or meta is None:
        return "—"
    if _score_display_format(meta) == "percentage":
        return f"{float(value) * 100:.1f}%"
    return f"{float(value):.1f}"


def _model_filter_score_label(model: dict[str, Any], columns: list[dict[str, Any]]) -> str:
    column_meta = _aggregate_column_meta(columns)
    average_score = _average_model_scope_score(model, columns)
    return _format_score_value(average_score, column_meta)


def _scope_status_payload(
    *,
    ready: bool,
    supporting_text: str,
    title_suffix: str,
) -> dict[str, Any]:
    return {
        "ready": ready,
        "status_badge": "ready" if ready else "needs coverage",
        "status_tone": "ready" if ready else "limited",
        "supporting_text": supporting_text,
        "sort_priority": 0 if ready else 1,
        "title_suffix": title_suffix,
    }


def _task_scope_availability(
    *,
    task_name: str,
    group_model_count: int,
    latest_model_count: int,
) -> dict[str, Any]:
    ready = latest_model_count >= _MIN_PAIRWISE_READY_MODELS
    model_label = _pluralized_label(latest_model_count, "model")
    if ready:
        supporting_text = f"paired test ready with {latest_model_count} latest {model_label}"
    elif latest_model_count == 1:
        supporting_text = "needs coverage: only 1 latest model has a score"
    else:
        supporting_text = "needs coverage: no latest models have a score"

    return _scope_status_payload(
        ready=ready,
        supporting_text=supporting_text,
        title_suffix=(
            f"paired test ready now with {latest_model_count} latest {model_label}; "
            f"{group_model_count} group-level runs scored this task"
            if ready
            else f"only {latest_model_count} latest {model_label} scored this task; "
            "click to see what is missing"
        ),
    )


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
        ready = covered_tasks == total_tasks and full_model_count >= _MIN_PAIRWISE_READY_MODELS
        if covered_tasks < total_tasks:
            missing = total_tasks - covered_tasks
            supporting_text = (
                f"needs coverage: {missing} suite task(s) are still missing in this group"
            )
            title_suffix = (
                f"only {covered_tasks}/{total_tasks} suite tasks appear in the group; "
                "click to see what still needs to run"
            )
        elif full_model_count >= _MIN_PAIRWISE_READY_MODELS:
            model_label = _pluralized_label(full_model_count, "model")
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
        ready = partial_model_count >= _MIN_PAIRWISE_READY_MODELS
        model_label = _pluralized_label(partial_model_count, "model")
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

    return _scope_status_payload(
        ready=ready,
        supporting_text=supporting_text,
        title_suffix=title_suffix,
    )


def _build_group_browser_data(
    session: Session,
    group_name: str,
    *,
    keep_all: bool,
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

    raw_task_rows = [
        {
            "name": str(task_name or ""),
            "task_hash": str(task_hash) if task_hash else None,
            "models": int(model_count or 0),
            "metric": metric or "",
        }
        for task_name, task_hash, model_count, metric in session.execute(
            select(
                TaskResult.task_name,
                TaskResult.task_hash,
                func.count(distinct(Experiment.model_hash)).label("models"),
                func.max(TaskResult.primary_metric).label("metric"),
            )
            .join(Experiment, Experiment.id == TaskResult.experiment_pk)
            .where(Experiment.experiment_group == group_name)
            .group_by(TaskResult.task_name, TaskResult.task_hash)
            .order_by(TaskResult.task_name, TaskResult.task_hash)
        ).all()
    ]
    task_rows, task_ids_by_name = _annotate_task_variants(raw_task_rows)

    present_tasks = {row["name"] for row in task_rows}
    results_table = _build_results_table(session, group_name, keep_all=keep_all)
    availability_table = (
        results_table if not keep_all else _build_results_table(session, group_name, keep_all=False)
    )
    latest_models = list(availability_table.get("models", []))
    task_column_by_id: dict[str, dict[str, Any]] = {
        str(column["id"]): column for column in results_table.get("task_columns", [])
    }
    latest_task_column_by_id: dict[str, dict[str, Any]] = {
        str(column["id"]): column for column in availability_table.get("task_columns", [])
    }
    suite_rows: list[dict[str, Any]] = []
    for suite_name in list_suites():
        expanded_tasks = get_suite(suite_name).expanded_tasks
        visible_task_names = list(
            dict.fromkeys(task_name for task_name in expanded_tasks if task_name in present_tasks)
        )
        total = len(expanded_tasks)
        covered = len(visible_task_names)
        if covered == 0:
            continue
        ratio = covered / total if total else 0.0
        visible_task_ids = [
            task_id
            for task_name in visible_task_names
            for task_id in task_ids_by_name.get(task_name, [])
        ]
        availability_task_ids: list[str] = []
        for task_name in visible_task_names:
            task_variant_ids = task_ids_by_name.get(task_name, [])
            if not task_variant_ids:
                continue
            availability_task_ids.append(
                max(
                    task_variant_ids,
                    key=lambda task_id: (
                        int(latest_task_column_by_id.get(task_id, {}).get("model_count", 0)),
                        str(task_id),
                    ),
                )
            )
        suite_rows.append(
            {
                "name": suite_name,
                "covered": covered,
                "total": total,
                "ratio": ratio,
                "task_ids": visible_task_ids,
                "visible_task_ids": visible_task_ids,
                "visible_task_names": visible_task_names,
                "availability_task_ids": availability_task_ids,
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
            all_task_ids=list(suite_row["availability_task_ids"]),
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
        task_id = str(task_row["id"])
        task_column = task_column_by_id.get(task_id) or {}
        latest_model_count = int(latest_task_column_by_id.get(task_id, {}).get("model_count", 0))
        task_row["latest_models"] = latest_model_count
        task_row["metric_options"] = list(task_column.get("metric_options", []))
        task_row["default_metric"] = str(task_column.get("metric") or "")
        task_row["availability"] = _task_scope_availability(
            task_name=str(task_row["full_label"]),
            group_model_count=int(task_row["models"]),
            latest_model_count=latest_model_count,
        )
    task_rows.sort(
        key=lambda row: (
            int(row["availability"]["sort_priority"]),
            -int(row["latest_models"]),
            str(row["full_label"]),
        )
    )

    scope_options: list[dict[str, Any]] = [
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
            "key": _make_scope_key(
                "task-hash" if task_row["hash_qualified"] else "task",
                str(task_row["task_hash"] or task_row["name"]),
            ),
            "kind": "task",
            "label": f"{task_row['full_label']} · {task_row['models']} models",
            "value": task_row["full_label"],
            "task_name": task_row["name"],
            "task_hash": task_row["task_hash"],
            "task_ids": [task_row["id"]],
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
    placeholder_label: str | None = None,
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

    selected_value_text = _clean_inline_text(selected_value or "")
    selected_option = next(
        (
            option
            for option in options
            if _clean_inline_text(option["value"]) == selected_value_text
        ),
        None,
    )
    summary_state_class = ""
    if selected_option is not None:
        resolved_selected_value = _clean_inline_text(selected_option["value"])
        resolved_selected_label = _clean_inline_text(
            selected_label or selected_option["summary_text"]
        )
    elif placeholder_label is not None:
        resolved_selected_value = ""
        resolved_selected_label = _clean_inline_text(placeholder_label)
        summary_state_class = " is-placeholder"
    else:
        fallback_option = options[0]
        resolved_selected_value = _clean_inline_text(fallback_option["value"])
        resolved_selected_label = _clean_inline_text(
            selected_label or fallback_option["summary_text"]
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
              class="control-summary search-select-summary{summary_state_class}"
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
            task_label = _pluralized_label(task_count, "task")
            scope_label = f"{scope_label} ({task_count} {task_label})"
    shared_n = meta.get("shared_n")
    if shared_n is not None:
        scope_label = f"{scope_label} · N={shared_n}"
    return scope_label


def _model_key(model: dict[str, Any]) -> str:
    if model.get("timestamp"):
        return _result_model_export_ref(model.get("model_hash"), model.get("timestamp"))
    return str(
        model.get("model_hash")
        or model.get("model_name")
        or model.get("display_label")
        or model.get("index")
        or ""
    )


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


def _slugify_export_part(value: str | None, fallback: str = "export") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or fallback


def _viewer_export_base_name(group_name: str, scope_label: str | None) -> str:
    group_key = _slugify_export_part(group_name, "results")
    scope_key = _slugify_export_part(scope_label, "paired-test")
    return group_key if group_key == scope_key else f"{group_key}-{scope_key}"


def _result_model_export_ref(model_hash: str | None, timestamp: str | None) -> str:
    return f"{str(model_hash or '')}|{str(timestamp or '')}"


def _build_viewer_export_metadata(
    *,
    group_name: str,
    result: Any,
    compared_model_count: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "group_name": group_name,
        "scope_name": _result_scope_name(result),
        "scope_kind": _result_scope_kind(result),
        "metric_name": result.metric,
        "compared_model_count": compared_model_count,
        **extra,
    }


def _export_experiment_identity(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_display_rank": int(experiment["display_rank"]),
        "model_label": str(experiment["display_label"]),
        "model_name": str(experiment["model_name"]),
        "model_hash": str(experiment["model_hash"]),
        "timestamp": experiment["timestamp"],
    }


def _write_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                fieldname: "" if row.get(fieldname) is None else row.get(fieldname)
                for fieldname in fieldnames
            }
        )
    return stream.getvalue().encode("utf-8")


def _resolve_export_group(
    session: Session,
    *,
    requested_group: str | None,
    keep_all: bool,
    require_full_coverage: bool,
) -> tuple[str, dict[str, Any]]:
    groups = _list_groups(session)
    if not requested_group:
        raise ValueError("Missing group for viewer export.")

    if not any(group["name"] == requested_group for group in groups):
        raise ValueError(f"Unknown group '{requested_group}' for viewer export.")

    return (
        requested_group,
        _build_group_browser_data(
            session,
            requested_group,
            keep_all=keep_all,
            require_full_coverage=require_full_coverage,
        ),
    )


def _resolve_export_scope_and_metric(
    *,
    group_data: dict[str, Any],
    requested_scope: str | None,
    requested_metric: str | None,
) -> tuple[str, str, str, str | None]:
    if not requested_scope:
        raise ValueError("Missing scope for viewer export.")

    scope_option = _selected_scope_option(group_data, requested_scope)
    if scope_option is None:
        raise ValueError(f"Unknown scope '{requested_scope}' for viewer export.")

    scope_kind, scope_value = _parse_scope_key(requested_scope)
    if scope_kind is None or scope_value is None:
        raise ValueError(f"Invalid scope '{requested_scope}' for viewer export.")

    selected_metric = _pick_metric_for_scope(group_data, requested_scope, requested_metric)
    if requested_metric and selected_metric != requested_metric:
        raise ValueError(
            f"Metric '{requested_metric}' is not available for scope '{requested_scope}'."
        )

    return requested_scope, scope_kind, scope_value, selected_metric


def _compute_group_pairwise(
    session: Session,
    *,
    group_name: str,
    scope_kind: str,
    scope_value: str,
    selected_metric: str | None,
    margin: float,
    keep_all: bool,
    require_full_coverage: bool,
) -> Any:
    scope_target = _scope_target(scope_kind, scope_value)
    return compute_pairwise(
        session=session,
        task_name=scope_target.task_name,
        task_hash=scope_target.task_hash,
        margin=margin,
        experiment_groups=[group_name],
        metric=selected_metric,
        keep_all=keep_all,
        require_full_coverage=require_full_coverage,
        suite_name=scope_target.suite_name,
    )


def _resolve_compared_experiments_for_result(
    session: Session,
    *,
    group_name: str,
    result: Any,
) -> list[dict[str, Any]]:
    from olmo_eval.storage.backends.postgres.models import Experiment

    compared_hashes = sorted(
        {model.model_hash for model in result.models if getattr(model, "model_hash", None)}
    )
    if not compared_hashes:
        return []

    experiment_rows = session.execute(
        select(
            Experiment.id,
            Experiment.model_name,
            Experiment.model_hash,
            Experiment.timestamp,
            Experiment.s3_location,
        )
        .where(Experiment.experiment_group == group_name)
        .where(Experiment.model_hash.in_(compared_hashes))
    ).all()

    rows_by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in experiment_rows:
        timestamp = row.timestamp.isoformat() if row.timestamp is not None else None
        rows_by_ref.setdefault(
            _result_model_export_ref(row.model_hash, timestamp),
            [],
        ).append(
            {
                "experiment_pk": int(row.id),
                "model_name": str(row.model_name or ""),
                "model_hash": str(row.model_hash or ""),
                "timestamp": timestamp,
                "results_root": row.s3_location,
            }
        )

    compared: list[dict[str, Any]] = []
    for display_rank, model in enumerate(result.models, start=1):
        ref = _result_model_export_ref(model.model_hash, model.timestamp)
        candidates = list(rows_by_ref.get(ref, []))
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate["model_name"] == getattr(model, "model_name", "")
            ),
            candidates[0] if candidates else None,
        )
        if selected is None:
            raise ValueError(
                f"Could not resolve the compared run for '{model.label.replace(chr(10), ' ')}'."
            )
        compared.append(
            {
                **selected,
                "display_rank": display_rank,
                "display_label": model.label.replace("\n", " "),
                "export_ref": ref,
            }
        )

    return compared


def _order_compared_experiments(
    compared_experiments: list[dict[str, Any]],
    requested_model_refs: list[str],
) -> list[dict[str, Any]]:
    if not requested_model_refs:
        return compared_experiments

    by_ref = {str(experiment["export_ref"]): experiment for experiment in compared_experiments}
    ordered = [by_ref[ref] for ref in requested_model_refs if ref in by_ref]
    if not ordered:
        return compared_experiments

    return [
        {
            **experiment,
            "display_rank": display_rank,
        }
        for display_rank, experiment in enumerate(ordered, start=1)
    ]


def _load_compared_scope_task_rows(
    session: Session,
    *,
    compared_experiments: list[dict[str, Any]],
    result: Any,
) -> list[dict[str, Any]]:
    from olmo_eval.storage.backends.postgres.models import TaskResult

    experiment_ids = [int(experiment["experiment_pk"]) for experiment in compared_experiments]
    task_hashes = list(getattr(result, "task_hashes", ()) or ())
    task_names = list(result.task_names or ((result.task_name,) if result.task_name else ()))
    if not experiment_ids or (not task_hashes and not task_names):
        return []

    stmt = select(
        TaskResult.experiment_pk,
        TaskResult.task_name,
        TaskResult.task_hash,
        TaskResult.num_instances,
        TaskResult.primary_metric,
        TaskResult.s3_metrics_key,
        TaskResult.s3_predictions_key,
        TaskResult.s3_requests_key,
    ).where(TaskResult.experiment_pk.in_(experiment_ids))
    if task_hashes:
        stmt = stmt.where(TaskResult.task_hash.in_(task_hashes))
    elif task_names:
        stmt = stmt.where(TaskResult.task_name.in_(task_names))
    task_rows = session.execute(stmt).all()

    return [
        {
            "experiment_pk": int(row.experiment_pk),
            "task_name": str(row.task_name or ""),
            "task_hash": str(row.task_hash or ""),
            "num_instances": row.num_instances,
            "primary_metric": row.primary_metric,
            "task_metrics_file": row.s3_metrics_key,
            "predictions_file": row.s3_predictions_key,
            "requests_file": row.s3_requests_key,
        }
        for row in task_rows
    ]


def _build_instance_results_export_data(
    session: Session,
    *,
    group_name: str,
    result: Any,
    compared_experiments: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    selected_metric: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from olmo_eval.storage.backends.postgres.models import InstancePrediction

    experiment_ids = [int(experiment["experiment_pk"]) for experiment in compared_experiments]
    task_hash_to_metric: dict[str, str] = {}
    task_hash_to_name: dict[str, str] = {}
    task_profile_by_hash: dict[str, Any] = {}

    for row in task_rows:
        task_hash = str(row["task_hash"])
        task_name = str(row["task_name"])
        metric_key = selected_metric or str(row.get("primary_metric") or "")
        if not metric_key:
            continue
        task_hash_to_metric[task_hash] = metric_key
        task_hash_to_name[task_hash] = task_name
        task_profile_by_hash[task_hash] = get_task_metric_profile(task_name, metric_key)

    score_expr = _build_pairwise_score_sql_expr(task_hash_to_metric, task_profile_by_hash)
    if not experiment_ids or score_expr is None:
        metadata = _build_viewer_export_metadata(
            group_name=group_name,
            result=result,
            compared_model_count=len(compared_experiments),
            shared_instance_count=0,
            row_count=0,
            score_display_format=result.score_display_format,
            score_unit=result.score_unit,
            score_higher_is_better=result.higher_is_better,
        )
        return metadata, []

    score_rows = session.execute(
        select(
            InstancePrediction.experiment_pk,
            InstancePrediction.native_id,
            InstancePrediction.task_hash,
            score_expr,
        )
        .where(InstancePrediction.experiment_pk.in_(experiment_ids))
        .where(InstancePrediction.task_hash.in_(list(task_hash_to_metric)))
    ).all()

    raw_scores_by_pk = {experiment_id: {} for experiment_id in experiment_ids}
    comparison_scores_by_pk = {experiment_id: {} for experiment_id in experiment_ids}

    for experiment_pk, native_id, task_hash, raw_score in score_rows:
        if raw_score is None:
            continue
        resolved_task_hash = str(task_hash)
        task_name = task_hash_to_name.get(resolved_task_hash)
        if not task_name:
            continue
        key = (resolved_task_hash, str(native_id))
        raw_scores_by_pk[int(experiment_pk)][key] = float(raw_score)
        comparison_scores_by_pk[int(experiment_pk)][key] = _comparison_score(
            float(raw_score),
            task_profile_by_hash.get(resolved_task_hash),
        )

    shared_ids: set[tuple[str, str]] = set(
        comparison_scores_by_pk.get(experiment_ids[0], {}).keys()
    )
    for experiment_id in experiment_ids[1:]:
        shared_ids &= set(comparison_scores_by_pk.get(experiment_id, {}).keys())

    rows: list[dict[str, Any]] = []
    for task_hash, native_id in sorted(
        shared_ids,
        key=lambda pair: (task_hash_to_name.get(pair[0], ""), pair[0], pair[1]),
    ):
        task_name = task_hash_to_name.get(task_hash)
        if not task_name:
            continue
        metric_key = task_hash_to_metric.get(task_hash)
        profile = task_profile_by_hash.get(task_hash)
        for experiment in compared_experiments:
            experiment_id = int(experiment["experiment_pk"])
            key = (task_hash, native_id)
            rows.append(
                {
                    **_export_experiment_identity(experiment),
                    "task_name": task_name,
                    "task_hash": task_hash,
                    "native_id": native_id,
                    "task_metric_key": metric_key,
                    "raw_score": raw_scores_by_pk[experiment_id].get(key),
                    "comparison_score": comparison_scores_by_pk[experiment_id].get(key),
                    "score_display_format": (
                        profile.display_format
                        if profile is not None
                        else result.score_display_format
                    ),
                    "score_unit": profile.unit if profile is not None else result.score_unit,
                    "score_higher_is_better": (
                        profile.higher_is_better if profile is not None else result.higher_is_better
                    ),
                }
            )

    metadata = _build_viewer_export_metadata(
        group_name=group_name,
        result=result,
        compared_model_count=len(compared_experiments),
        shared_instance_count=len(shared_ids),
        row_count=len(rows),
        score_display_format=result.score_display_format,
        score_unit=result.score_unit,
        score_higher_is_better=result.higher_is_better,
    )
    return metadata, rows


def _build_stored_files_export_data(
    *,
    group_name: str,
    result: Any,
    compared_experiments: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    selected_metric: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment_by_pk = {
        int(experiment["experiment_pk"]): experiment for experiment in compared_experiments
    }
    rows: list[dict[str, Any]] = []
    for task_row in sorted(
        task_rows,
        key=lambda row: (
            int(experiment_by_pk.get(int(row["experiment_pk"]), {}).get("display_rank", 0)),
            str(row["task_name"]),
        ),
    ):
        experiment = experiment_by_pk.get(int(task_row["experiment_pk"]))
        if experiment is None:
            continue
        rows.append(
            {
                **_export_experiment_identity(experiment),
                "task_name": str(task_row["task_name"]),
                "task_hash": str(task_row["task_hash"]),
                "task_metric_key": selected_metric or task_row.get("primary_metric"),
                "num_instances": task_row.get("num_instances"),
                "predictions_file": task_row.get("predictions_file"),
                "requests_file": task_row.get("requests_file"),
                "task_metrics_file": task_row.get("task_metrics_file"),
                "run_results_root": experiment.get("results_root"),
            }
        )

    metadata = _build_viewer_export_metadata(
        group_name=group_name,
        result=result,
        compared_model_count=len(compared_experiments),
        task_count=len({str(row["task_hash"] or row["task_name"]) for row in task_rows}),
        row_count=len(rows),
    )
    return metadata, rows


def _serialize_viewer_export(
    *,
    kind: str,
    format_name: str,
    base_name: str,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[str, str, bytes]:
    if kind == "instance-results":
        suffix = "instance-results"
        csv_fields = [
            "model_display_rank",
            "model_label",
            "model_name",
            "model_hash",
            "timestamp",
            "task_name",
            "task_hash",
            "native_id",
            "task_metric_key",
            "raw_score",
            "comparison_score",
            "score_display_format",
            "score_unit",
            "score_higher_is_better",
        ]
    elif kind == "stored-files":
        suffix = "stored-files"
        csv_fields = [
            "model_display_rank",
            "model_label",
            "model_name",
            "model_hash",
            "timestamp",
            "task_name",
            "task_hash",
            "task_metric_key",
            "num_instances",
            "predictions_file",
            "requests_file",
            "task_metrics_file",
            "run_results_root",
        ]
    else:
        raise ValueError(f"Unsupported viewer export kind '{kind}'.")

    payload = {"metadata": metadata, "rows": rows}
    filename_base = f"{base_name}-{suffix}"
    if format_name == "json":
        return (
            f"{filename_base}.json",
            "application/json; charset=utf-8",
            (json.dumps(payload, indent=2) + "\n").encode("utf-8"),
        )
    if format_name == "csv":
        return (
            f"{filename_base}.csv",
            "text/csv; charset=utf-8",
            _write_csv_bytes(rows, csv_fields),
        )
    raise ValueError(f"Unsupported viewer export format '{format_name}'.")


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


def _render_run_mode_control(*, selected_run_mode: str) -> str:
    resolved_run_mode = "repeated" if selected_run_mode == "repeated" else "latest"
    latest_selected = ' selected="selected"' if resolved_run_mode == "latest" else ""
    repeated_selected = ' selected="selected"' if resolved_run_mode == "repeated" else ""
    return dedent(
        f"""
        <label class="select run-mode-control">
          <span class="select-label">runs</span>
          <div class="select-wrap">
            <select
              id="run-mode-select"
              name="runs"
              class="control-select"
              aria-label="runs"
            >
              <option value="latest"{latest_selected}>
                latest only
              </option>
              <option value="repeated"{repeated_selected}>
                repeated runs
              </option>
            </select>
          </div>
        </label>
        """
    ).strip()


def _build_viewer_error_payload(
    *,
    code: str,
    summary: str,
    scope_label: str | None,
    notes: list[str],
    suggestions: list[str],
    message: str,
    filter_summary: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "summary": summary,
        "scope_label": scope_label,
        "notes": notes,
        "suggestions": suggestions,
        "filter_summary": filter_summary,
        "message": message,
        **{field: [] for field in _VIEWER_ERROR_EMPTY_COLLECTION_FIELDS},
    }


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
                "Switch the runs control from latest only to repeated runs if you want "
                "separate heatmap rows for reruns of the same checkpoint.",
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
        return _build_viewer_error_payload(
            code="missing_primary_metric",
            summary=f"'{task_name}' does not define a default metric for the paired test.",
            scope_label=task_name,
            notes=[
                "The paired-test view needs one metric per task so it knows which "
                "per-instance scores to compare.",
            ],
            suggestions=[
                "Choose another task or suite that already has a default metric.",
                "If this task should support paired comparison, set a primary metric in the "
                "stored task results and rerun it.",
            ],
            message=message,
        )

    if message.startswith("Suite '") and " not found" in message:
        suite_name = message.split("Suite '", 1)[1].split("'", 1)[0]
        suggestions = ["Choose a suite directly from the suite / task selector."]
        if "Did you mean:" in message:
            hint = message.split("Did you mean:", 1)[1].strip().rstrip("?")
            if hint:
                suggestions.append(f"Nearby suite names: {hint}.")
        return _build_viewer_error_payload(
            code="unknown_suite",
            summary=f"'{suite_name}' is not a valid suite in this viewer.",
            scope_label=suite_name,
            notes=["This usually means the page was opened with a stale or mistyped suite name."],
            suggestions=suggestions,
            message=message,
        )

    if message.startswith("Suite '") and "resolved to zero tasks" in message:
        suite_name = message.split("Suite '", 1)[1].split("'", 1)[0]
        return _build_viewer_error_payload(
            code="empty_suite_scope",
            summary=f"'{suite_name}' currently has no tasks available for paired comparison.",
            scope_label=suite_name,
            notes=[
                "The suite selector landed on a scope that does not currently expand to any "
                "tasks the viewer can compare."
            ],
            suggestions=[
                "Choose another suite or a single task from the selector.",
                "Switch to the Results tab to inspect which tasks are present in this group.",
            ],
            message=message,
        )

    if message.startswith("--task-hash prefix '"):
        task_hash = message.split("prefix '", 1)[1].split("'", 1)[0]
        return _build_viewer_error_payload(
            code="ambiguous_task_scope",
            summary=f"The saved task link '{task_hash}' is ambiguous in this viewer.",
            scope_label=task_hash,
            notes=[
                "This scope matches more than one task, so the viewer cannot decide which "
                "single task to compare."
            ],
            suggestions=[
                "Pick the specific task from the suite / task selector instead of using this "
                "ambiguous saved URL.",
            ],
            message=message,
        )

    if message.startswith("Task '") and "matches" in message and "task configs" in message:
        task_name = message.split("Task '", 1)[1].split("'", 1)[0]
        return _build_viewer_error_payload(
            code="ambiguous_task_scope",
            summary=f"'{task_name}' has multiple task configs in this viewer.",
            scope_label=task_name,
            notes=[
                "The same task name appears with more than one saved config hash, so the "
                "viewer will not merge them into one paired test."
            ],
            suggestions=[
                "Pick one of the hash-qualified task entries from the suite / task selector.",
                "Switch to the Results tab to inspect the separate task columns for each config.",
            ],
            message=message,
        )

    if message.startswith("Task '") and "excluded" in message:
        task_name = message.split("Task '", 1)[1].split("'", 1)[0]
        return _build_viewer_error_payload(
            code="excluded_task_scope",
            summary=f"'{task_name}' is not available in this viewer scope.",
            scope_label=task_name,
            notes=[
                "The current paired-test request points at a task that is not available "
                "for comparison."
            ],
            suggestions=["Choose another task from the suite / task selector."],
            message=message,
        )

    return _build_viewer_error_payload(
        code="viewer_pairwise_error",
        summary="The viewer could not render this paired test.",
        scope_label=None,
        notes=[
            "This scope hit a paired-test configuration or data issue that the viewer "
            "could not resolve automatically."
        ],
        suggestions=[
            "Choose another suite or task from the selector.",
            *(
                ["Switch to the Results tab to inspect what data is available in this group."]
                if selected_group
                else []
            ),
        ],
        message=message,
    )


def render_results_viewer_page(
    *,
    groups: list[dict[str, Any]],
    selected_group: str | None,
    group_data: dict[str, Any] | None,
    selected_scope_key: str | None,
    pairwise_data: dict[str, Any] | None,
    pairwise_error: str | None,
    pairwise_error_details: dict[str, Any] | None = None,
    selected_metric: str | None = None,
    selected_run_mode: str = "latest",
) -> str:
    """Render the viewer page with server-populated selectors and payload."""
    browser_payload = {
        "has_groups": bool(groups),
        "selected_group": selected_group,
        "group_data": group_data,
        "selected_scope_key": selected_scope_key,
        "selected_metric": selected_metric,
        "selected_run_mode": selected_run_mode,
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
        placeholder_label="select group...",
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
    model_filter_columns: list[dict[str, Any]] = []
    selected_scope_option = _selected_scope_option(group_data, selected_scope_key)
    if group_data and group_data.get("results_table"):
        model_filter_columns = _scoped_task_columns(
            group_data.get("results_table"),
            selected_scope_option,
        )
        model_filter_models = sorted(
            list(group_data["results_table"].get("models", [])),
            key=lambda model: str(model.get("display_label") or "").lower(),
        )

    metric_control = _render_metric_control(
        scope_option=selected_scope_option,
        selected_metric=selected_metric,
        pairwise_error_details=pairwise_error_details,
    )
    run_mode_control = _render_run_mode_control(selected_run_mode=selected_run_mode)

    model_filter_options = "".join(
        (
            '<label class="tt-menu-row">'
            f'<input type="checkbox" data-action="toggle-model-checkbox" '
            f'data-model-key="{html.escape(_model_key(model))}" checked />'
            f'<span class="tt-menu-name">'
            f"{html.escape(str(model.get('display_label') or '').replace(chr(10), ' '))}"
            "</span>"
            f'<span class="tt-menu-n">'
            f"{html.escape(_model_filter_score_label(model, model_filter_columns))}"
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
        placeholder_label="select suite or task...",
    )

    return (
        render_template(
            "browser.html",
            page_title="olmo-eval results viewer",
            styles=_PAIRWISE_SHARED_CSS + "\n" + _BROWSER_EXTRA_CSS,
            group_select=group_select,
            scope_select=scope_select,
            run_mode_control=run_mode_control,
            metric_control=metric_control,
            model_filter_control=model_filter_control,
            payload_json=payload_json,
            script=_BROWSER_JS,
        ).strip()
        + "\n"
    )


def serve_results_viewer(
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

    class ResultsViewerHandler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            *,
            body: bytes,
            status: int = 200,
            content_type: str,
            filename: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def _send_text(
            self,
            *,
            text: str,
            status: int = 200,
            content_type: str = "text/plain; charset=utf-8",
        ) -> None:
            self._send_bytes(
                status=status,
                body=text.encode("utf-8"),
                content_type=content_type,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            params = parse_qs(parsed.query)
            if parsed.path == "/export":
                requested_group = params.get("group", [""])[0] or None
                requested_scope = params.get("scope", [""])[0] or None
                requested_metric = params.get("metric", [""])[0] or None
                requested_run_mode = params.get("runs", [""])[0] or None
                requested_kind = params.get("kind", [""])[0] or None
                requested_format = params.get("format", ["csv"])[0] or "csv"
                requested_model_refs = [
                    ref for ref in params.get("model_ref", []) if isinstance(ref, str) and ref
                ]
                _, selected_keep_all = _resolve_run_mode(
                    requested_run_mode,
                    default_keep_all=keep_all,
                )

                if requested_kind not in {"instance-results", "stored-files"}:
                    self._send_text(status=400, text="Unsupported export kind.\n")
                    return

                if requested_format not in {"csv", "json"}:
                    self._send_text(status=400, text="Unsupported export format.\n")
                    return

                try:
                    with db.session() as session:
                        selected_group, group_data = _resolve_export_group(
                            session,
                            requested_group=requested_group,
                            keep_all=selected_keep_all,
                            require_full_coverage=require_full_coverage,
                        )
                        _, scope_kind, scope_value, selected_metric = (
                            _resolve_export_scope_and_metric(
                                group_data=group_data,
                                requested_scope=requested_scope,
                                requested_metric=requested_metric,
                            )
                        )
                        result = _compute_group_pairwise(
                            session=session,
                            group_name=selected_group,
                            scope_kind=scope_kind,
                            scope_value=scope_value,
                            selected_metric=selected_metric,
                            margin=margin,
                            keep_all=selected_keep_all,
                            require_full_coverage=require_full_coverage,
                        )
                        compared_experiments = _order_compared_experiments(
                            _resolve_compared_experiments_for_result(
                                session,
                                group_name=selected_group,
                                result=result,
                            ),
                            requested_model_refs,
                        )
                        task_rows = _load_compared_scope_task_rows(
                            session,
                            compared_experiments=compared_experiments,
                            result=result,
                        )
                        if requested_kind == "instance-results":
                            metadata, rows = _build_instance_results_export_data(
                                session,
                                group_name=selected_group,
                                result=result,
                                compared_experiments=compared_experiments,
                                task_rows=task_rows,
                                selected_metric=selected_metric,
                            )
                        else:
                            metadata, rows = _build_stored_files_export_data(
                                group_name=selected_group,
                                result=result,
                                compared_experiments=compared_experiments,
                                task_rows=task_rows,
                                selected_metric=selected_metric,
                            )
                        filename, content_type, body = _serialize_viewer_export(
                            kind=requested_kind,
                            format_name=requested_format,
                            base_name=_viewer_export_base_name(
                                selected_group,
                                _result_scope_name(result),
                            ),
                            metadata=metadata,
                            rows=rows,
                        )
                except PairwiseEligibilityError as error:
                    self._send_text(status=409, text=str(error) + "\n")
                    return
                except ValueError as error:
                    self._send_text(status=400, text=str(error) + "\n")
                    return

                self._send_bytes(
                    body=body,
                    content_type=content_type,
                    filename=filename,
                )
                return

            if parsed.path not in {"", "/"}:
                self.send_response(404)
                self.end_headers()
                return

            requested_group = params.get("group", [initial_group or ""])[0] or None
            requested_scope = params.get("scope", [initial_scope_key or ""])[0] or None
            requested_metric = params.get("metric", [""])[0] or None
            requested_run_mode = params.get("runs", [""])[0] or None
            selected_run_mode, selected_keep_all = _resolve_run_mode(
                requested_run_mode,
                default_keep_all=keep_all,
            )

            with db.session() as session:
                groups = groups_cache.get_or_set(
                    ("groups", 500),
                    lambda: _list_groups(session),
                )
                selected_group = _pick_group(groups, requested_group)
                group_data = (
                    group_browser_cache.get_or_set(
                        (selected_group, selected_keep_all, require_full_coverage),
                        lambda: _build_group_browser_data(
                            session,
                            selected_group,
                            keep_all=selected_keep_all,
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
                        result = _compute_group_pairwise(
                            session=session,
                            group_name=selected_group,
                            scope_kind=scope_kind,
                            scope_value=scope_value,
                            selected_metric=selected_metric,
                            margin=margin,
                            keep_all=selected_keep_all,
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

            page = render_results_viewer_page(
                groups=groups,
                selected_group=selected_group,
                group_data=group_data,
                selected_scope_key=selected_scope_key,
                selected_metric=selected_metric,
                selected_run_mode=selected_run_mode,
                pairwise_data=pairwise_data,
                pairwise_error=pairwise_error,
                pairwise_error_details=pairwise_error_details,
            )
            encoded = page.encode("utf-8")
            self._send_bytes(body=encoded, content_type="text/html; charset=utf-8")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), ResultsViewerHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return server.server_port

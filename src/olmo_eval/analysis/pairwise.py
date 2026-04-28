"""Pairwise comparisons from instance-level scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from olmo_eval.common.types.base import EvalResult


@dataclass(frozen=True)
class ModelMeta:
    """Display and identity fields for one matrix row."""

    label: str
    model_name: str = ""
    model_hash: str = ""
    timestamp: str | None = None


@dataclass(frozen=True)
class PairStats:
    """Head-to-head counts and variance terms for one ordered pair."""

    index_a: int
    index_b: int
    wins_a: int
    wins_b: int
    ties: int
    var_paired_diff: float = 0.0
    var_marginal_sum: float = 0.0

    @property
    def n_contested(self) -> int:
        """Return the number of shared instances that were not tied."""
        return self.wins_a + self.wins_b

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.n_contested if self.n_contested > 0 else 0.5

    @property
    def win_rate_b(self) -> float:
        return 1.0 - self.win_rate_a

    @property
    def se(self) -> float:
        """Return the CLT standard error of the contested win rate."""
        n = self.n_contested
        if n <= 1:
            return 0.0
        p = self.wins_a / n
        sample_var = n / (n - 1) * p * (1 - p)
        return math.sqrt(sample_var / n)

    @property
    def prob_a_gt_b(self) -> float:
        """Approximate ``P(A > B)`` under the same normal approximation as the SE."""
        se = self.se
        win_rate = self.win_rate_a
        if se <= 0:
            if win_rate > 0.5:
                return 1.0
            if win_rate < 0.5:
                return 0.0
            return 0.5
        z = (win_rate - 0.5) / se
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    @property
    def p_value(self) -> float:
        """Return the two-sided p-value for the same normal approximation."""
        se = self.se
        win_rate = self.win_rate_a
        if se <= 0:
            return 1.0 if win_rate == 0.5 else 0.0
        z = abs(win_rate - 0.5) / se
        return math.erfc(z / math.sqrt(2.0))


@dataclass(frozen=True)
class FilteredModel:
    """One model dropped for lacking full suite coverage."""

    model_name: str
    model_hash: str
    missing_tasks: tuple[str, ...] = ()
    instance_shortfalls: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class MetricProfile:
    """Resolved pairwise semantics for one task metric."""

    supports_scorer_fallback: bool
    higher_is_better: bool
    display_format: str
    unit: str


class PairwiseEligibilityError(ValueError):
    """Structured failure that explains why the paired test cannot render."""

    def __init__(
        self,
        *,
        code: str,
        summary: str,
        message: str | None = None,
        scope_label: str | None = None,
        filter_summary: str | None = None,
        counts: list[dict[str, Any]] | None = None,
        matched_runs: list[dict[str, Any]] | None = None,
        compared_models: list[dict[str, Any]] | None = None,
        dropped_duplicate_runs: list[dict[str, Any]] | None = None,
        dropped_partial_coverage_models: list[dict[str, Any]] | None = None,
        scored_models: list[dict[str, Any]] | None = None,
        unscored_models: list[dict[str, Any]] | None = None,
        unsupported_task_metrics: list[str] | None = None,
        per_model_instance_counts: list[dict[str, Any]] | None = None,
        notes: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        self.code = code
        self.summary = summary
        self.scope_label = scope_label
        self.filter_summary = filter_summary
        self.counts = counts or []
        self.matched_runs = matched_runs or []
        self.compared_models = compared_models or []
        self.dropped_duplicate_runs = dropped_duplicate_runs or []
        self.dropped_partial_coverage_models = dropped_partial_coverage_models or []
        self.scored_models = scored_models or []
        self.unscored_models = unscored_models or []
        self.unsupported_task_metrics = unsupported_task_metrics or []
        self.per_model_instance_counts = per_model_instance_counts or []
        self.notes = notes or []
        self.suggestions = suggestions or []
        super().__init__(message or summary)

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "summary": self.summary,
            "message": str(self),
            "scope_label": self.scope_label,
            "filter_summary": self.filter_summary,
            "counts": self.counts,
            "matched_runs": self.matched_runs,
            "compared_models": self.compared_models,
            "dropped_duplicate_runs": self.dropped_duplicate_runs,
            "dropped_partial_coverage_models": self.dropped_partial_coverage_models,
            "scored_models": self.scored_models,
            "unscored_models": self.unscored_models,
            "unsupported_task_metrics": self.unsupported_task_metrics,
            "per_model_instance_counts": self.per_model_instance_counts,
            "notes": self.notes,
            "suggestions": self.suggestions,
        }


@dataclass
class PairwiseResult:
    """Pairwise comparison output for one task or suite scope."""

    task_name: str
    metric: str
    margin: float
    instance_count: int
    models: list[ModelMeta]
    pairs: list[PairStats]
    suite_name: str | None = None
    task_names: tuple[str, ...] = ()
    task_hashes: tuple[str, ...] = ()
    n_experiments_matched: int = 0
    n_experiments_dropped: int = 0
    filtered_models: tuple[FilteredModel, ...] = ()
    model_costs: tuple[float | None, ...] = ()
    task_metric_keys: tuple[str | None, ...] = ()
    model_task_scores: tuple[tuple[float | None, ...], ...] = ()
    model_shared_scores: tuple[float | None, ...] = ()
    score_display_format: str = "percentage"
    score_unit: str | None = "proportion"
    higher_is_better: bool | None = True


def get_win_rate(pairs: list[PairStats], row: int, col: int) -> float:
    """Look up the win rate for models[row] vs models[col]."""
    for p in pairs:
        if p.index_a == row and p.index_b == col:
            return p.win_rate_a
        if p.index_a == col and p.index_b == row:
            return p.win_rate_b
    return 0.5


def get_se(pairs: list[PairStats], row: int, col: int) -> float:
    """Look up the win-rate standard error for models[row] vs models[col]."""
    for p in pairs:
        if (p.index_a == row and p.index_b == col) or (p.index_a == col and p.index_b == row):
            return p.se
    return 0.0


def _matches_prefix(value: str | None, prefixes: list[str] | None) -> bool:
    """Return True when value starts with any configured prefix."""
    return (
        value is not None
        and prefixes is not None
        and any(value.startswith(prefix) for prefix in prefixes)
    )


def _matches_exact(value: str | None, values: list[str] | None) -> bool:
    """Return True when value exactly matches one of the configured values."""
    return value is not None and values is not None and value in values


def _is_excluded_experiment(
    model_name: str | None,
    model_hash: str | None,
    exclude_model_names: list[str] | None = None,
    exclude_model_hashes: list[str] | None = None,
) -> bool:
    """Return True when an experiment should be dropped from pairwise analysis."""
    return _matches_prefix(model_name, exclude_model_names) or _matches_prefix(
        model_hash, exclude_model_hashes
    )


def _is_excluded_task(
    task_name: str | None,
    task_hash: str | None,
    exclude_task_names: list[str] | None = None,
    exclude_task_hashes: list[str] | None = None,
) -> bool:
    """Return True when a task row should be excluded from pairwise analysis."""
    return _matches_exact(task_name, exclude_task_names) or _matches_prefix(
        task_hash, exclude_task_hashes
    )


def _filter_suite_task_names(
    task_names: tuple[str, ...],
    exclude_task_names: list[str] | None = None,
) -> tuple[str, ...]:
    """Remove excluded exact task names while preserving suite expansion order."""
    if not exclude_task_names:
        return task_names

    excluded = set(exclude_task_names)
    return tuple(task_name for task_name in task_names if task_name not in excluded)


def _timestamp_label(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def _short_task_hash(task_hash: str | None) -> str:
    return (task_hash or "")[:8]


def _format_task_variant_label(task_name: str, task_hash: str | None = None) -> str:
    hash_short = _short_task_hash(task_hash)
    return f"{task_name} [{hash_short}]" if hash_short else task_name


def _format_experiment_label(
    model_name: str | None,
    model_hash: str | None,
    timestamp: Any = None,
    *,
    keep_all: bool,
) -> str:
    resolved_name = model_name or "unnamed model"
    hash_short = (model_hash or "")[:8]
    if keep_all and timestamp is not None:
        timestamp_label = _timestamp_label(timestamp)
        if timestamp_label:
            return f"{resolved_name} ({hash_short} @ {timestamp_label})"
    return f"{resolved_name} ({hash_short})"


def _serialize_experiment_debug(exp: Any, *, keep_all: bool) -> dict[str, Any]:
    timestamp = getattr(exp, "timestamp", None)
    return {
        "label": _format_experiment_label(
            getattr(exp, "model_name", None),
            getattr(exp, "model_hash", None),
            timestamp,
            keep_all=keep_all,
        ),
        "model_name": getattr(exp, "model_name", "") or "",
        "model_hash": getattr(exp, "model_hash", "") or "",
        "model_hash_short": (getattr(exp, "model_hash", "") or "")[:8],
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "timestamp_label": _timestamp_label(timestamp),
        "experiment_id": getattr(exp, "experiment_id", None),
        "experiment_group": getattr(exp, "experiment_group", None),
    }


def _serialize_model_meta_debug(meta: ModelMeta) -> dict[str, Any]:
    return {
        "label": meta.label.replace("\n", " "),
        "model_name": meta.model_name or "",
        "model_hash": meta.model_hash or "",
        "model_hash_short": (meta.model_hash or "")[:8],
        "timestamp": meta.timestamp,
    }


def _serialize_filtered_model_debug(model: FilteredModel) -> dict[str, Any]:
    reasons: list[str] = []
    if model.missing_tasks:
        preview = ", ".join(model.missing_tasks[:3])
        if len(model.missing_tasks) > 3:
            preview = f"{preview}, +{len(model.missing_tasks) - 3} more"
        reasons.append(f"missing {len(model.missing_tasks)} task(s): {preview}")
    if model.instance_shortfalls:
        preview = ", ".join(
            f"{task_name} ({have}/{expected})"
            for task_name, have, expected in model.instance_shortfalls[:3]
        )
        if len(model.instance_shortfalls) > 3:
            preview = f"{preview}, +{len(model.instance_shortfalls) - 3} more"
        reasons.append(
            f"fewer shared instances on {len(model.instance_shortfalls)} task(s): {preview}"
        )
    return {
        "label": _format_experiment_label(
            model.model_name,
            model.model_hash,
            keep_all=False,
        ),
        "model_name": model.model_name,
        "model_hash": model.model_hash,
        "model_hash_short": model.model_hash[:8],
        "missing_tasks": list(model.missing_tasks),
        "instance_shortfalls": [
            {
                "task_name": task_name,
                "instances": have,
                "expected_instances": expected,
            }
            for task_name, have, expected in model.instance_shortfalls
        ],
        "reason_summary": "; ".join(reasons) if reasons else "partial suite coverage",
    }


def _build_ordered_models(selected: list[Any], *, keep_all: bool) -> list[tuple[int, ModelMeta]]:
    ordered: list[tuple[int, ModelMeta]] = []
    for exp in selected:
        ts_iso = exp.timestamp.isoformat() if exp.timestamp is not None else None
        if keep_all and exp.timestamp is not None:
            timestamp_label = _timestamp_label(exp.timestamp)
            if timestamp_label:
                label = f"{exp.model_name}\n({exp.model_hash[:8]} @ {timestamp_label})"
            else:
                label = f"{exp.model_name}\n({exp.model_hash[:8]})"
        else:
            label = f"{exp.model_name}\n({exp.model_hash[:8]})"
        ordered.append(
            (
                exp.id,
                ModelMeta(
                    label=label,
                    model_name=exp.model_name,
                    model_hash=exp.model_hash,
                    timestamp=ts_iso,
                ),
            )
        )
    return ordered


def _build_suite_coverage_lookup(
    *,
    task_rows: list[Any],
    selected_pks: set[int],
    suite_task_names: tuple[str, ...],
) -> dict[int, dict[str, dict[str, int]]]:
    allowed_task_names = set(suite_task_names)
    rows_by_pk_name_hash: dict[int, dict[str, dict[str, int]]] = {}
    for row in task_rows:
        pk = int(row.experiment_pk)
        task_name = str(row.task_name or "")
        task_hash = str(row.task_hash or "")
        if pk not in selected_pks or task_name not in allowed_task_names or not task_hash:
            continue
        rows_by_pk_name_hash.setdefault(pk, {}).setdefault(task_name, {})[task_hash] = int(
            row.num_instances or 0
        )
    return rows_by_pk_name_hash


def _best_suite_task_hashes(
    rows_by_pk_name_hash: dict[int, dict[str, dict[str, int]]],
    selected_pks: set[int],
    suite_task_names: tuple[str, ...],
) -> tuple[dict[str, str | None], dict[str, int]]:
    chosen_hash_by_task_name: dict[str, str | None] = {}
    max_instances_by_hash: dict[str, int] = {}

    for task_name in suite_task_names:
        coverage_by_hash: dict[str, set[int]] = {}
        instances_by_hash: dict[str, int] = {}
        for pk in selected_pks:
            hash_rows = rows_by_pk_name_hash.get(pk, {}).get(task_name, {})
            for task_hash, instance_count in hash_rows.items():
                coverage_by_hash.setdefault(task_hash, set()).add(pk)
                instances_by_hash[task_hash] = max(
                    instances_by_hash.get(task_hash, 0),
                    instance_count,
                )
        if not coverage_by_hash:
            chosen_hash_by_task_name[task_name] = None
            continue
        chosen_hash = max(
            coverage_by_hash,
            key=lambda task_hash: (
                len(coverage_by_hash[task_hash]),
                instances_by_hash.get(task_hash, 0),
                task_hash,
            ),
        )
        chosen_hash_by_task_name[task_name] = chosen_hash
        max_instances_by_hash[chosen_hash] = instances_by_hash.get(chosen_hash, 0)

    return chosen_hash_by_task_name, max_instances_by_hash


def _suite_hashes_shared_by_models(
    rows_by_pk_name_hash: dict[int, dict[str, dict[str, int]]],
    selected_pks: set[int],
    suite_task_names: tuple[str, ...],
) -> set[str]:
    if not selected_pks:
        return set()

    common_hashes: set[str] = set()
    for task_name in suite_task_names:
        hash_sets = [
            set(rows_by_pk_name_hash.get(pk, {}).get(task_name, {})) for pk in selected_pks
        ]
        if hash_sets:
            common_hashes.update(set.intersection(*hash_sets))
    return common_hashes


def _build_filtered_suite_model(
    exp: Any,
    *,
    rows_by_pk_name_hash: dict[int, dict[str, dict[str, int]]],
    chosen_hash_by_task_name: dict[str, str | None],
    max_instances_by_hash: dict[str, int],
    suite_task_names: tuple[str, ...],
) -> FilteredModel:
    task_hashes_by_name = rows_by_pk_name_hash.get(int(exp.id), {})
    missing: list[str] = []
    shortfalls: list[tuple[str, int, int]] = []

    for task_name in suite_task_names:
        chosen_hash = chosen_hash_by_task_name.get(task_name)
        if not chosen_hash:
            missing.append(task_name)
            continue

        task_label = _format_task_variant_label(task_name, chosen_hash)
        instance_count = task_hashes_by_name.get(task_name, {}).get(chosen_hash)
        if instance_count is None:
            missing.append(task_label)
            continue

        expected_instances = max_instances_by_hash.get(chosen_hash, 0)
        if expected_instances > instance_count:
            shortfalls.append((task_label, instance_count, expected_instances))

    return FilteredModel(
        model_name=exp.model_name,
        model_hash=exp.model_hash,
        missing_tasks=tuple(sorted(set(missing))),
        instance_shortfalls=tuple(sorted(shortfalls)),
    )


def _filter_full_coverage_suite_models(
    *,
    task_rows: list[Any],
    selected: list[Any],
    suite_task_names: tuple[str, ...],
) -> tuple[list[Any], list[FilteredModel], set[str]]:
    if not selected or not suite_task_names:
        return list(selected), [], set()

    selected_by_pk = {int(exp.id): exp for exp in selected}
    rows_by_pk_name_hash = _build_suite_coverage_lookup(
        task_rows=task_rows,
        selected_pks=set(selected_by_pk),
        suite_task_names=suite_task_names,
    )

    remaining_pks = set(selected_by_pk)
    chosen_hash_by_task_name: dict[str, str | None] = {}
    max_instances_by_hash: dict[str, int] = {}

    while remaining_pks:
        chosen_hash_by_task_name, max_instances_by_hash = _best_suite_task_hashes(
            rows_by_pk_name_hash,
            remaining_pks,
            suite_task_names,
        )
        next_remaining = {
            pk
            for pk in remaining_pks
            if all(
                (chosen_hash := chosen_hash_by_task_name.get(task_name))
                and rows_by_pk_name_hash.get(pk, {}).get(task_name, {}).get(chosen_hash)
                == max_instances_by_hash.get(chosen_hash, 0)
                for task_name in suite_task_names
            )
        }
        if next_remaining == remaining_pks:
            break
        remaining_pks = next_remaining

    kept = [exp for exp in selected if int(exp.id) in remaining_pks]
    filtered_models = [
        _build_filtered_suite_model(
            exp,
            rows_by_pk_name_hash=rows_by_pk_name_hash,
            chosen_hash_by_task_name=chosen_hash_by_task_name,
            max_instances_by_hash=max_instances_by_hash,
            suite_task_names=suite_task_names,
        )
        for exp in selected
        if int(exp.id) not in remaining_pks
    ]

    return (
        kept,
        filtered_models,
        _suite_hashes_shared_by_models(
            rows_by_pk_name_hash,
            remaining_pks,
            suite_task_names,
        ),
    )


def _compute_pairs(
    scores_by_idx: dict[int, dict[tuple[str, str], float]],
    n: int,
    shared_ids: set[tuple[str, str]],
    margin: float,
) -> list[PairStats]:
    """Aggregate win/loss/tie counts and variances from aligned scores."""

    def _sample_var_from_sums(count: int, total: float, total_sq: float) -> float:
        if count < 2:
            return 0.0
        centered_sum_sq = total_sq - (total * total) / count
        if centered_sum_sq <= 0:
            return 0.0
        return centered_sum_sq / (count - 1)

    ordered_shared_ids = tuple(shared_ids)
    aligned_scores = [
        tuple(scores_by_idx.get(idx, {}).get(key) for key in ordered_shared_ids) for idx in range(n)
    ]

    results: list[PairStats] = []
    for i in range(n):
        for j in range(i + 1, n):
            wins_a = 0
            wins_b = 0
            ties = 0
            compared = 0
            diff_total = 0.0
            diff_total_sq = 0.0
            score_a_total = 0.0
            score_a_total_sq = 0.0
            score_b_total = 0.0
            score_b_total_sq = 0.0

            for score_a, score_b in zip(aligned_scores[i], aligned_scores[j], strict=False):
                if score_a is None or score_b is None:
                    continue
                diff = score_a - score_b
                compared += 1
                diff_total += diff
                diff_total_sq += diff * diff
                score_a_total += score_a
                score_a_total_sq += score_a * score_a
                score_b_total += score_b
                score_b_total_sq += score_b * score_b
                if abs(diff) <= margin:
                    ties += 1
                elif diff > 0:
                    wins_a += 1
                else:
                    wins_b += 1
            var_d = _sample_var_from_sums(compared, diff_total, diff_total_sq)
            var_marginal = _sample_var_from_sums(
                compared,
                score_a_total,
                score_a_total_sq,
            ) + _sample_var_from_sums(
                compared,
                score_b_total,
                score_b_total_sq,
            )
            results.append(
                PairStats(
                    index_a=i,
                    index_b=j,
                    wins_a=wins_a,
                    wins_b=wins_b,
                    ties=ties,
                    var_paired_diff=var_d,
                    var_marginal_sum=var_marginal,
                )
            )
    return results


def _sum_numeric_leaves(value: Any) -> float | None:
    """Return the sum of numeric leaves under ``value`` or None when absent."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        total = 0.0
        found = False
        for nested in value.values():
            nested_total = _sum_numeric_leaves(nested)
            if nested_total is not None:
                total += nested_total
                found = True
        return total if found else None
    if isinstance(value, list | tuple):
        total = 0.0
        found = False
        for nested in value:
            nested_total = _sum_numeric_leaves(nested)
            if nested_total is not None:
                total += nested_total
                found = True
        return total if found else None
    return None


def _extract_cost_from_task_metrics(metrics: object) -> float | None:
    """Extract a task-level cost from heterogeneous metrics payloads."""
    if not isinstance(metrics, dict):
        return None
    metric_values = {key: value for key, value in metrics.items() if isinstance(key, str)}
    for key in ("overall_cost", "total_cost", "cost"):
        if key in metric_values:
            return _sum_numeric_leaves(metric_values[key])
    return None


@cache
def _get_task_metric_profile(task_name: str, metric_key: str) -> MetricProfile | None:
    """Resolve pairwise semantics for one task metric."""
    from olmo_eval.runners.processing.utils import parse_metric_key

    parsed = parse_metric_key(metric_key)
    if parsed is None:
        return None
    metric_name, scorer_name = parsed

    try:
        import olmo_eval.evals.tasks  # noqa: F401  # ensure task registration side effects
        from olmo_eval.evals.tasks.common.registry import get_task
    except Exception:
        return None

    try:
        task = get_task(task_name)
    except Exception:
        return None

    for candidate in task.config.metrics:
        try:
            candidate_scorer = candidate.scorer().name
        except Exception:
            continue
        if candidate.name == metric_name and candidate_scorer == scorer_name:
            return MetricProfile(
                supports_scorer_fallback=candidate.supports_pairwise_scorer_fallback(),
                higher_is_better=candidate.pairwise_higher_is_better(),
                display_format=candidate.pairwise_display_format(),
                unit=candidate.pairwise_unit(),
            )

    primary_metric = task.config.get_primary_metric()
    if primary_metric is None:
        return None

    try:
        primary_scorer = primary_metric.scorer().name
    except Exception:
        return None

    if primary_metric.name == metric_name and primary_scorer == scorer_name:
        return MetricProfile(
            supports_scorer_fallback=primary_metric.supports_pairwise_scorer_fallback(),
            higher_is_better=primary_metric.pairwise_higher_is_better(),
            display_format=primary_metric.pairwise_display_format(),
            unit=primary_metric.pairwise_unit(),
        )

    return None


def _extract_pairwise_instance_score(
    *,
    task_name: str,
    metric_key: str,
    instance_metrics: dict[str, dict[str, float]],
) -> float | None:
    """Extract the per-instance score that matches pairwise math for one task.

    We always prefer an exact metric key when it exists. Only metrics whose task-level
    value is literally an average of per-instance scorer outputs may fall back to the
    stored scorer channel (``scorer:scorer``).
    """
    from olmo_eval.runners.processing.utils import extract_score_from_metrics, parse_metric_key

    score = extract_score_from_metrics(instance_metrics, metric_key)
    if score is not None:
        return score

    profile = _get_task_metric_profile(task_name, metric_key)
    if profile is not None and not profile.supports_scorer_fallback:
        return None

    parsed = parse_metric_key(metric_key)
    if parsed is None:
        return None
    scorer = parsed[1]
    return extract_score_from_metrics(instance_metrics, f"{scorer}:{scorer}")


def _comparison_score(raw_score: float, profile: MetricProfile | None) -> float:
    """Map a raw task score to a higher-is-better comparison scalar."""
    if profile is None or profile.higher_is_better:
        return raw_score
    return -raw_score


def _build_pairwise_score_sql_expr(
    task_hash_to_metric: dict[str, str],
    task_profile_by_hash: dict[str, MetricProfile | None],
):
    """Build a SQL expression that extracts the pairwise score per task hash."""
    from sqlalchemy import Float, case, cast, func

    from olmo_eval.runners.processing.utils import parse_metric_key
    from olmo_eval.storage.backends.postgres.models import InstancePrediction

    grouped_hashes: dict[tuple[str, str, bool], list[str]] = {}
    for task_hash, metric_key in sorted(task_hash_to_metric.items()):
        parsed = parse_metric_key(metric_key)
        if parsed is None:
            continue
        metric_name, scorer_name = parsed
        profile = task_profile_by_hash.get(task_hash)
        supports_fallback = profile.supports_scorer_fallback if profile is not None else True
        grouped_hashes.setdefault((metric_name, scorer_name, supports_fallback), []).append(
            task_hash
        )

    if not grouped_hashes:
        return None

    cases: list[tuple[Any, Any]] = []
    for (metric_name, scorer_name, supports_fallback), task_hashes in grouped_hashes.items():
        exact_expr = cast(
            InstancePrediction.instance_metrics[metric_name][scorer_name].astext,
            Float,
        )
        if supports_fallback:
            scorer_expr = cast(
                InstancePrediction.instance_metrics[scorer_name][scorer_name].astext,
                Float,
            )
            value_expr = func.coalesce(exact_expr, scorer_expr)
        else:
            value_expr = exact_expr
        cases.append((InstancePrediction.task_hash.in_(task_hashes), value_expr))

    return case(*cases, else_=None).label("score")


def get_task_metric_profile(task_name: str, metric_key: str) -> MetricProfile | None:
    """Public wrapper for resolving pairwise metric semantics."""
    return _get_task_metric_profile(task_name, metric_key)


def _build_experiment_refetch_stmt(eval_results: list[EvalResult]):
    """Build the exact-pair re-fetch statement for matched experiments."""
    from sqlalchemy import select, tuple_

    from olmo_eval.storage.backends.postgres.models import Experiment

    exact_pairs = sorted(
        {
            (result.experiment_id, result.model_hash)
            for result in eval_results
            if result.model_hash is not None
        }
    )
    if not exact_pairs:
        return None
    return select(Experiment).where(
        tuple_(Experiment.experiment_id, Experiment.model_hash).in_(exact_pairs)
    )


def _validate_requested_task_scope(
    task_rows: list[Any],
    *,
    task_name: str | None,
    task_hash: str | None,
) -> None:
    distinct_task_hashes = sorted({str(row.task_hash or "") for row in task_rows if row.task_hash})

    if task_name and len(distinct_task_hashes) > 1:
        variants = [
            _format_task_variant_label(task_name, resolved_task_hash)
            for resolved_task_hash in distinct_task_hashes
        ]
        raise ValueError(
            f"Task '{task_name}' matches {len(distinct_task_hashes)} task configs in this "
            f"scope: {variants}. Use --task-hash or narrow the filters."
        )

    if task_hash and len(distinct_task_hashes) > 1:
        raise ValueError(
            f"--task-hash prefix '{task_hash}' matches {len(distinct_task_hashes)} task "
            f"hashes in this scope: {distinct_task_hashes}. Use a longer prefix."
        )

    if task_hash:
        distinct_task_names = sorted(
            {str(row.task_name or "") for row in task_rows if row.task_name}
        )
        if len(distinct_task_names) > 1:
            raise ValueError(
                f"--task-hash prefix '{task_hash}' matches {len(distinct_task_names)} "
                f"distinct task names: {distinct_task_names}. Use a longer prefix, "
                "pass --task for a single task, or --suite to pool intentionally."
            )


def _restrict_task_rows_to_selected_models(
    *,
    task_rows: list[Any],
    selected: list[Any],
    keep_all: bool,
    suite_task_names: tuple[str, ...],
    require_full_coverage: bool,
) -> tuple[list[Any], list[tuple[int, ModelMeta]], list[Any], list[FilteredModel]]:
    filtered_models: list[FilteredModel] = []
    common_suite_hashes: set[str] | None = None
    if require_full_coverage and suite_task_names:
        selected, filtered_models, common_suite_hashes = _filter_full_coverage_suite_models(
            task_rows=task_rows,
            selected=selected,
            suite_task_names=suite_task_names,
        )

    ordered = _build_ordered_models(selected, keep_all=keep_all)
    selected_pks = {pk for pk, _ in ordered}
    allowed_hashes = (
        {str(task_hash) for task_hash in common_suite_hashes}
        if common_suite_hashes is not None
        else None
    )
    filtered_task_rows = [
        row
        for row in task_rows
        if row.experiment_pk in selected_pks
        and (allowed_hashes is None or row.task_hash in allowed_hashes)
    ]
    return selected, ordered, filtered_task_rows, filtered_models


def _raise_insufficient_compared_models(
    *,
    ordered: list[tuple[int, ModelMeta]],
    selected: list[Any],
    keep_all: bool,
    filtered_models: list[FilteredModel],
    n_matched: int,
    selected_before_coverage: list[Any],
    scope_label: str,
    scope_str: str,
    candidate_experiments: list[Any],
    dropped_duplicate_runs: list[dict[str, Any]],
) -> None:
    if filtered_models:
        detail = (
            f"after --require-full-coverage dropped {len(filtered_models)} "
            "partial-coverage model(s)"
        )
        code = "insufficient_full_coverage_models"
        summary = (
            f"Only {len(ordered)} model(s) still satisfy the full-suite coverage "
            "requirement for the paired test."
        )
        notes = [
            "The results tab can still show models with partial suite coverage, but the "
            "paired test requires every compared model to cover the same task hashes."
        ]
        suggestions = [
            "Choose a narrower task or suite that more models completed.",
            "Rerun the missing task configs for the dropped model hashes, or relax the "
            "full-coverage requirement.",
        ]
    elif not keep_all:
        detail = "after deduping by model_hash"
        code = "insufficient_unique_models_after_dedupe"
        summary = (
            f"Only {len(ordered)} unique model(s) remain after collapsing to the latest "
            "run per model hash."
        )
        notes = [
            "The paired test uses one run per model hash by default so head-to-head cells "
            "do not mix multiple runs of the same checkpoint."
        ]
        suggestions = [
            "Broaden the filters to include more distinct model hashes.",
            "Keep repeated runs separate if you intentionally want multiple rows for the "
            "same model hash.",
        ]
    else:
        detail = "matched"
        code = "insufficient_compared_models"
        summary = (
            f"Only {len(ordered)} model(s) are available for pairwise comparison in this scope."
        )
        notes = []
        suggestions = ["Broaden the filters to include more models."]

    raise PairwiseEligibilityError(
        code=code,
        summary=summary,
        message=(
            f"Only {len(ordered)} unique model(s) {detail} — "
            "need at least 2. Broaden the filters to include more models."
        ),
        scope_label=scope_label,
        filter_summary=scope_str,
        counts=[
            {"label": "matched runs", "value": n_matched},
            {"label": "after latest-run dedupe", "value": len(selected_before_coverage)},
            {"label": "eligible compared models", "value": len(ordered)},
            {"label": "minimum required", "value": 2},
        ],
        matched_runs=[
            _serialize_experiment_debug(exp, keep_all=keep_all) for exp in candidate_experiments
        ],
        compared_models=[_serialize_experiment_debug(exp, keep_all=keep_all) for exp in selected],
        dropped_duplicate_runs=dropped_duplicate_runs,
        dropped_partial_coverage_models=[
            _serialize_filtered_model_debug(model) for model in filtered_models
        ],
        notes=notes,
        suggestions=suggestions,
    )


def _resolve_result_task_name(
    *,
    suite_name: str | None,
    task_hash: str | None,
    contributing_task_hashes: tuple[str, ...],
    task_hash_to_name: dict[str, str],
    fallback_task_name: str,
) -> str:
    if suite_name:
        return suite_name

    if not contributing_task_hashes:
        return fallback_task_name

    first_task_hash = contributing_task_hashes[0]
    task_name_for_hash = task_hash_to_name.get(first_task_hash, fallback_task_name)
    if task_hash is not None:
        return _format_task_variant_label(task_name_for_hash, first_task_hash)
    return task_name_for_hash


def compute_pairwise(
    session: Session,
    task_name: str | None = None,
    metric: str | None = None,
    margin: float = 0.0,
    experiment_ids: list[str] | None = None,
    model_names: list[str] | None = None,
    model_hashes: list[str] | None = None,
    exclude_model_names: list[str] | None = None,
    exclude_model_hashes: list[str] | None = None,
    task_hash: str | None = None,
    exclude_task_names: list[str] | None = None,
    exclude_task_hashes: list[str] | None = None,
    experiment_groups: list[str] | None = None,
    suite_name: str | None = None,
    keep_all: bool = False,
    require_full_coverage: bool = True,
) -> PairwiseResult:
    """Compute pairwise stats across the matched experiments.

    Scores are aligned by ``(task_hash, native_id)`` so differently configured
    variants of the same task name never get merged together.
    """
    from sqlalchemy import select

    from olmo_eval.runners.processing.utils import extract_score_from_metrics
    from olmo_eval.storage.backends.postgres.models import InstancePrediction, TaskResult
    from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

    scope_count = sum(bool(x) for x in (task_name, task_hash, suite_name))
    if scope_count != 1:
        raise ValueError(
            "Provide exactly one of task_name, task_hash, or suite_name to scope the comparison"
        )

    if task_name and _matches_exact(task_name, exclude_task_names):
        raise ValueError(
            f"Task '{task_name}' was excluded by --exclude-task. "
            "Remove the exclusion or choose a different scope."
        )
    if (
        task_hash
        and exclude_task_hashes
        and any(task_hash.startswith(excluded_hash) for excluded_hash in exclude_task_hashes)
    ):
        raise ValueError(
            f"Task hash prefix '{task_hash}' was excluded by --exclude-task-hash. "
            "Remove the exclusion or choose a different scope."
        )

    suite_task_names: tuple[str, ...] = ()
    if suite_name:
        from olmo_eval.evals.suites.registry import (
            get_suite,
            search_suites,
            suite_exists,
        )

        if not suite_exists(suite_name):
            hints = search_suites(suite_name)
            hint_str = f" Did you mean: {', '.join(hints)}?" if hints else ""
            raise ValueError(f"Suite '{suite_name}' not found.{hint_str}")
        suite_task_names = get_suite(suite_name).expand()
        suite_task_names = _filter_suite_task_names(suite_task_names, exclude_task_names)
        if not suite_task_names:
            raise ValueError(
                f"Suite '{suite_name}' resolved to zero tasks after applying --exclude-task "
                f"filters: {sorted(set(exclude_task_names or []))}"
            )

    task_names_filter: list[str] | None
    if suite_name:
        task_names_filter = list(suite_task_names)
    elif task_name:
        task_names_filter = [task_name]
    else:
        task_names_filter = None
    scope_label = suite_name or task_name or task_hash or ""

    scope_bits: list[str] = []
    if experiment_groups:
        scope_bits.append(f"groups={experiment_groups}")
    if model_names:
        scope_bits.append(f"models={model_names}")
    if model_hashes:
        scope_bits.append(f"hashes={model_hashes}")
    if exclude_model_names:
        scope_bits.append(f"exclude_models={exclude_model_names}")
    if exclude_model_hashes:
        scope_bits.append(f"exclude_hashes={exclude_model_hashes}")
    if experiment_ids:
        scope_bits.append(f"experiments={experiment_ids}")
    if suite_name:
        scope_bits.append(f"suite={suite_name!r} ({len(suite_task_names)} tasks)")
    elif task_name:
        scope_bits.append(f"task={task_name!r}")
    elif task_hash:
        scope_bits.append(f"task_hash={task_hash!r}")
    if exclude_task_names:
        scope_bits.append(f"exclude_tasks={exclude_task_names}")
    if exclude_task_hashes:
        scope_bits.append(f"exclude_task_hashes={exclude_task_hashes}")
    scope_str = ", ".join(scope_bits) if scope_bits else "(no filters)"

    repo = ExperimentRepository(session)
    candidate_experiments = repo.query_rows(
        experiment_ids=experiment_ids,
        model_names=model_names,
        model_hashes=model_hashes,
        task_names=task_names_filter,
        task_hashes=[task_hash] if task_hash else None,
        experiment_groups=experiment_groups,
    )
    candidate_experiments = [
        exp
        for exp in candidate_experiments
        if not _is_excluded_experiment(
            model_name=exp.model_name,
            model_hash=exp.model_hash,
            exclude_model_names=exclude_model_names,
            exclude_model_hashes=exclude_model_hashes,
        )
    ]

    if len(candidate_experiments) < 2:
        hint = ""
        if experiment_groups:
            hint = (
                f"\nTry: olmo-eval results group {experiment_groups[0]}"
                " to inspect the group's models and suite coverage."
            )
        raise PairwiseEligibilityError(
            code="insufficient_matched_experiments",
            summary=(
                f"Only {len(candidate_experiments)} run(s) matched the paired-test "
                "requirements for this scope."
            ),
            message=(
                f"Only {len(candidate_experiments)} experiment(s) matched the filters — "
                "need at least 2."
                f"\n  filters: {scope_str}{hint}"
            ),
            scope_label=scope_label,
            filter_summary=scope_str,
            counts=[
                {"label": "matched runs", "value": len(candidate_experiments)},
                {"label": "minimum required", "value": 2},
            ],
            matched_runs=[
                _serialize_experiment_debug(exp, keep_all=keep_all) for exp in candidate_experiments
            ],
            notes=[
                "The paired test only uses runs that match the selected group/model/task filters "
                "for the current scope."
            ],
            suggestions=[
                "Broaden the filters or choose a scope that at least two runs completed.",
                *(
                    [
                        f"Run `olmo-eval results group {experiment_groups[0]}` to inspect which "
                        "models and suites are present in the group."
                    ]
                    if experiment_groups
                    else []
                ),
            ],
        )

    n_matched = len(candidate_experiments)

    if keep_all:
        selected = sorted(candidate_experiments, key=lambda e: e.timestamp, reverse=True)
    else:
        chosen: dict[str, Any] = {}
        for exp in candidate_experiments:
            existing = chosen.get(exp.model_hash)
            if existing is None or exp.timestamp > existing.timestamp:
                chosen[exp.model_hash] = exp
        selected = list(chosen.values())

    n_dropped = n_matched - len(selected)
    selected_before_coverage = list(selected)
    selected_id_set_before_coverage = {exp.id for exp in selected_before_coverage}
    dropped_duplicate_runs = (
        [
            _serialize_experiment_debug(exp, keep_all=keep_all)
            for exp in candidate_experiments
            if exp.id not in selected_id_set_before_coverage
        ]
        if not keep_all
        else []
    )

    filtered_models: list[FilteredModel] = []
    ordered = _build_ordered_models(selected, keep_all=keep_all)
    pks = [pk for pk, _ in ordered]

    tr_stmt = select(
        TaskResult.experiment_pk,
        TaskResult.task_name,
        TaskResult.task_hash,
        TaskResult.num_instances,
        TaskResult.metrics,
        TaskResult.primary_metric,
    ).where(TaskResult.experiment_pk.in_(pks))
    if suite_name:
        tr_stmt = tr_stmt.where(TaskResult.task_name.in_(suite_task_names))
    elif task_name:
        tr_stmt = tr_stmt.where(TaskResult.task_name == task_name)
    elif task_hash:
        tr_stmt = tr_stmt.where(TaskResult.task_hash.startswith(task_hash))
    task_rows = session.execute(tr_stmt).all()
    task_rows = [
        row
        for row in task_rows
        if not _is_excluded_task(
            task_name=row.task_name,
            task_hash=row.task_hash,
            exclude_task_names=exclude_task_names,
            exclude_task_hashes=exclude_task_hashes,
        )
    ]

    if not task_rows:
        hint = ""
        notes = [
            "At least two runs matched the high-level filters, but none of the retained runs "
            "had task-result rows for this exact scope."
        ]
        if task_name:
            candidates = (
                session.execute(
                    select(TaskResult.task_name)
                    .where(TaskResult.experiment_pk.in_(pks))
                    .where(TaskResult.task_name.ilike(f"%{task_name}%"))
                    .distinct()
                    .limit(10)
                )
                .scalars()
                .all()
            )
            if candidates:
                hint = (
                    "\n--task uses exact matching. Similar task names in "
                    f"scope: {sorted(candidates)}"
                )
                notes.append(
                    "--task uses exact matching, so a nearby task name can still miss the scope."
                )
        if suite_name and experiment_groups:
            hint = (
                f"\nTry: olmo-eval results suites -G {experiment_groups[0]}"
                " to see which suites have coverage in this group."
            )
            notes.append(
                "The selected suite name is valid, but these retained runs do not cover it."
            )
        raise PairwiseEligibilityError(
            code="missing_task_rows",
            summary=f"No retained runs have task results for '{scope_label}'.",
            message="".join(
                [
                    f"No task results found for '{scope_label}' in the matched experiments",
                    (
                        " after applying exclusions"
                        if exclude_task_names or exclude_task_hashes
                        else ""
                    ),
                    ".",
                    hint,
                ]
            ),
            scope_label=scope_label,
            filter_summary=scope_str,
            counts=[
                {"label": "matched runs", "value": n_matched},
                {"label": "retained comparison models", "value": len(ordered)},
                {"label": "task-result rows", "value": 0},
            ],
            compared_models=[_serialize_model_meta_debug(meta) for _, meta in ordered],
            notes=notes,
            suggestions=[
                "Choose another suite or task that these runs actually completed.",
                *(
                    [
                        f"Run `olmo-eval results suites -G {experiment_groups[0]}` to inspect "
                        "suite coverage for the group."
                    ]
                    if suite_name and experiment_groups
                    else []
                ),
            ],
        )

    _validate_requested_task_scope(task_rows, task_name=task_name, task_hash=task_hash)
    selected, ordered, task_rows, filtered_models = _restrict_task_rows_to_selected_models(
        task_rows=task_rows,
        selected=selected,
        keep_all=keep_all,
        suite_task_names=suite_task_names if suite_name else (),
        require_full_coverage=require_full_coverage,
    )
    pks = [pk for pk, _ in ordered]

    if len(ordered) < 2:
        _raise_insufficient_compared_models(
            ordered=ordered,
            selected=selected,
            keep_all=keep_all,
            filtered_models=filtered_models,
            n_matched=n_matched,
            selected_before_coverage=selected_before_coverage,
            scope_label=scope_label,
            scope_str=scope_str,
            candidate_experiments=candidate_experiments,
            dropped_duplicate_runs=dropped_duplicate_runs,
        )

    task_hash_to_metric: dict[str, str] = {}
    task_hash_to_name: dict[str, str] = {}
    for task_row in task_rows:
        resolved_metric = metric if metric else task_row.primary_metric
        if not resolved_metric:
            raise ValueError(
                f"No primary_metric set for task '{task_row.task_name}' — "
                "specify --metric explicitly"
            )
        task_hash_to_metric[task_row.task_hash] = resolved_metric
        task_hash_to_name[task_row.task_hash] = task_row.task_name

    unique_task_hashes = set(task_hash_to_metric.keys())
    distinct_metrics = set(task_hash_to_metric.values())
    display_metric = (
        next(iter(distinct_metrics)) if len(distinct_metrics) == 1 else "per-task primary"
    )
    task_profile_by_hash = {
        task_hash: _get_task_metric_profile(
            task_hash_to_name[task_hash],
            task_hash_to_metric[task_hash],
        )
        for task_hash in unique_task_hashes
    }
    score_expr = _build_pairwise_score_sql_expr(task_hash_to_metric, task_profile_by_hash)
    rows = (
        session.execute(
            select(
                InstancePrediction.experiment_pk,
                InstancePrediction.native_id,
                InstancePrediction.task_hash,
                score_expr,
            ).where(
                InstancePrediction.experiment_pk.in_(pks),
                InstancePrediction.task_hash.in_(unique_task_hashes),
            )
        ).all()
        if score_expr is not None
        else []
    )

    raw_scores_by_pk: dict[int, dict[tuple[str, str], float]] = {pk: {} for pk in pks}
    comparison_scores_by_pk: dict[int, dict[tuple[str, str], float]] = {pk: {} for pk in pks}
    task_score_by_pk: dict[int, dict[str, float]] = {pk: {} for pk in pks}
    for task_row in task_rows:
        if task_row.experiment_pk not in task_score_by_pk:
            continue
        task_metric = task_hash_to_metric.get(task_row.task_hash)
        if task_metric is None:
            continue
        task_score = extract_score_from_metrics(task_row.metrics, task_metric)
        if task_score is not None:
            task_score_by_pk[task_row.experiment_pk][task_row.task_hash] = task_score

    for exp_pk, native_id, th, raw_score in rows:
        if exp_pk not in raw_scores_by_pk or exp_pk not in comparison_scores_by_pk:
            continue
        if raw_score is None or th not in task_hash_to_name:
            continue
        key = (str(th), str(native_id))
        raw_scores_by_pk[exp_pk][key] = raw_score
        comparison_scores_by_pk[exp_pk][key] = _comparison_score(
            raw_score,
            task_profile_by_hash.get(th),
        )

    # Ignore runs with no extractable instance scores.
    active: list[tuple[int, ModelMeta]] = []
    for pk, meta in ordered:
        if comparison_scores_by_pk[pk]:
            active.append((pk, meta))

    if len(active) < 2:
        instance_row_count = len(rows)
        scored_count = sum(1 for pk in pks if comparison_scores_by_pk[pk])
        sample_metric_keys: list[str] = []
        if rows:
            sample_instance_metrics = session.execute(
                select(InstancePrediction.instance_metrics)
                .where(
                    InstancePrediction.experiment_pk.in_(pks),
                    InstancePrediction.task_hash.in_(unique_task_hashes),
                )
                .limit(1)
            ).scalar_one_or_none()
            if isinstance(sample_instance_metrics, dict):
                sample_metric_keys = sorted(str(key) for key in sample_instance_metrics)
        unsupported_metrics = sorted(
            {
                f"{task_hash_to_name[task_hash]} ({task_hash_to_metric[task_hash]})"
                for task_hash in unique_task_hashes
                if (profile := task_profile_by_hash.get(task_hash)) is not None
                and not profile.supports_scorer_fallback
            }
        )
        unsupported_note = ""
        if unsupported_metrics:
            preview = ", ".join(unsupported_metrics[:4])
            if len(unsupported_metrics) > 4:
                preview = f"{preview}, +{len(unsupported_metrics) - 4} more"
            unsupported_note = (
                "\nThese tasks use derived or weighted aggregate metrics whose scorer-level "
                "instance values are not valid pairwise scores. Pairwise is only available "
                "when per-instance storage includes the exact metric key. "
                f"Affected tasks: {preview}"
            )
        sample_metrics = (
            f", sample instance_metrics keys: {sample_metric_keys}" if sample_metric_keys else ""
        )
        raise PairwiseEligibilityError(
            code="insufficient_extractable_instance_scores",
            summary=(
                f"Only {scored_count} model(s) have pairwise-eligible instance scores "
                f"for '{scope_label}'."
            ),
            message=(
                f"Only {scored_count} of {len(ordered)} experiment(s) have extractable "
                f"instance scores for '{scope_label}' using metric='{display_metric}' "
                f"(fetched {instance_row_count} instance rows from DB{sample_metrics})"
                f"{unsupported_note}"
            ),
            scope_label=scope_label,
            filter_summary=scope_str,
            counts=[
                {"label": "retained comparison models", "value": len(ordered)},
                {"label": "models with pairwise scores", "value": scored_count},
                {"label": "minimum required", "value": 2},
                {"label": "fetched instance rows", "value": instance_row_count},
            ],
            compared_models=[_serialize_model_meta_debug(meta) for _, meta in ordered],
            scored_models=[_serialize_model_meta_debug(meta) for _, meta in active],
            unscored_models=[
                _serialize_model_meta_debug(meta)
                for pk, meta in ordered
                if not comparison_scores_by_pk[pk]
            ],
            unsupported_task_metrics=unsupported_metrics,
            notes=[
                *(
                    [
                        "These tasks use aggregate metrics whose scorer-level instance values do "
                        "not equal the per-instance metric needed for pairwise math."
                    ]
                    if unsupported_metrics
                    else []
                ),
                *(
                    ["Sample stored instance metric keys: " + ", ".join(sample_metric_keys)]
                    if sample_metric_keys
                    else []
                ),
            ],
            suggestions=[
                "Store the exact per-instance metric key required by this scope, or choose "
                "a scope with pairwise-eligible instance scores.",
                "If these runs should be comparable, inspect the stored "
                "`instance_metrics` payload for the affected model hashes.",
            ],
        )

    models = [meta for _, meta in active]
    scores_by_idx: dict[int, dict[tuple[str, str], float]] = {}
    for idx, (pk, _) in enumerate(active):
        scores_by_idx[idx] = comparison_scores_by_pk[pk]

    id_sets = [set(scores.keys()) for scores in scores_by_idx.values()]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids = shared_ids & s

    if not shared_ids:
        per_model = sorted(
            ((models[i].label.replace("\n", " "), len(id_sets[i])) for i in range(len(active))),
            key=lambda t: t[1],
        )
        breakdown = "\n  ".join(f"{lbl}: {n} instances" for lbl, n in per_model)
        hint = (
            "\nIn suite mode this usually means models ran disjoint subsets "
            "of the suite's tasks — scope to a narrower suite or a single "
            "task that every model covered."
            if suite_name
            else ""
        )
        raise PairwiseEligibilityError(
            code="no_shared_instances",
            summary=(f"The retained models do not share any common instances for '{scope_label}'."),
            message=(
                f"No shared instances across the {len(active)} active model(s) "
                f"for '{scope_label}'. Per-model instance counts:\n  {breakdown}"
                f"{hint}"
            ),
            scope_label=scope_label,
            filter_summary=scope_str,
            counts=[
                {"label": "active models", "value": len(active)},
                {"label": "shared instances", "value": 0},
            ],
            compared_models=[_serialize_model_meta_debug(meta) for _, meta in active],
            per_model_instance_counts=[
                {"label": label, "instance_count": count} for label, count in per_model
            ],
            notes=[
                "Pairwise cells require the same instance ids to exist for every compared model."
            ],
            suggestions=[
                (
                    "Choose a narrower suite or a single task that all retained models completed."
                    if suite_name
                    else "Choose a scope where the retained models overlap on instance ids."
                ),
            ],
        )

    contributing_task_hashes = tuple(
        sorted(
            {task_hash for task_hash, _ in shared_ids},
            key=lambda resolved_hash: (task_hash_to_name.get(resolved_hash, ""), resolved_hash),
        )
    )
    contributing_task_hash_set = set(contributing_task_hashes)
    contributing_task_names = tuple(
        task_hash_to_name.get(task_hash, "") for task_hash in contributing_task_hashes
    )

    per_pk_task_counts: dict[int, int] = {}
    per_pk_cost_counts: dict[int, int] = {}
    per_pk_cost_totals: dict[int, float] = {}
    for task_row in task_rows:
        if (
            task_row.experiment_pk not in pks
            or task_row.task_hash not in contributing_task_hash_set
        ):
            continue
        per_pk_task_counts[task_row.experiment_pk] = (
            per_pk_task_counts.get(task_row.experiment_pk, 0) + 1
        )
        cost = _extract_cost_from_task_metrics(task_row.metrics)
        if cost is None:
            continue
        per_pk_cost_counts[task_row.experiment_pk] = (
            per_pk_cost_counts.get(task_row.experiment_pk, 0) + 1
        )
        per_pk_cost_totals[task_row.experiment_pk] = (
            per_pk_cost_totals.get(task_row.experiment_pk, 0.0) + cost
        )

    model_costs: list[float | None] = []
    for pk, _ in active:
        task_count = per_pk_task_counts.get(pk, 0)
        cost_count = per_pk_cost_counts.get(pk, 0)
        if task_count > 0 and task_count == cost_count:
            model_costs.append(per_pk_cost_totals[pk])
        else:
            model_costs.append(None)
    model_task_scores = [
        tuple(task_score_by_pk.get(pk, {}).get(task_hash) for task_hash in contributing_task_hashes)
        for pk, _ in active
    ]
    task_metric_keys = tuple(
        task_hash_to_metric.get(task_hash) for task_hash in contributing_task_hashes
    )
    contributing_profiles = [
        profile
        for task_hash in contributing_task_hashes
        if (profile := task_profile_by_hash.get(task_hash)) is not None
    ]
    units = {profile.unit for profile in contributing_profiles}
    display_formats = {profile.display_format for profile in contributing_profiles}
    directions = {profile.higher_is_better for profile in contributing_profiles}
    score_unit = next(iter(units)) if len(units) == 1 else None
    score_display_format = next(iter(display_formats)) if len(display_formats) == 1 else "mixed"
    higher_is_better = next(iter(directions)) if len(directions) == 1 else None
    model_shared_scores = (
        [
            (sum(raw_scores_by_pk[pk][key] for key in shared_ids) / len(shared_ids))
            if shared_ids
            else None
            for pk, _ in active
        ]
        if score_unit is not None
        else [None for _ in active]
    )

    pairs = _compute_pairs(scores_by_idx, len(active), shared_ids, margin)
    models, pairs, model_costs, model_task_scores, model_shared_scores = _order_by_overall_win_rate(
        models,
        pairs,
        model_costs=model_costs,
        model_task_scores=model_task_scores,
        model_shared_scores=model_shared_scores,
    )

    result_task_name = _resolve_result_task_name(
        suite_name=suite_name,
        task_hash=task_hash,
        contributing_task_hashes=contributing_task_hashes,
        task_hash_to_name=task_hash_to_name,
        fallback_task_name=task_rows[0].task_name,
    )

    return PairwiseResult(
        task_name=result_task_name,
        metric=display_metric,
        margin=margin,
        instance_count=len(shared_ids),
        models=models,
        pairs=pairs,
        suite_name=suite_name,
        task_names=contributing_task_names,
        task_hashes=contributing_task_hashes,
        n_experiments_matched=n_matched,
        n_experiments_dropped=n_dropped,
        filtered_models=tuple(filtered_models),
        model_costs=tuple(model_costs),
        task_metric_keys=task_metric_keys,
        model_task_scores=tuple(model_task_scores),
        model_shared_scores=tuple(model_shared_scores),
        score_display_format=score_display_format,
        score_unit=score_unit,
        higher_is_better=higher_is_better,
    )


def _order_by_overall_win_rate(
    models: list[ModelMeta],
    pairs: list[PairStats],
    model_costs: list[float | None] | None = None,
    model_task_scores: list[tuple[float | None, ...]] | None = None,
    model_shared_scores: list[float | None] | None = None,
) -> tuple[
    list[ModelMeta],
    list[PairStats],
    list[float | None],
    list[tuple[float | None, ...]],
    list[float | None],
]:
    """Order rows by overall win rate and remap pair indices."""
    n = len(models)
    wins: dict[int, int] = {i: 0 for i in range(n)}
    losses: dict[int, int] = {i: 0 for i in range(n)}
    for p in pairs:
        wins[p.index_a] += p.wins_a
        losses[p.index_a] += p.wins_b
        wins[p.index_b] += p.wins_b
        losses[p.index_b] += p.wins_a

    def _wr(i: int) -> float:
        total = wins[i] + losses[i]
        return wins[i] / total if total > 0 else 0.5

    order = sorted(range(n), key=_wr, reverse=True)
    old_to_new = {old: new for new, old in enumerate(order)}

    reordered_models = [models[old] for old in order]

    reordered_pairs: list[PairStats] = []
    for p in pairs:
        new_a = old_to_new[p.index_a]
        new_b = old_to_new[p.index_b]
        if new_a <= new_b:
            reordered_pairs.append(
                PairStats(
                    index_a=new_a,
                    index_b=new_b,
                    wins_a=p.wins_a,
                    wins_b=p.wins_b,
                    ties=p.ties,
                    var_paired_diff=p.var_paired_diff,
                    var_marginal_sum=p.var_marginal_sum,
                )
            )
        else:
            reordered_pairs.append(
                PairStats(
                    index_a=new_b,
                    index_b=new_a,
                    wins_a=p.wins_b,
                    wins_b=p.wins_a,
                    ties=p.ties,
                    var_paired_diff=p.var_paired_diff,
                    var_marginal_sum=p.var_marginal_sum,
                )
            )
    reordered_costs = [model_costs[old] for old in order] if model_costs is not None else []
    reordered_task_scores = (
        [model_task_scores[old] for old in order] if model_task_scores is not None else []
    )
    reordered_shared_scores = (
        [model_shared_scores[old] for old in order] if model_shared_scores is not None else []
    )
    return (
        reordered_models,
        reordered_pairs,
        reordered_costs,
        reordered_task_scores,
        reordered_shared_scores,
    )

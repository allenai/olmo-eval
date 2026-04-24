"""Shared pairwise metrics and table helpers for browser-oriented renderers."""

from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple

import numpy as np

from olmo_eval.analysis.pairwise import ModelMeta, PairwiseResult, get_task_metric_profile


class _PairCell(NamedTuple):
    win_rate: float
    se: float
    prob_a_gt_b: float


class RowMetrics(NamedTuple):
    rating: float
    dominance: int
    avg_win_rate: float
    best_task_label: str | None
    best_task_score: float | None
    worst_task_label: str | None
    worst_task_score: float | None


class TaskMetrics(NamedTuple):
    task_label: str
    median_score: float | None
    spread: float | None
    best_model_label: str | None
    best_model_score: float | None
    worst_model_label: str | None
    worst_model_score: float | None


type PairCellLookup = dict[tuple[int, int], _PairCell]

_SIGNIFICANCE_Z = 2.0


class TaskDisplayEntry(NamedTuple):
    id: str
    task_name: str
    task_hash: str | None
    label: str
    full_label: str


def _build_pair_cell_lookup(result: PairwiseResult) -> PairCellLookup:
    """Index every ordered matrix cell by its pairwise win rate and standard error."""
    pair_lookup: PairCellLookup = {}
    for pair in result.pairs:
        se = pair.se
        prob_a_gt_b = pair.prob_a_gt_b
        pair_lookup[(pair.index_a, pair.index_b)] = _PairCell(
            win_rate=pair.win_rate_a,
            se=se,
            prob_a_gt_b=prob_a_gt_b,
        )
        pair_lookup[(pair.index_b, pair.index_a)] = _PairCell(
            win_rate=pair.win_rate_b,
            se=se,
            prob_a_gt_b=1.0 - prob_a_gt_b,
        )
    return pair_lookup


def build_win_rate_matrix(
    result: PairwiseResult,
    *,
    pair_lookup: PairCellLookup | None = None,
) -> np.ndarray:
    """Return the NxN win-rate matrix with NaN on the diagonal."""
    n = len(result.models)
    matrix = np.full((n, n), 0.5)
    np.fill_diagonal(matrix, np.nan)
    if pair_lookup is None:
        pair_lookup = _build_pair_cell_lookup(result)
    for (row, col), pair_cell in pair_lookup.items():
        matrix[row, col] = pair_cell.win_rate
    return matrix


def build_se_matrix(
    result: PairwiseResult,
    *,
    pair_lookup: PairCellLookup | None = None,
) -> np.ndarray:
    """Return the NxN standard-error matrix with NaN on the diagonal."""
    n = len(result.models)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, np.nan)
    if pair_lookup is None:
        pair_lookup = _build_pair_cell_lookup(result)
    for (row, col), pair_cell in pair_lookup.items():
        matrix[row, col] = pair_cell.se
    return matrix


def build_probability_matrix(
    result: PairwiseResult,
    *,
    pair_lookup: PairCellLookup | None = None,
) -> np.ndarray:
    """Return the NxN matrix of approximate ``P(row > col)`` with NaN on the diagonal."""
    n = len(result.models)
    matrix = np.full((n, n), 0.5)
    np.fill_diagonal(matrix, np.nan)
    if pair_lookup is None:
        pair_lookup = _build_pair_cell_lookup(result)
    for (row, col), pair_cell in pair_lookup.items():
        matrix[row, col] = pair_cell.prob_a_gt_b
    return matrix


def _format_model_table_label(model: ModelMeta) -> str:
    if model.model_name and model.model_hash:
        return f"{model.model_name} ({model.model_hash[:8]})"
    return model.label.replace("\n", " ")


def _task_hash_at(task_hashes: tuple[str, ...], task_idx: int) -> str | None:
    if task_idx >= len(task_hashes):
        return None
    task_hash = task_hashes[task_idx]
    return task_hash or None


def _build_task_label_lookup(task_names: tuple[str, ...]) -> dict[str, str]:
    if not task_names:
        return {}

    if len(task_names) == 1:
        parts = task_names[0].split(":")
        if len(parts) > 1:
            return {task_names[0]: ":".join(parts[:-1])}
        return {task_names[0]: task_names[0]}

    split_names = [task_name.split(":") for task_name in task_names]
    prefix_len = 0
    while all(prefix_len < len(parts) for parts in split_names):
        token = split_names[0][prefix_len]
        if any(parts[prefix_len] != token for parts in split_names[1:]):
            break
        prefix_len += 1

    suffix_len = 0
    while all(suffix_len < len(parts) - prefix_len for parts in split_names):
        token = split_names[0][-(suffix_len + 1)]
        if any(parts[-(suffix_len + 1)] != token for parts in split_names[1:]):
            break
        suffix_len += 1

    labels: dict[str, str] = {}
    for task_name, parts in zip(task_names, split_names, strict=True):
        end = len(parts) - suffix_len if suffix_len else len(parts)
        core = parts[prefix_len:end]
        if not core:
            core = [parts[-1]]
        if len(core) == 1 and end == len(parts) and len(parts) > 1:
            core = parts[max(0, len(parts) - 2) : len(parts)]
        labels[task_name] = ":".join(core)

    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1

    return {
        task_name: (label if counts[label] == 1 else task_name)
        for task_name, label in labels.items()
    }


def build_task_display_entries(
    task_names: tuple[str, ...],
    task_hashes: tuple[str, ...] = (),
) -> list[TaskDisplayEntry]:
    if not task_names:
        return []

    label_lookup = _build_task_label_lookup(task_names)
    duplicate_name_counts = Counter(task_names)
    entries: list[TaskDisplayEntry] = []
    for task_idx, task_name in enumerate(task_names):
        task_hash = _task_hash_at(task_hashes, task_idx)
        task_id = task_hash or task_name
        hash_suffix = (
            f" [{task_hash[:8]}]"
            if task_hash is not None and duplicate_name_counts[task_name] > 1
            else ""
        )
        base_label = label_lookup.get(task_name, task_name)
        entries.append(
            TaskDisplayEntry(
                id=task_id,
                task_name=task_name,
                task_hash=task_hash,
                label=f"{base_label}{hash_suffix}",
                full_label=f"{task_name}{hash_suffix}",
            )
        )
    return entries


def _is_nonsignificant(win_rate: float, se: float) -> bool:
    return abs(win_rate - 0.5) <= _SIGNIFICANCE_Z * se


def _score_order_key(score: float, *, higher_is_better: bool) -> float:
    return score if higher_is_better else -score


def _task_higher_is_better_flags(result: PairwiseResult) -> list[bool]:
    default = True if result.higher_is_better is None else result.higher_is_better
    flags: list[bool] = []
    for task_idx, task_name in enumerate(result.task_names):
        metric_key = (
            result.task_metric_keys[task_idx] if task_idx < len(result.task_metric_keys) else None
        )
        profile = get_task_metric_profile(task_name, metric_key) if metric_key else None
        flags.append(profile.higher_is_better if profile is not None else default)
    return flags


def _build_overall_win_rate_vector(result: PairwiseResult) -> np.ndarray:
    n = len(result.models)
    wins = np.zeros(n, dtype=float)
    losses = np.zeros(n, dtype=float)
    for pair in result.pairs:
        wins[pair.index_a] += pair.wins_a
        losses[pair.index_a] += pair.wins_b
        wins[pair.index_b] += pair.wins_b
        losses[pair.index_b] += pair.wins_a
    totals = wins + losses
    win_rates = np.full(n, 0.5, dtype=float)
    np.divide(wins, totals, out=win_rates, where=totals > 0)
    return win_rates


def _build_bradley_terry_rating_vector(result: PairwiseResult) -> np.ndarray:
    n = len(result.models)
    if n == 0:
        return np.array([], dtype=float)

    wins = np.zeros((n, n), dtype=float)
    contested = np.zeros((n, n), dtype=float)
    for pair in result.pairs:
        wins[pair.index_a, pair.index_b] = pair.wins_a
        wins[pair.index_b, pair.index_a] = pair.wins_b
        contested[pair.index_a, pair.index_b] = pair.n_contested
        contested[pair.index_b, pair.index_a] = pair.n_contested

    abilities = np.ones(n, dtype=float)
    eps = 1e-6
    for _ in range(200):
        prev = abilities.copy()
        for i in range(n):
            numer = wins[i].sum()
            denom = 0.0
            for j in range(n):
                if i == j or contested[i, j] <= 0:
                    continue
                denom += contested[i, j] / max(abilities[i] + abilities[j], eps)
            abilities[i] = max(numer / denom, eps) if denom > 0 else eps
        abilities = np.clip(abilities, eps, None)
        abilities /= np.exp(np.mean(np.log(abilities)))
        prev = np.clip(prev, eps, None)
        prev /= np.exp(np.mean(np.log(prev)))
        if np.max(np.abs(np.log(abilities) - np.log(prev))) < 1e-8:
            break

    return 1500.0 + (400.0 / math.log(10.0)) * np.log(abilities)


def build_row_metrics(result: PairwiseResult) -> list[RowMetrics]:
    overall_win_rates = _build_overall_win_rate_vector(result)
    ratings = _build_bradley_terry_rating_vector(result)
    task_entries = build_task_display_entries(result.task_names, result.task_hashes)
    task_higher_is_better = _task_higher_is_better_flags(result)
    dominance = np.zeros(len(result.models), dtype=int)
    for pair in result.pairs:
        if _is_nonsignificant(pair.win_rate_a, pair.se):
            continue
        if pair.win_rate_a > 0.5:
            dominance[pair.index_a] += 1
            dominance[pair.index_b] -= 1
        elif pair.win_rate_a < 0.5:
            dominance[pair.index_a] -= 1
            dominance[pair.index_b] += 1

    row_metrics: list[RowMetrics] = []
    for idx in range(len(result.models)):
        task_scores = result.model_task_scores[idx] if idx < len(result.model_task_scores) else ()
        scored_tasks = [
            (task_idx, task_entry, score)
            for task_idx, (task_entry, score) in enumerate(
                zip(task_entries, task_scores, strict=False)
            )
            if score is not None
        ]
        best_task = (
            max(
                scored_tasks,
                key=lambda item: _score_order_key(
                    item[2],
                    higher_is_better=task_higher_is_better[item[0]],
                ),
            )
            if scored_tasks
            else None
        )
        worst_task = (
            min(
                scored_tasks,
                key=lambda item: _score_order_key(
                    item[2],
                    higher_is_better=task_higher_is_better[item[0]],
                ),
            )
            if scored_tasks
            else None
        )
        row_metrics.append(
            RowMetrics(
                rating=ratings[idx],
                dominance=int(dominance[idx]),
                avg_win_rate=overall_win_rates[idx],
                best_task_label=best_task[1].label if best_task else None,
                best_task_score=best_task[2] if best_task else None,
                worst_task_label=worst_task[1].label if worst_task else None,
                worst_task_score=worst_task[2] if worst_task else None,
            )
        )
    return row_metrics


def build_task_metrics(result: PairwiseResult) -> list[TaskMetrics]:
    model_labels = [_format_model_table_label(model) for model in result.models]
    task_entries = build_task_display_entries(result.task_names, result.task_hashes)
    task_higher_is_better = _task_higher_is_better_flags(result)

    task_metrics: list[TaskMetrics] = []
    for task_idx, task_entry in enumerate(task_entries):
        scored_models: list[tuple[str, float]] = []
        for model_idx, task_scores in enumerate(result.model_task_scores):
            if task_idx >= len(task_scores):
                continue
            score = task_scores[task_idx]
            if score is None:
                continue
            scored_models.append((model_labels[model_idx], float(score)))
        if not scored_models:
            task_metrics.append(
                TaskMetrics(
                    task_label=task_entry.full_label,
                    median_score=None,
                    spread=None,
                    best_model_label=None,
                    best_model_score=None,
                    worst_model_label=None,
                    worst_model_score=None,
                )
            )
            continue

        scores = [score for _, score in scored_models]
        higher_is_better = (
            task_higher_is_better[task_idx] if task_idx < len(task_higher_is_better) else True
        )
        best_model = max(
            scored_models,
            key=lambda item: _score_order_key(item[1], higher_is_better=higher_is_better),
        )
        worst_model = min(
            scored_models,
            key=lambda item: _score_order_key(item[1], higher_is_better=higher_is_better),
        )
        task_metrics.append(
            TaskMetrics(
                task_label=task_entry.full_label,
                median_score=float(np.median(scores)),
                spread=max(scores) - min(scores),
                best_model_label=best_model[0],
                best_model_score=best_model[1],
                worst_model_label=worst_model[0],
                worst_model_score=worst_model[1],
            )
        )
    return task_metrics

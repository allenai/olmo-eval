"""Shared payload builder for the results viewer UI."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from olmo_eval.analysis.eval_power import minimum_detectable_effect
from olmo_eval.analysis.pairwise import PairwiseResult, get_task_metric_profile
from olmo_eval.analysis.pairwise_metrics import (
    build_probability_matrix,
    build_row_metrics,
    build_se_matrix,
    build_task_display_entries,
    build_task_metrics,
    build_win_rate_matrix,
)


def _default_title(result: PairwiseResult) -> str:
    if result.suite_name:
        task_count = len(result.task_names)
        task_label = "task" if task_count == 1 else "tasks"
        return f"{result.suite_name} ({task_count} {task_label})"
    return f"{result.task_name} - {result.metric}"


def _display_model_label(model_index: int, result: PairwiseResult) -> str:
    if model_index >= len(result.models):
        return ""
    return result.models[model_index].label.replace("\n", " ")


def _matrix_to_payload(matrix: np.ndarray) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in matrix.tolist():
        rows.append(
            [
                (
                    None
                    if value is None or (isinstance(value, float) and np.isnan(value))
                    else float(value)
                )
                for value in row
            ]
        )
    return rows


def _empty_square(size: int) -> list[list[int | None]]:
    return [[None for _ in range(size)] for _ in range(size)]


def _build_count_matrices(result: PairwiseResult) -> dict[str, list[list[int | None]]]:
    size = len(result.models)
    wins = _empty_square(size)
    losses = _empty_square(size)
    contested = _empty_square(size)
    ties = _empty_square(size)

    for pair in result.pairs:
        a = pair.index_a
        b = pair.index_b
        wins[a][b] = pair.wins_a
        wins[b][a] = pair.wins_b
        losses[a][b] = pair.wins_b
        losses[b][a] = pair.wins_a
        contested[a][b] = pair.n_contested
        contested[b][a] = pair.n_contested
        ties[a][b] = pair.ties
        ties[b][a] = pair.ties

    return {
        "wins": wins,
        "losses": losses,
        "contested": contested,
        "ties": ties,
    }


def _build_score_diff_matrix(result: PairwiseResult) -> list[list[float | None]]:
    size = len(result.models)
    score_matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
    shared_scores = list(result.model_shared_scores)
    if result.score_unit is None:
        return score_matrix
    for row in range(size):
        for col in range(size):
            if row == col:
                continue
            row_score = shared_scores[row] if row < len(shared_scores) else None
            col_score = shared_scores[col] if col < len(shared_scores) else None
            if row_score is None or col_score is None:
                continue
            score_matrix[row][col] = row_score - col_score
    return score_matrix


def _build_p_value_matrix(result: PairwiseResult) -> list[list[float | None]]:
    size = len(result.models)
    matrix: list[list[float | None]] = [[None for _ in range(size)] for _ in range(size)]
    for pair in result.pairs:
        matrix[pair.index_a][pair.index_b] = pair.p_value
        matrix[pair.index_b][pair.index_a] = pair.p_value
    return matrix


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _storage_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "pairwise"


def _build_matrix_mde80_map(result: PairwiseResult) -> dict[str, float]:
    if result.instance_count <= 0 or not result.pairs or result.score_unit is None:
        return {}

    paired_vars = [float(pair.var_paired_diff) for pair in result.pairs]
    if not paired_vars:
        return {}

    median_paired_var = float(np.median(np.asarray(paired_vars, dtype=float)))
    alpha_options = (0.10, 0.05, 0.01, 0.001)
    return {
        str(alpha): minimum_detectable_effect(
            n=result.instance_count,
            omega2=median_paired_var,
            alpha=alpha,
            power=0.80,
        )
        for alpha in alpha_options
    }


def build_pairwise_viewer_payload(
    result: PairwiseResult, title: str | None = None
) -> dict[str, Any]:
    page_title = title or _default_title(result)
    row_metrics = build_row_metrics(result)
    task_metrics = build_task_metrics(result) if result.task_names else []
    task_entries = build_task_display_entries(result.task_names, result.task_hashes)
    probability_matrix = build_probability_matrix(result)
    win_rate_matrix = build_win_rate_matrix(result)
    se_matrix = build_se_matrix(result)
    count_matrices = _build_count_matrices(result)
    score_diff_matrix = _build_score_diff_matrix(result)
    mde80_by_alpha = _build_matrix_mde80_map(result)

    task_columns = []
    for task_idx, task_entry in enumerate(task_entries):
        metric_key = (
            result.task_metric_keys[task_idx] if task_idx < len(result.task_metric_keys) else None
        )
        profile = get_task_metric_profile(task_entry.task_name, metric_key) if metric_key else None
        task_columns.append(
            {
                "id": task_entry.id,
                "task_name": task_entry.task_name,
                "task_hash": task_entry.task_hash,
                "label": task_entry.label,
                "full_label": task_entry.full_label,
                "score_display_format": (
                    profile.display_format if profile is not None else result.score_display_format
                ),
                "score_unit": profile.unit if profile is not None else result.score_unit,
                "higher_is_better": (
                    profile.higher_is_better if profile is not None else result.higher_is_better
                ),
            }
        )

    models: list[dict[str, Any]] = []
    for index, model in enumerate(result.models):
        task_scores = (
            list(result.model_task_scores[index]) if index < len(result.model_task_scores) else []
        )
        task_score_map = {
            task_entry.id: task_scores[task_idx] if task_idx < len(task_scores) else None
            for task_idx, task_entry in enumerate(task_entries)
        }
        scored_values = [float(score) for score in task_scores if score is not None]
        avg_task_score = _mean(scored_values)
        shared_score = (
            result.model_shared_scores[index] if index < len(result.model_shared_scores) else None
        )
        metrics = row_metrics[index] if index < len(row_metrics) else None
        model_cost = result.model_costs[index] if index < len(result.model_costs) else None
        models.append(
            {
                "index": index,
                "display_label": _display_model_label(index, result),
                "model_name": model.model_name or model.label.replace("\n", " "),
                "model_hash": model.model_hash,
                "model_hash_short": (model.model_hash or "")[:8],
                "timestamp": model.timestamp,
                "shared_score": shared_score,
                "avg_task_score": avg_task_score,
                "display_score": shared_score if shared_score is not None else None,
                "strength": metrics.rating if metrics is not None else None,
                "avg_win_rate": metrics.avg_win_rate if metrics is not None else None,
                "dominance": metrics.dominance if metrics is not None else None,
                "best_task_label": metrics.best_task_label if metrics is not None else None,
                "best_task_score": metrics.best_task_score if metrics is not None else None,
                "worst_task_label": metrics.worst_task_label if metrics is not None else None,
                "worst_task_score": metrics.worst_task_score if metrics is not None else None,
                "cost": model_cost,
                "task_scores": task_score_map,
            }
        )

    task_stats: list[dict[str, Any]] = []
    for task_entry, metric in zip(task_entries, task_metrics, strict=False):
        task_stats.append(
            {
                "id": task_entry.id,
                "task_name": task_entry.task_name,
                "task_hash": task_entry.task_hash,
                "label": task_entry.label,
                "full_label": task_entry.full_label,
                "median_score": metric.median_score,
                "spread": metric.spread,
                "best_model_label": metric.best_model_label,
                "best_model_score": metric.best_model_score,
                "worst_model_label": metric.worst_model_label,
                "worst_model_score": metric.worst_model_score,
            }
        )

    return {
        "meta": {
            "title": page_title,
            "scope_label": result.suite_name or result.task_name,
            "scope_kind": "suite" if result.suite_name else "task",
            "metric": result.metric,
            "shared_n": result.instance_count,
            "task_count": len(result.task_names),
            "model_count": len(result.models),
            "margin": result.margin,
            "mde80": mde80_by_alpha.get("0.05"),
            "mde80_by_alpha": mde80_by_alpha,
            "score_display_format": result.score_display_format,
            "score_unit": result.score_unit,
            "higher_is_better": result.higher_is_better,
            "score_scale_comparable": result.score_unit is not None,
            "matched_experiments": result.n_experiments_matched,
            "dropped_experiments": result.n_experiments_dropped,
            "has_costs": any(cost is not None for cost in result.model_costs),
            "storage_key": _storage_key(page_title),
        },
        "models": models,
        "task_columns": task_columns,
        "task_stats": task_stats,
        "matrix": {
            "win_rate": _matrix_to_payload(win_rate_matrix),
            "se": _matrix_to_payload(se_matrix),
            "probability": _matrix_to_payload(probability_matrix),
            "p_value": _build_p_value_matrix(result),
            "score_diff": score_diff_matrix,
            **count_matrices,
        },
    }

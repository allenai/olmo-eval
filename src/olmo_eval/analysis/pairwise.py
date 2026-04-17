"""Pairwise comparison of instance-level scores across experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ModelMeta:
    """Metadata for a model in the pairwise comparison."""

    label: str


@dataclass(frozen=True)
class PairStats:
    """Win/loss/tie counts for a single ordered pair of models.

    ``index_a`` and ``index_b`` refer to positions in the
    ``PairwiseResult.models`` list (index_a < index_b).
    """

    index_a: int
    index_b: int
    wins_a: int
    wins_b: int
    ties: int

    @property
    def win_rate_a(self) -> float:
        contested = self.wins_a + self.wins_b
        return self.wins_a / contested if contested > 0 else 0.5

    @property
    def win_rate_b(self) -> float:
        return 1.0 - self.win_rate_a


@dataclass
class PairwiseResult:
    """Complete result of a pairwise comparison across N models."""

    task_name: str
    metric: str
    margin: float
    instance_count: int
    models: list[ModelMeta]
    pairs: list[PairStats]


def get_win_rate(pairs: list[PairStats], row: int, col: int) -> float:
    """Look up the win rate for models[row] vs models[col]."""
    for p in pairs:
        if p.index_a == row and p.index_b == col:
            return p.win_rate_a
        if p.index_a == col and p.index_b == row:
            return p.win_rate_b
    return 0.5


def _compute_pairs(
    scores_by_idx: dict[int, dict[str, float]],
    n: int,
    shared_ids: set[str],
    margin: float,
) -> list[PairStats]:
    """Compute pairwise win/loss/tie stats from pre-fetched scores.

    Args:
        scores_by_idx: Mapping of model index -> {native_id: score}.
        n: Number of models.
        shared_ids: Set of native_ids present in all experiments.
        margin: Tie threshold — scores within this margin are ties.

    Returns:
        One PairStats per unique (i, j) pair where i < j.
    """
    results: list[PairStats] = []
    for i in range(n):
        for j in range(i + 1, n):
            wins_a = 0
            wins_b = 0
            ties = 0
            for native_id in shared_ids:
                score_a = scores_by_idx[i].get(native_id)
                score_b = scores_by_idx[j].get(native_id)
                if score_a is None or score_b is None:
                    continue
                diff = score_a - score_b
                if abs(diff) <= margin:
                    ties += 1
                elif diff > 0:
                    wins_a += 1
                else:
                    wins_b += 1
            results.append(
                PairStats(
                    index_a=i,
                    index_b=j,
                    wins_a=wins_a,
                    wins_b=wins_b,
                    ties=ties,
                )
            )
    return results


def compute_pairwise(
    session: Session,
    task_name: str,
    metric: str | None = None,
    margin: float = 0.0,
    experiment_ids: list[str] | None = None,
    model_names: list[str] | None = None,
    experiment_groups: list[str] | None = None,
) -> PairwiseResult:
    """Compute pairwise win/loss/tie comparison across experiments.

    Discovers experiments using the same filter pattern as ``results query``,
    then fetches instance-level scores and computes head-to-head win rates on
    shared instances.

    Args:
        session: Active SQLAlchemy Session.
        task_name: Task name to scope the comparison.
        metric: Metric in "metric_name:scorer_name" format.  If None, uses
            the task's primary_metric.
        margin: Tie threshold for continuous metrics (default 0.0).
        experiment_ids: Filter by experiment ID strings.
        model_names: Filter by model name prefixes.
        experiment_groups: Filter by experiment group prefixes.

    Returns:
        PairwiseResult with model metadata and pairwise stats.

    Raises:
        ValueError: If fewer than 2 experiments match, or if the task is not
            found in any matched experiment.
    """
    from sqlalchemy import select

    from olmo_eval.runners.processing.utils import extract_score_from_metrics
    from olmo_eval.storage.backends.postgres.models import (
        Experiment,
        InstancePrediction,
        TaskResult,
    )
    from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

    repo = ExperimentRepository(session)
    eval_results = repo.query(
        experiment_ids=experiment_ids,
        model_names=model_names,
        experiment_groups=experiment_groups,
        task_names=[task_name],
    )

    if len(eval_results) < 2:
        raise ValueError(
            f"Need at least 2 experiments, but only {len(eval_results)} matched the filters"
        )

    # Resolve each EvalResult back to its Experiment PK via experiment_id + model_hash.
    # EvalResult doesn't carry the PK, so re-fetch the Experiment rows.
    experiment_id_list = [r.experiment_id for r in eval_results]
    model_hash_list = [r.model_hash for r in eval_results]
    experiments = (
        session.execute(
            select(Experiment).where(
                Experiment.experiment_id.in_(experiment_id_list),
                Experiment.model_hash.in_(model_hash_list),
            )
        )
        .scalars()
        .all()
    )

    # Build a lookup by (experiment_id, model_hash) -> Experiment
    exp_lookup: dict[tuple[str, str], Experiment] = {}
    for exp in experiments:
        key = (exp.experiment_id, exp.model_hash)
        if key not in exp_lookup:
            exp_lookup[key] = exp

    # Build ordered list of (PK, label) matching eval_results order
    ordered: list[tuple[int, str]] = []
    for r in eval_results:
        if r.model_hash is None:
            continue
        exp = exp_lookup.get((r.experiment_id, r.model_hash))
        if exp is None:
            continue
        ordered.append((exp.id, exp.model_name))

    if len(ordered) < 2:
        raise ValueError(
            f"Need at least 2 experiments, but only {len(ordered)} resolved successfully"
        )

    pks = [pk for pk, _ in ordered]
    models = [ModelMeta(label=label) for _, label in ordered]

    # --- Resolve task_hash and metric ---
    task_results = (
        session.execute(
            select(TaskResult).where(
                TaskResult.experiment_pk.in_(pks),
                TaskResult.task_name == task_name,
            )
        )
        .scalars()
        .all()
    )
    if not task_results:
        raise ValueError(f"No task results found for task '{task_name}' in matched experiments")

    task_hash = task_results[0].task_hash
    if metric is None:
        metric = task_results[0].primary_metric
        if metric is None:
            raise ValueError(
                f"No primary_metric set for task '{task_name}' — specify --metric explicitly"
            )

    # --- Fetch all instance scores in one query ---
    rows = session.execute(
        select(
            InstancePrediction.experiment_pk,
            InstancePrediction.native_id,
            InstancePrediction.instance_metrics,
        ).where(
            InstancePrediction.experiment_pk.in_(pks),
            InstancePrediction.task_hash == task_hash,
        )
    ).all()

    # --- Extract scores, group by model index ---
    pk_to_idx = {pk: idx for idx, pk in enumerate(pks)}
    scores_by_idx: dict[int, dict[str, float]] = {i: {} for i in range(len(pks))}
    for exp_pk, native_id, instance_metrics in rows:
        idx = pk_to_idx.get(exp_pk)
        if idx is None:
            continue
        score = extract_score_from_metrics(instance_metrics, metric)
        if score is not None:
            scores_by_idx[idx][native_id] = score

    # --- Intersect to shared instances ---
    id_sets = [set(scores.keys()) for scores in scores_by_idx.values()]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids = shared_ids & s

    # --- Compute pairs and return ---
    pairs = _compute_pairs(scores_by_idx, len(pks), shared_ids, margin)

    return PairwiseResult(
        task_name=task_name,
        metric=metric,
        margin=margin,
        instance_count=len(shared_ids),
        models=models,
        pairs=pairs,
    )

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
    task_name: str | None = None,
    metric: str | None = None,
    margin: float = 0.0,
    experiment_ids: list[str] | None = None,
    model_names: list[str] | None = None,
    model_hashes: list[str] | None = None,
    task_hash: str | None = None,
    experiment_groups: list[str] | None = None,
) -> PairwiseResult:
    """Compute pairwise win/loss/tie comparison across experiments.

    Discovers experiments using the same filter pattern as ``results query``,
    then fetches instance-level scores and computes head-to-head win rates on
    shared instances.

    Provide ``task_name`` or ``task_hash`` (not both) to scope the comparison.

    Args:
        session: Active SQLAlchemy Session.
        task_name: Task name to scope the comparison.
        metric: Metric in "metric_name:scorer_name" format.  If None, uses
            the task's primary_metric.
        margin: Tie threshold for continuous metrics (default 0.0).
        experiment_ids: Filter by experiment ID strings.
        model_names: Filter by model name prefixes.
        model_hashes: Filter by model hash prefixes.
        task_hash: Task hash prefix to filter by.
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

    if not task_name and not task_hash:
        raise ValueError("Provide task_name or task_hash to scope the comparison")

    repo = ExperimentRepository(session)
    eval_results = repo.query(
        experiment_ids=experiment_ids,
        model_names=model_names,
        model_hashes=model_hashes,
        task_names=[task_name] if task_name else None,
        task_hashes=[task_hash] if task_hash else None,
        experiment_groups=experiment_groups,
    )

    if len(eval_results) < 2:
        raise ValueError(
            f"Need at least 2 experiments, but only {len(eval_results)} matched the filters"
        )

    # Resolve EvalResults back to Experiment PKs. EvalResult doesn't carry the
    # PK, so re-fetch Experiment rows using the (experiment_id, model_hash) pairs.
    experiment_id_list = [r.experiment_id for r in eval_results]
    model_hash_list = [r.model_hash for r in eval_results if r.model_hash is not None]
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

    exp_lookup: dict[tuple[str, str], Experiment] = {}
    for exp in experiments:
        key = (exp.experiment_id, exp.model_hash)
        if key not in exp_lookup:
            exp_lookup[key] = exp

    ordered: list[tuple[int, str]] = []
    for r in eval_results:
        if r.model_hash is None:
            continue
        exp = exp_lookup.get((r.experiment_id, r.model_hash))
        if exp is None:
            continue
        label = f"{exp.model_name}\n({exp.model_hash[:8]})"
        ordered.append((exp.id, label))

    if len(ordered) < 2:
        raise ValueError(
            f"Need at least 2 experiments, but only {len(ordered)} resolved successfully"
        )

    pks = [pk for pk, _ in ordered]

    # --- Resolve task_hash, task_name, and metric ---
    tr_stmt = select(TaskResult).where(TaskResult.experiment_pk.in_(pks))
    if task_name:
        tr_stmt = tr_stmt.where(TaskResult.task_name == task_name)
    if task_hash:
        tr_stmt = tr_stmt.where(TaskResult.task_hash.startswith(task_hash))
    task_results = session.execute(tr_stmt).scalars().all()

    task_label = task_name or task_hash or ""
    if not task_results:
        raise ValueError(f"No task results found for '{task_label}' in matched experiments")

    resolved_task_hash = task_results[0].task_hash
    resolved_task_name = task_results[0].task_name
    if metric is None:
        metric = task_results[0].primary_metric
        if metric is None:
            raise ValueError(
                f"No primary_metric set for task '{task_label}' — specify --metric explicitly"
            )

    # --- Fetch all instance scores in one query ---
    rows = session.execute(
        select(
            InstancePrediction.experiment_pk,
            InstancePrediction.native_id,
            InstancePrediction.instance_metrics,
        ).where(
            InstancePrediction.experiment_pk.in_(pks),
            InstancePrediction.task_hash == resolved_task_hash,
        )
    ).all()

    # --- Extract scores, group by experiment PK ---
    #
    # Instance-level metrics are stored as {scorer: {scorer: value}} (see
    # runners/io/builders.py), while task-level primary_metric uses the
    # "metric:scorer" format.  Try the task-level format first; if it misses,
    # fall back to the instance-level convention (scorer as both keys).
    from olmo_eval.runners.processing.utils import parse_metric_key

    instance_metric_key = metric
    parsed = parse_metric_key(metric)
    if parsed:
        scorer = parsed[1]
        instance_metric_key = f"{scorer}:{scorer}"

    scores_by_pk: dict[int, dict[str, float]] = {pk: {} for pk in pks}
    for exp_pk, native_id, instance_metrics in rows:
        if exp_pk not in scores_by_pk:
            continue
        # Try the instance-level key first, then the raw metric key.
        score = extract_score_from_metrics(instance_metrics, instance_metric_key)
        if score is None:
            score = extract_score_from_metrics(instance_metrics, metric)
        if score is not None:
            scores_by_pk[exp_pk][native_id] = score

    # --- Drop experiments that have no instance scores (e.g. instances not
    # stored for that run) so they don't zero-out the shared intersection. ---
    active: list[tuple[int, str]] = []
    for pk, label in ordered:
        if scores_by_pk[pk]:
            active.append((pk, label))

    if len(active) < 2:
        # Build diagnostic info to help identify the root cause.
        instance_row_count = len(rows)
        scored_count = sum(1 for pk in pks if scores_by_pk[pk])
        sample_metrics = ""
        if rows:
            sample_metrics = f", sample instance_metrics keys: {list(rows[0][2].keys())}"
        raise ValueError(
            f"Only {scored_count} of {len(ordered)} experiment(s) have extractable "
            f"instance scores for '{task_label}' using metric='{metric}' "
            f"(fetched {instance_row_count} instance rows from DB{sample_metrics})"
        )

    # --- Rebuild index mapping for the active set ---
    models = [ModelMeta(label=label) for _, label in active]
    scores_by_idx: dict[int, dict[str, float]] = {}
    for idx, (pk, _) in enumerate(active):
        scores_by_idx[idx] = scores_by_pk[pk]

    # --- Intersect to shared instances ---
    id_sets = [set(scores.keys()) for scores in scores_by_idx.values()]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids = shared_ids & s

    # --- Compute pairs and return ---
    pairs = _compute_pairs(scores_by_idx, len(active), shared_ids, margin)

    return PairwiseResult(
        task_name=resolved_task_name,
        metric=metric,
        margin=margin,
        instance_count=len(shared_ids),
        models=models,
        pairs=pairs,
    )

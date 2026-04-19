"""Pairwise comparison of instance-level scores across experiments.

Uncertainty is reported as a standard error via the CLT:
``SE = sqrt(sample_variance / n)``. For the binary win/loss scores produced
here this reduces to ``sqrt(p(1-p) / (n-1))``, but the general form keeps the
approach reusable for fractional-score metrics without bootstrapping.
"""

from __future__ import annotations

import math
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

    ``var_paired_diff`` is the per-question sample variance of
    ``d_i = score_a_i - score_b_i`` across shared instances. ``var_marginal_sum``
    is ``sigma_A^2 + sigma_B^2``, the sum of each model's per-instance score
    variance on the same instances. Both feed the CLT power helpers in
    ``eval_power``; their ratio gives the paired design effect.
    """

    index_a: int
    index_b: int
    wins_a: int
    wins_b: int
    ties: int
    var_paired_diff: float = 0.0
    var_marginal_sum: float = 0.0

    @property
    def win_rate_a(self) -> float:
        contested = self.wins_a + self.wins_b
        return self.wins_a / contested if contested > 0 else 0.5

    @property
    def win_rate_b(self) -> float:
        return 1.0 - self.win_rate_a

    @property
    def se(self) -> float:
        """Standard error of the win rate via CLT: sqrt(sample_variance / n).

        Ties are excluded from ``n``. For binary 0/1 scores (ties excluded)
        the sample variance is ``n / (n-1) * p * (1-p)``; the general CLT
        form is kept so the same shape works for fractional metrics later.
        Symmetric: ``se(a vs b) == se(b vs a)``.
        """
        n = self.wins_a + self.wins_b
        if n <= 1:
            return 0.0
        p = self.wins_a / n
        sample_var = n / (n - 1) * p * (1 - p)
        return math.sqrt(sample_var / n)


@dataclass
class PairwiseResult:
    """Complete result of a pairwise comparison across N models.

    ``task_name`` is the primary display label — the concrete task name in
    single-task mode, or the suite name in suite mode. ``suite_name`` is
    populated only when the comparison pooled instances across a suite, and
    ``task_names`` lists the concrete tasks that contributed instances.
    """

    task_name: str
    metric: str
    margin: float
    instance_count: int
    models: list[ModelMeta]
    pairs: list[PairStats]
    suite_name: str | None = None
    task_names: tuple[str, ...] = ()


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


def _compute_pairs(
    scores_by_idx: dict[int, dict[tuple[str, str], float]],
    n: int,
    shared_ids: set[tuple[str, str]],
    margin: float,
) -> list[PairStats]:
    """Compute pairwise win/loss/tie stats from pre-fetched scores.

    Args:
        scores_by_idx: Mapping of model index -> {(task_name, native_id): score}.
        n: Number of models.
        shared_ids: Set of (task_name, native_id) keys present in all experiments.
        margin: Tie threshold — scores within this margin are ties.

    Returns:
        One PairStats per unique (i, j) pair where i < j.
    """

    def _sample_var(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    results: list[PairStats] = []
    for i in range(n):
        for j in range(i + 1, n):
            wins_a = 0
            wins_b = 0
            ties = 0
            diffs: list[float] = []
            scores_a_list: list[float] = []
            scores_b_list: list[float] = []
            for key in shared_ids:
                score_a = scores_by_idx[i].get(key)
                score_b = scores_by_idx[j].get(key)
                if score_a is None or score_b is None:
                    continue
                diff = score_a - score_b
                diffs.append(diff)
                scores_a_list.append(score_a)
                scores_b_list.append(score_b)
                if abs(diff) <= margin:
                    ties += 1
                elif diff > 0:
                    wins_a += 1
                else:
                    wins_b += 1
            var_d = _sample_var(diffs)
            var_marginal = _sample_var(scores_a_list) + _sample_var(scores_b_list)
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
    suite_name: str | None = None,
) -> PairwiseResult:
    """Compute pairwise win/loss/tie comparison across experiments.

    Discovers experiments using the same filter pattern as ``results query``,
    then fetches instance-level scores and computes head-to-head win rates on
    shared instances.

    Provide exactly one of ``task_name``, ``task_hash``, or ``suite_name`` to
    scope the comparison. In suite mode, instances from every task in the suite
    are pooled into a single paired comparison keyed by (task_hash, native_id).

    Args:
        session: Active SQLAlchemy Session.
        task_name: Task name to scope the comparison.
        metric: Metric in "metric_name:scorer_name" format.  If None, uses
            each task's primary_metric.
        margin: Tie threshold for continuous metrics (default 0.0).
        experiment_ids: Filter by experiment ID strings.
        model_names: Filter by model name prefixes.
        model_hashes: Filter by model hash prefixes.
        task_hash: Task hash prefix to filter by.
        experiment_groups: Filter by experiment group prefixes.
        suite_name: Name of a registered suite (e.g. ``olmobase:math``).
            Pools instances across every task the suite resolves to.

    Returns:
        PairwiseResult with model metadata and pairwise stats.

    Raises:
        ValueError: If fewer than 2 experiments match, or if the task / suite is
            not found in any matched experiment.
    """
    from sqlalchemy import select

    from olmo_eval.runners.processing.utils import extract_score_from_metrics
    from olmo_eval.storage.backends.postgres.models import (
        Experiment,
        InstancePrediction,
        TaskResult,
    )
    from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

    scope_count = sum(bool(x) for x in (task_name, task_hash, suite_name))
    if scope_count != 1:
        raise ValueError(
            "Provide exactly one of task_name, task_hash, or suite_name to scope the comparison"
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
        if not suite_task_names:
            raise ValueError(f"Suite '{suite_name}' resolved to zero tasks")

    task_names_filter: list[str] | None
    if suite_name:
        task_names_filter = list(suite_task_names)
    elif task_name:
        task_names_filter = [task_name]
    else:
        task_names_filter = None

    repo = ExperimentRepository(session)
    eval_results = repo.query(
        experiment_ids=experiment_ids,
        model_names=model_names,
        model_hashes=model_hashes,
        task_names=task_names_filter,
        task_hashes=[task_hash] if task_hash else None,
        experiment_groups=experiment_groups,
    )

    if len(eval_results) < 2:
        scope_bits: list[str] = []
        if experiment_groups:
            scope_bits.append(f"groups={experiment_groups}")
        if model_names:
            scope_bits.append(f"models={model_names}")
        if model_hashes:
            scope_bits.append(f"hashes={model_hashes}")
        if experiment_ids:
            scope_bits.append(f"experiments={experiment_ids}")
        if suite_name:
            scope_bits.append(f"suite={suite_name!r} ({len(suite_task_names)} tasks)")
        elif task_name:
            scope_bits.append(f"task={task_name!r}")
        elif task_hash:
            scope_bits.append(f"task_hash={task_hash!r}")
        scope_str = ", ".join(scope_bits) if scope_bits else "(no filters)"
        hint = ""
        if experiment_groups:
            hint = (
                f"\nTry: olmo-eval results group {experiment_groups[0]}"
                " to inspect the group's models and suite coverage."
            )
        raise ValueError(
            f"Only {len(eval_results)} experiment(s) matched the filters — need at least 2."
            f"\n  filters: {scope_str}{hint}"
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

    # Dedupe by (model_name, model_hash), keeping the most recent experiment.
    # Re-runs of the same model/hash under different experiment_ids would
    # otherwise produce duplicate rows in the matrix.
    chosen: dict[tuple[str, str], Experiment] = {}
    for r in eval_results:
        if r.model_hash is None:
            continue
        exp = exp_lookup.get((r.experiment_id, r.model_hash))
        if exp is None:
            continue
        key = (exp.model_name, exp.model_hash)
        existing = chosen.get(key)
        if existing is None or exp.timestamp > existing.timestamp:
            chosen[key] = exp

    ordered: list[tuple[int, str]] = []
    for exp in chosen.values():
        label = f"{exp.model_name}\n({exp.model_hash[:8]})"
        ordered.append((exp.id, label))

    if len(ordered) < 2:
        raise ValueError(
            f"Only {len(ordered)} unique model(s) after deduping by (model_name, model_hash) — "
            "need at least 2. Broaden the filters to include more models."
        )

    pks = [pk for pk, _ in ordered]

    # --- Resolve task_hash, task_name, and metric ---
    tr_stmt = select(TaskResult).where(TaskResult.experiment_pk.in_(pks))
    if suite_name:
        tr_stmt = tr_stmt.where(TaskResult.task_name.in_(suite_task_names))
    elif task_name:
        tr_stmt = tr_stmt.where(TaskResult.task_name == task_name)
    elif task_hash:
        tr_stmt = tr_stmt.where(TaskResult.task_hash.startswith(task_hash))
    task_results = session.execute(tr_stmt).scalars().all()

    scope_label = suite_name or task_name or task_hash or ""
    if not task_results:
        hint = ""
        if suite_name and experiment_groups:
            hint = (
                f"\nTry: olmo-eval results suites -G {experiment_groups[0]}"
                " to see which suites have coverage in this group."
            )
        raise ValueError(
            f"No task results found for '{scope_label}' in the matched experiments.{hint}"
        )

    # Build a per-task_hash metric map. User-supplied --metric overrides per-task
    # primary_metric; in suite mode this lets each task use its own default.
    task_hash_to_metric: dict[str, str] = {}
    task_hash_to_name: dict[str, str] = {}
    for tr in task_results:
        resolved_metric = metric if metric else tr.primary_metric
        if not resolved_metric:
            raise ValueError(
                f"No primary_metric set for task '{tr.task_name}' — specify --metric explicitly"
            )
        task_hash_to_metric[tr.task_hash] = resolved_metric
        task_hash_to_name[tr.task_hash] = tr.task_name

    unique_task_hashes = set(task_hash_to_metric.keys())
    # Representative display metric: concrete value if unique across tasks,
    # otherwise the sentinel "per-task primary" for the summary title.
    distinct_metrics = set(task_hash_to_metric.values())
    display_metric = (
        next(iter(distinct_metrics)) if len(distinct_metrics) == 1 else "per-task primary"
    )

    # --- Fetch all instance scores in one query ---
    rows = session.execute(
        select(
            InstancePrediction.experiment_pk,
            InstancePrediction.native_id,
            InstancePrediction.task_hash,
            InstancePrediction.instance_metrics,
        ).where(
            InstancePrediction.experiment_pk.in_(pks),
            InstancePrediction.task_hash.in_(unique_task_hashes),
        )
    ).all()

    # --- Extract scores, group by experiment PK ---
    #
    # Instance-level metrics are stored as {scorer: {scorer: value}} (see
    # runners/io/builders.py), while task-level primary_metric uses the
    # "metric:scorer" format.  Try the task-level format first; if it misses,
    # fall back to the instance-level convention (scorer as both keys).
    from olmo_eval.runners.processing.utils import parse_metric_key

    scores_by_pk: dict[int, dict[tuple[str, str], float]] = {pk: {} for pk in pks}
    for exp_pk, native_id, th, instance_metrics in rows:
        if exp_pk not in scores_by_pk:
            continue
        row_metric = task_hash_to_metric.get(th)
        row_task_name = task_hash_to_name.get(th)
        if row_metric is None or row_task_name is None:
            continue
        instance_metric_key = row_metric
        parsed = parse_metric_key(row_metric)
        if parsed:
            scorer = parsed[1]
            instance_metric_key = f"{scorer}:{scorer}"
        score = extract_score_from_metrics(instance_metrics, instance_metric_key)
        if score is None:
            score = extract_score_from_metrics(instance_metrics, row_metric)
        if score is not None:
            scores_by_pk[exp_pk][(row_task_name, native_id)] = score

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
            sample_metrics = f", sample instance_metrics keys: {list(rows[0][3].keys())}"
        raise ValueError(
            f"Only {scored_count} of {len(ordered)} experiment(s) have extractable "
            f"instance scores for '{scope_label}' using metric='{display_metric}' "
            f"(fetched {instance_row_count} instance rows from DB{sample_metrics})"
        )

    # --- Rebuild index mapping for the active set ---
    models = [ModelMeta(label=label) for _, label in active]
    scores_by_idx: dict[int, dict[tuple[str, str], float]] = {}
    for idx, (pk, _) in enumerate(active):
        scores_by_idx[idx] = scores_by_pk[pk]

    # --- Intersect to shared instances ---
    id_sets = [set(scores.keys()) for scores in scores_by_idx.values()]
    shared_ids = id_sets[0]
    for s in id_sets[1:]:
        shared_ids = shared_ids & s

    # --- Compute pairs and return ---
    pairs = _compute_pairs(scores_by_idx, len(active), shared_ids, margin)

    contributing_task_names = tuple(sorted({tn for tn, _ in shared_ids}))
    result_task_name = suite_name or task_results[0].task_name

    return PairwiseResult(
        task_name=result_task_name,
        metric=display_metric,
        margin=margin,
        instance_count=len(shared_ids),
        models=models,
        pairs=pairs,
        suite_name=suite_name,
        task_names=contributing_task_names,
    )

"""Paloma macro bits per byte: the 16 Paloma sources averaged with equal weight."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.paloma import PALOMA_SUBSETS

make_suite(
    name="paloma",
    tasks=tuple(f"paloma_{subset}" for subset in PALOMA_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Paloma macro bits per byte: the unweighted mean over Paloma's 16 sources of each "
        "source's byte-weighted bits per byte."
    ),
)

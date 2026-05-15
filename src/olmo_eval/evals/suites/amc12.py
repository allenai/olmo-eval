from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.amc12 import AMC12_YEARS

make_suite(
    "amc12_full",
    tuple(f"amc12_{year}" for year in AMC12_YEARS),
    aggregation=AggregationStrategy.AVERAGE,
)

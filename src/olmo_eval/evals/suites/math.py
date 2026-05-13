from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS

make_suite(
    "minerva_math",
    tuple(f"minerva_math_{t}" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "math:posttrain:dev",
    (
        "aime_2025:pass_at_32:16k",
        "aime_2026:pass_at_32:16k",
        "hmmt_nov_2025:pass_at_32:16k",
        "hmmt_feb_2026:pass_at_32:16k",
    ),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Dev set for post-training math experiments: AIME 2025/2026 and HMMT Nov 2025 / Feb 2026."
    ),
)

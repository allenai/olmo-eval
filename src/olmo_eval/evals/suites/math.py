from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS

make_suite(
    "minerva_math",
    tuple(f"minerva_math_{t}" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "minerva_math_olmo3",
    tuple(f"minerva_math_{t}:olmo3" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Olmo 3 Base Eval for Minerva",
)

make_suite(
    "minerva_math_olmo3base",
    tuple(f"minerva_math_{t}:olmo3base_gen" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "minerva_math_bpb_olmo3base",
    tuple(f"minerva_math_{t}:bpb::olmo3base" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

"""
Safety suites for thinking and instruct models, and wildguard vs openai models
"""

from olmo_eval.common.metrics import AccuracyMetric, SafetyErrorMetric, SubsetAccuracyMetric
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

SAFETY_TASKS = [
    "do_anything_now",
    "harmbench",
    "trustllm_jailbreaktrigger",
    "wildguardtest",
    "wildjailbreak",
    "xstest",
]

make_suite(
    "safety_thinking",
    (*(f"{task}:wg_judge_thinking" for task in SAFETY_TASKS), "bbq:mcq", "wmdp:mcq"),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained reasoning models with a wildguard judge",
)

make_suite(
    "safety_instruct",
    (*(f"{task}:wg_judge" for task in SAFETY_TASKS), "bbq:mcq", "wmdp:mcq"),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with a wildguard judge",
)

make_suite(
    "safety_openai",
    (*(f"{task}:openai_judge" for task in SAFETY_TASKS), "bbq:mcq", "wmdp:mcq"),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with an openai judge",
)

make_suite(
    "safety_base",
    (*(f"{task}:base" for task in SAFETY_TASKS), "bbq:base", "wmdp:base"),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for pretrained models",
)


def safety_metrics(scorer, subsets):
    """Build the full metric tuple for a safety judge scorer."""
    return (
        AccuracyMetric(scorer=scorer),
        SafetyErrorMetric(scorer=scorer),
        *(SubsetAccuracyMetric(name=name, scorer=scorer) for name in subsets),
    )

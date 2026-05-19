"""
Safety suites for thinking and instruct models, and wildguard vs openai models
"""

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
    tuple(f"{task}:wg_judge_thinking" for task in SAFETY_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained reasoning models with a wildguard judge",
)

make_suite(
    "safety_instruct",
    tuple(f"{task}:wg_judge" for task in SAFETY_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with a wildguard judge",
)

make_suite(
    "safety_openai",
    tuple(f"{task}:openai_judge" for task in SAFETY_TASKS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Safety evals for posttrained instruct models with an openai judge",
)

"""Evaluation suites for adaptive-expert experiments."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

make_suite(
    "adaptive_experts:hybrid_pilot",
    (
        "math500:chat",
        "gpqa_diamond:qwen3_thinking",
        "ifeval_ood:qwen3_thinking",
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Chat-native reasoning and instruction-following pilot.",
)

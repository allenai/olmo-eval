"""Science evaluation suites."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# =============================================================================
# GPQA Suite
# =============================================================================

_GPQA_TASKS = (
    "gpqa_diamond",
    "gpqa_main",
    "gpqa_extended",
)

GPQA = make_suite(
    "gpqa",
    _GPQA_TASKS,
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA graduate-level science Q&A (diamond/main/extended)",
)

GPQA_MC = make_suite(
    "gpqa:mc",
    tuple(f"{t}:mc" for t in _GPQA_TASKS),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA with logprob-based MC scoring",
)

GPQA_BPB = make_suite(
    "gpqa:bpb",
    tuple(f"{t}:bpb" for t in _GPQA_TASKS),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA with bits-per-byte evaluation",
)

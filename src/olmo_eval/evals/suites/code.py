"""Code evaluation suites."""

from olmo_eval.evals.constants.code import MULTILINGUAL_MBPP_TASKS_V2
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register

# =============================================================================
# Multilingual Code Suites
# =============================================================================

MT_MBPP_V2FIX = make_suite(
    "mt_mbpp_v2fix",
    tuple(MULTILINGUAL_MBPP_TASKS_V2),
    description="Multilingual MBPP v2 with fixes",
)

MT_MBPP_V2FIX_BPB = make_suite(
    "mt_mbpp_v2fix:bpb",
    tuple(f"{t}:bpb" for t in MULTILINGUAL_MBPP_TASKS_V2),
    description="Multilingual MBPP v2 with BPB evaluation",
)

MT_MBPP_V2FIX_3SHOT = make_suite(
    "mt_mbpp_v2fix:3shot",
    tuple(f"{t}:3shot" for t in MULTILINGUAL_MBPP_TASKS_V2),
    description="Multilingual MBPP v2 with 3-shot prompting",
)

MT_MBPP_V2FIX_3SHOT_BPB = make_suite(
    "mt_mbpp_v2fix:3shot:bpb",
    tuple(f"{t}:3shot:bpb" for t in MULTILINGUAL_MBPP_TASKS_V2),
    description="Multilingual MBPP v2 with 3-shot BPB evaluation",
)


# =============================================================================
# OLMo3 Aggregate Code Suites (Average of Averages)
# =============================================================================

# Nested suite for mt_mbpp_v2fix with 3-shot BPB evaluation
_MT_MBPP_V2FIX_3SHOT_BPB_NESTED = Suite(
    name="mt_mbpp_v2fix:3shot:bpb",
    tasks=tuple(f"{t}:3shot:bpb" for t in MULTILINGUAL_MBPP_TASKS_V2),
    aggregation=AggregationStrategy.AVERAGE,
    description="Multilingual MBPP v2 with 3-shot BPB evaluation",
)

# OLMo3 base_easy code BPB suite (average of averages)
# Each child (task or nested suite) gets equal weight:
OLMO3_BASE_EASY_CODE_BPB = register(
    Suite(
        name="olmo3:base_easy:code:bpb",
        tasks=("humaneval:3shot:bpb", "mbpp:3shot:bpb", _MT_MBPP_V2FIX_3SHOT_BPB_NESTED),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="OLMo3 base_easy code BPB suite (average of averages)",
    )
)

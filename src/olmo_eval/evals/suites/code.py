"""Code evaluation suites."""

from olmo_eval.evals.constants.code import (
    ALL_CODEX_TASKS,
    CRUX_EVAL_TASKS,
    FIM_TASKS,
    MULTILINGUAL_MBPP_TASKS_V2,
    MULTIPL_E_HE_TASKS,
    MULTIPL_E_MBPP_TASKS,
    STARCODER_CODEX_TASKS,
    STARCODER_PASS_AT_1_TASKS,
)
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register

# =============================================================================
# Code Generation Suites
# =============================================================================

CODE = make_suite(
    "code",
    tuple(ALL_CODEX_TASKS),
    description="Code generation benchmarks",
)

STARCODER = make_suite(
    "starcoder",
    tuple(STARCODER_CODEX_TASKS),
    description="StarCoder code tasks",
)

STARCODER_PASS_AT_1 = make_suite(
    "starcoder::pass@1",
    tuple(STARCODER_PASS_AT_1_TASKS),
    description="StarCoder with pass@1 metric",
)


# =============================================================================
# Fill-in-the-Middle Suites
# =============================================================================

FIM_OLMO3 = make_suite(
    "fim::olmo3",
    tuple(f"{t}::olmo3" for t in FIM_TASKS),
    description="Fill-in-the-middle with OLMo3",
)


# =============================================================================
# Code Understanding Suites
# =============================================================================

CRUX_EVAL = make_suite(
    "crux-eval",
    tuple(CRUX_EVAL_TASKS),
    description="CRUXEval code understanding",
)


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

MULTIPL_E_HE_N32_V2 = make_suite(
    "multipl-e-humaneval:n32:v2",
    tuple(f"{t}:n32:v2" for t in MULTIPL_E_HE_TASKS),
    description="MultiPL-E HumanEval with n32 v2",
)

MULTIPL_E_MBPP_N32_V2 = make_suite(
    "multipl-e-mbpp:n32:v2",
    tuple(f"{t}:n32:v2" for t in MULTIPL_E_MBPP_TASKS),
    description="MultiPL-E MBPP with n32 v2",
)

MULTIPL_E_HE_N32_V2_PASS_AT_16 = make_suite(
    "multipl-e-humaneval:n32:v2:pass_at_16",
    tuple(f"{t}:n32:v2:pass_at_16" for t in MULTIPL_E_HE_TASKS),
    description="MultiPL-E HumanEval with pass@16",
)

MULTIPL_E_MBPP_N32_V2_PASS_AT_16 = make_suite(
    "multipl-e-mbpp:n32:v2:pass_at_16",
    tuple(f"{t}:n32:v2:pass_at_16" for t in MULTIPL_E_MBPP_TASKS),
    description="MultiPL-E MBPP with pass@16",
)


# =============================================================================
# OLMo3 Aggregate Code Suites (Average of Averages)
# =============================================================================

# Nested suite for mt_mbpp_v2fix with BPB evaluation
_MT_MBPP_V2FIX_BPB_NESTED = Suite(
    name="mt_mbpp_v2fix:bpb",
    tasks=tuple(f"{t}:bpb" for t in MULTILINGUAL_MBPP_TASKS_V2),
    aggregation=AggregationStrategy.AVERAGE,
    description="Multilingual MBPP v2 with BPB evaluation",
)

# OLMo3 base_easy code BPB suite (average of averages)
# Each child (task or nested suite) gets equal weight:
OLMO3_BASE_EASY_CODE_BPB = register(
    Suite(
        name="olmo3:base_easy:code:bpb",
        tasks=(
            "codex_humaneval:3shot:bpb",
            "mbpp:3shot:bpb",
            # _MT_MBPP_V2FIX_BPB_NESTED, HF IS HAVING ISSUES
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="OLMo3 base_easy code BPB suite (average of averages)",
    )
)

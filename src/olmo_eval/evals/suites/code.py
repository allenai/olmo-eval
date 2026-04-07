"""Code evaluation suites."""

from olmo_eval.evals.constants.code import (
    MULTILINGUAL_MBPP_TASKS_V2,
    MULTIPL_E_HUMANEVAL_TASKS,
    MULTIPL_E_MBPP_TASKS,
)
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register

# =============================================================================
# Multilingual Code Suites
# =============================================================================

# Define variant configurations: (suffix, description_suffix)
_MT_MBPP_VARIANTS: tuple[tuple[str, str], ...] = (
    ("", ""),
    (":bpb", " with BPB evaluation"),
    (":3shot", " with 3-shot prompting"),
    (":3shot:bpb", " with 3-shot BPB evaluation"),
)

# Generate all suites programmatically
for _suffix, _desc_suffix in _MT_MBPP_VARIANTS:
    make_suite(
        f"mt_mbpp_v2fix{_suffix}",
        tuple(f"{t}{_suffix}" for t in MULTILINGUAL_MBPP_TASKS_V2),
        description=f"Multilingual MBPP v2 with fixes{_desc_suffix}",
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


# =============================================================================
# Serialized code BPB suite (pre-formatted data from oe-eval-internal)
# =============================================================================

_SERIALIZED_MT_MBPP_V2FIX_BPB = Suite(
    name="_serialized_mt_mbpp_v2fix_bpb",
    tasks=tuple(f"serialized:{t}" for t in MULTILINGUAL_MBPP_TASKS_V2),
    aggregation=AggregationStrategy.AVERAGE,
    description="Serialized multilingual MBPP v2 BPB evaluation",
)

OLMO3_BASE_EASY_CODE_BPB_SERIALIZED = register(
    Suite(
        name="olmo3:base_easy:code:bpb:serialized",
        tasks=(
            "serialized:codex_humaneval_3shot_bpb",
            "serialized:mbpp_3shot_bpb",
            _SERIALIZED_MT_MBPP_V2FIX_BPB,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="OLMo3 base_easy code BPB suite from serialized data",
    )
)
# =============================================================================
# MULTIPL_E Suites
# =============================================================================

_MULTIPL_E_VARIANTS: tuple[tuple[str, str], ...] = (
    ("", ""),
    (":pass_at_1", " with pass@1 execution evaluation"),
    (":pass_at_10", " with pass@10 execution evaluation"),
)

for _suffix, _desc_suffix in _MULTIPL_E_VARIANTS:
    make_suite(
        f"multipl_e_humaneval{_suffix}",
        tuple(f"{t}{_suffix}" for t in MULTIPL_E_HUMANEVAL_TASKS),
        description=f"MULTIPL_E HumanEval (6 languages){_desc_suffix}",
    )
    make_suite(
        f"multipl_e_mbpp{_suffix}",
        tuple(f"{t}{_suffix}" for t in MULTIPL_E_MBPP_TASKS),
        description=f"MULTIPL_E MBPP (6 languages){_desc_suffix}",
    )
    # Combined suite with both HumanEval and MBPP
    make_suite(
        f"multipl_e{_suffix}",
        tuple(f"{t}{_suffix}" for t in MULTIPL_E_HUMANEVAL_TASKS + MULTIPL_E_MBPP_TASKS),
        description=f"MULTIPL_E HumanEval + MBPP (6 languages each){_desc_suffix}",
    )

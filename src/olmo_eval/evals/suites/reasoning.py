"""Reasoning benchmark suites (AGI-Eval, BBH)."""

from olmo_eval.evals.constants.benchmarks import (
    AGI_EVAL_ENGLISH_TASKS,
    BBH_TASKS,
)
from olmo_eval.evals.suites.registry import format_tasks, make_suite

# =============================================================================
# AGI-Eval Suites
# =============================================================================

AGI_EVAL = make_suite(
    "agi_eval",
    format_tasks(AGI_EVAL_ENGLISH_TASKS, "{}:1shot::olmes"),
    description="AGI-Eval English standardized tests",
)

AGI_EVAL_THINKER = make_suite(
    "agi_eval_english:0shot_cot::olmo3:thinker",
    format_tasks(AGI_EVAL_ENGLISH_TASKS, "agi_eval_{}:0shot_cot::olmo3:thinker"),
    description="AGI-Eval with OLMo3 thinker",
)

AGI_EVAL_MIDTRAIN = make_suite(
    "agi_eval_english::olmo3:midtrain",
    format_tasks(AGI_EVAL_ENGLISH_TASKS, "agi_eval_{}::olmo3:midtrain"),
    description="AGI-Eval with OLMo3 midtrain",
)


# =============================================================================
# BIG-Bench Hard Suites
# =============================================================================

BBH_COT_THINKER = make_suite(
    "bbh:cot::olmo3:thinker",
    format_tasks(BBH_TASKS, "bbh_{}:cot::olmo3:thinker"),
    description="BIG-Bench Hard with CoT (thinker)",
)

BBH_COT_MIDTRAIN = make_suite(
    "bbh:cot::olmo3:midtrain",
    format_tasks(BBH_TASKS, "bbh_{}:cot::olmo3:midtrain"),
    description="BIG-Bench Hard with CoT (midtrain)",
)

BBH_COT_HELDOUT = make_suite(
    "bbh:cot::olmo3:heldout",
    format_tasks(BBH_TASKS, "bbh_{}:cot::olmo3:heldout"),
    description="BIG-Bench Hard with CoT (heldout)",
)

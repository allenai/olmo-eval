"""Multi-turn, styled, and OMEGA benchmark suites."""

from olmo_eval.evals.constants.benchmarks import (
    IFEVAL_MT_TASKS,
    MULTITURN_ALPACAEVAL_TASKS,
    OMEGA_SUB_CATEGORIES,
    STYLED_TASKS,
    STYLED_TASKS_POPQA,
)
from olmo_eval.evals.suites.registry import format_tasks, make_suite

# =============================================================================
# Multi-Turn Suites
# =============================================================================

IFEVAL_MT_THINKER = make_suite(
    "ifeval_mt::tulu-thinker",
    format_tasks(IFEVAL_MT_TASKS, "ifeval_mt_{}::tulu-thinker"),
    description="IFEval multi-turn with tulu-thinker",
)

MULTITURN_ALPACAEVAL = make_suite(
    "multiturn_alpacaeval::tulu",
    format_tasks(MULTITURN_ALPACAEVAL_TASKS, "multiturn_alpacaeval_{}::tulu"),
    description="AlpacaEval multi-turn variants",
)


# =============================================================================
# Styled Suites
# =============================================================================

STYLED_POPQA_THINKER = make_suite(
    "styled_popqa::tulu-thinker",
    format_tasks(STYLED_TASKS_POPQA, "styled_popqa_{}::tulu-thinker"),
    description="Styled PopQA with tulu-thinker",
)

STYLED_MATH500_THINKER = make_suite(
    "styled_math500::tulu-thinker",
    format_tasks(STYLED_TASKS, "styled_math500_{}::tulu-thinker"),
    description="Styled Math500 with tulu-thinker",
)

# Styled AlpacaEval has nested structure
_styled_alpacaeval_tasks = tuple(
    f"styled_alpacaeval_{task_type}_{ref}_ref::tulu-thinker"
    for task_type in STYLED_TASKS
    for ref in ("og", "new")
)
STYLED_ALPACAEVAL_THINKER = make_suite(
    "styled_alpacaeval::tulu-thinker",
    _styled_alpacaeval_tasks,
    description="Styled AlpacaEval with tulu-thinker",
)


# =============================================================================
# OMEGA Benchmark Suites
# =============================================================================


def _build_omega_tasks(regime_suffix: str) -> tuple[str, ...]:
    """Build OMEGA task names for a given regime."""
    tasks: list[str] = []
    for category, subcats in OMEGA_SUB_CATEGORIES.items():
        splits = ("test_in", "test_out") if category == "explorative" else ("test",)
        for subcat in subcats:
            for split in splits:
                tasks.append(f"omega_{category}_{subcat}_{split}{regime_suffix}")
    return tuple(tasks)


OMEGA_0SHOT_CHAT = make_suite(
    "omega:0-shot-chat",
    _build_omega_tasks(":0-shot-chat"),
    description="OMEGA benchmark with 0-shot chat",
)

OMEGA_MIDTRAIN = make_suite(
    "omega::olmo3:midtrain",
    _build_omega_tasks("::olmo3:midtrain"),
    description="OMEGA with OLMo3 midtrain regime",
)

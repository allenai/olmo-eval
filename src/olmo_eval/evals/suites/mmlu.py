"""MMLU benchmark suites."""

from olmo_eval.evals.constants.benchmarks import (
    MMLU_CATEGORIES,
    MMLU_PRO_CATEGORIES,
    MMLU_SUBCATEGORIES,
)
from olmo_eval.evals.suites.registry import format_tasks, make_suite

# =============================================================================
# MMLU Suites
# =============================================================================

MMLU_RC = make_suite(
    "mmlu:rc",
    format_tasks(MMLU_CATEGORIES, "{}:rc::olmes"),
    description="MMLU with ranked classification scoring",
)

MMLU_BPB = make_suite(
    "mmlu:bpb",
    format_tasks(MMLU_CATEGORIES, "{}:bpb::olmes"),
    description="MMLU with bits-per-byte scoring",
)

MMLU_MC = make_suite(
    "mmlu:mc",
    format_tasks(MMLU_CATEGORIES, "{}:mc::olmes"),
    description="MMLU with multiple choice scoring",
)

MMLU_STEM_MC = make_suite(
    "mmlu_stem:mc",
    format_tasks(tuple(f"mmlu_{c}" for c in MMLU_SUBCATEGORIES["stem"]), "{}:mc::olmes"),
    description="MMLU STEM subjects with MC scoring",
)

MMLU_HUMANITIES_MC = make_suite(
    "mmlu_humanities:mc",
    format_tasks(tuple(f"mmlu_{c}" for c in MMLU_SUBCATEGORIES["humanities"]), "{}:mc::olmes"),
    description="MMLU humanities subjects with MC scoring",
)

MMLU_SOCIAL_SCIENCES_MC = make_suite(
    "mmlu_social_sciences:mc",
    format_tasks(tuple(f"mmlu_{c}" for c in MMLU_SUBCATEGORIES["social_sciences"]), "{}:mc::olmes"),
    description="MMLU social sciences subjects with MC scoring",
)

MMLU_OTHER_MC = make_suite(
    "mmlu_other:mc",
    format_tasks(tuple(f"mmlu_{c}" for c in MMLU_SUBCATEGORIES["other"]), "{}:mc::olmes"),
    description="MMLU other subjects with MC scoring",
)

MMLU_COT_THINKER = make_suite(
    "mmlu:cot::olmo3:thinker",
    format_tasks(MMLU_CATEGORIES, "{}:cot::olmo3:thinker"),
    description="MMLU with chain-of-thought (OLMo3 thinker regime)",
)

MMLU_COT_MIDTRAIN = make_suite(
    "mmlu:cot::olmo3:midtrain",
    format_tasks(MMLU_CATEGORIES, "{}:cot::olmo3:midtrain"),
    description="MMLU with chain-of-thought (OLMo3 midtrain regime)",
)

MMLU_PRO_MC = make_suite(
    "mmlu_pro:mc",
    format_tasks(MMLU_PRO_CATEGORIES, "{}:mc::none"),
    description="MMLU-Pro enhanced benchmark with MC scoring",
)

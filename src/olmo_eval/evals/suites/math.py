"""Math benchmark suites (Minerva, GSM, DeepMind Math)."""

from olmo_eval.evals.constants.benchmarks import (
    ALL_GSM_SYMB_TASKS,
    ALL_MINERVA_TASKS,
    DEEPMIND_MATH_CATEGORIES,
)
from olmo_eval.evals.suites.registry import AggregationStrategy, format_tasks, make_suite

# =============================================================================
# Minerva Math Suites
# =============================================================================

MINERVA = make_suite(
    "minerva",
    format_tasks(ALL_MINERVA_TASKS, "{}::olmes"),
    description="Minerva math benchmark",
)

MINERVA_BPB = make_suite(
    "minerva:bpb",
    format_tasks(ALL_MINERVA_TASKS, "{}:bpb::olmes"),
    description="Minerva with BPB scoring",
)

MINERVA_N4_V2 = make_suite(
    "minerva:n4:v2",
    format_tasks(ALL_MINERVA_TASKS, "{}::olmes:n4:v2"),
    description="Minerva with n4 v2 config",
)

MINERVA_MIDTRAIN = make_suite(
    "minerva_math::olmo3:midtrain",
    format_tasks(ALL_MINERVA_TASKS, "{}::olmo3:midtrain"),
    description="Minerva with OLMo3 midtrain regime",
)


# =============================================================================
# DeepMind Math Suites
# =============================================================================

DEEPMIND_MATH_HELDOUT = make_suite(
    "deepmind_math::olmo3:heldout",
    format_tasks(DEEPMIND_MATH_CATEGORIES, "deepmind_math_{}::olmo3:heldout"),
    description="DeepMind Mathematics heldout set",
)


# =============================================================================
# GSM-Symbolic Suites
# =============================================================================

GSM_SYMB = make_suite(
    "gsm-symb",
    tuple(ALL_GSM_SYMB_TASKS),
    description="GSM-Symbolic reasoning tasks",
)

GSM_SYMB_N8_V2 = make_suite(
    "gsm-symb:n8:v2",
    tuple(f"{t}:n8:v2" for t in ALL_GSM_SYMB_TASKS),
    description="GSM-Symbolic with n8 v2 config",
)

GSM_SYMB_N8_V2_PASS_AT_4 = make_suite(
    "gsm-symb:n8:v2:pass_at_4",
    tuple(f"{t}:n8:v2:pass_at_4" for t in ALL_GSM_SYMB_TASKS),
    description="GSM-Symbolic with pass@4 metric",
)


# =============================================================================
# Combined Math Suite
# =============================================================================

MATH = make_suite(
    "math",
    ("gsm8k::olmo1", "gsm8k::olmes", MINERVA),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Combined math benchmark",
)

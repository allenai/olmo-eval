"""Core evaluation suites (Core, ARC, Basic Skills, Generation)."""

from olmo_eval.evals.constants.benchmarks import (
    ALL_CORE_TASKS,
    ALL_GEN_TASKS,
    ALL_GEN_XLARGE_TASKS,
    ARC_TASKS,
    BASIC_SKILLS_TASKS,
)
from olmo_eval.evals.suites.registry import format_tasks, make_suite

# =============================================================================
# Core Evaluation Suites
# =============================================================================

CORE_RC = make_suite(
    "core:rc",
    format_tasks(ALL_CORE_TASKS, "{}:rc::olmes"),
    description="Core OLMES tasks with ranked classification",
)

CORE_MC = make_suite(
    "core:mc",
    format_tasks(ALL_CORE_TASKS, "{}:mc::olmes"),
    description="Core OLMES tasks with multiple choice",
)

CORE_MC_FULL = make_suite(
    "core:mc::full",
    format_tasks(ALL_CORE_TASKS, "{}:mc::olmes::full"),
    description="Core OLMES tasks with MC (full split)",
)

CORE_RC_FULL = make_suite(
    "core:rc::full",
    format_tasks(ALL_CORE_TASKS, "{}:rc::olmes::full"),
    description="Core OLMES tasks with RC (full split)",
)


# =============================================================================
# ARC Suites
# =============================================================================

ARC_RC = make_suite(
    "arc:rc",
    format_tasks(ARC_TASKS, "{}:rc::olmes"),
    description="ARC with ranked classification",
)

ARC_MC = make_suite(
    "arc:mc",
    format_tasks(ARC_TASKS, "{}:mc::olmes"),
    description="ARC with multiple choice",
)

ARC_BPB_FULL = make_suite(
    "arc:bpb::full",
    format_tasks(ARC_TASKS, "{}:bpb::olmes:full"),
    description="ARC with BPB (full split)",
)

ARC_RC_FULL = make_suite(
    "arc:rc::full",
    format_tasks(ARC_TASKS, "{}:rc::olmes:full"),
    description="ARC with RC (full split)",
)

ARC_MC_FULL = make_suite(
    "arc:mc::full",
    format_tasks(ARC_TASKS, "{}:mc::olmes:full"),
    description="ARC with MC (full split)",
)

ARC_RC_XLARGE = make_suite(
    "arc:rc::xlarge",
    format_tasks(ARC_TASKS, "{}:rc::xlarge"),
    description="ARC with RC (xlarge config)",
)

ARC_MC_XLARGE = make_suite(
    "arc:mc::xlarge",
    format_tasks(ARC_TASKS, "{}:mc::xlarge"),
    description="ARC with MC (xlarge config)",
)


# =============================================================================
# Basic Skills Suites
# =============================================================================

BASIC_BPB = make_suite(
    "basic:bpb",
    format_tasks(BASIC_SKILLS_TASKS, "{}:bpb::olmes"),
    description="Basic skills with BPB scoring",
)

BASIC_RC = make_suite(
    "basic:rc",
    format_tasks(BASIC_SKILLS_TASKS, "{}:rc::olmes"),
    description="Basic skills with RC scoring",
)

BASIC_MC = make_suite(
    "basic:mc",
    format_tasks(BASIC_SKILLS_TASKS, "{}:mc::olmes"),
    description="Basic skills with MC scoring",
)


# =============================================================================
# Generation Suites
# =============================================================================

GEN = make_suite(
    "gen",
    tuple(ALL_GEN_TASKS),
    description="Standard generation tasks",
)

GEN_XLARGE = make_suite(
    "gen::xlarge",
    tuple(ALL_GEN_XLARGE_TASKS),
    description="Generation tasks with xlarge config",
)

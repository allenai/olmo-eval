"""OLMo Base evaluation suites.

Aggregate suites for the OLMo Base evaluation regime, organized by task type
(MCQA, generation, math) and scoring method (RC, BPB).
"""

import olmo_eval.evals.suites.mmlu  # noqa: F401 – register MMLU suites
from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    Suite,
    get_suite,
    make_suite,
    register,
)
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS

# =============================================================================
# Helper sub-suites
# =============================================================================

_GSM_SYMB = make_suite(
    "gsm_symb:olmo3base",
    (
        "gsm_symbolic::olmo3base",
        "gsm_symbolic:p1::olmo3base",
        "gsm_symbolic:p2::olmo3base",
    ),
)

_MINERVA_MATH = make_suite(
    "minerva_math::olmo3base",
    tuple(f"minerva_math_{t}::olmo3base" for t in MATH_SUBSETS),
)

_ARC_RC = make_suite(
    "arc:rc::olmo3base",
    ("arc_challenge:rc::olmo3base", "arc_easy:rc::olmo3base"),
)

_ARC_MC = make_suite(
    "arc:mc::olmo3base",
    ("arc_challenge:mc::olmo3base", "arc_easy:mc::olmo3base"),
)

_ARC_BPB = make_suite(
    "arc:bpb::olmo3base",
    ("arc_challenge:bpb::olmo3base", "arc_easy:bpb::olmo3base"),
)

# =============================================================================
# MCQA suites
# =============================================================================

register(
    Suite(
        name="olmobase:mcqa_stem",
        tasks=(
            get_suite("arc:mc::olmo3base"),
            get_suite("mmlu:stem:mc::olmo3base"),
            "medmcqa:mc::olmo3base",
            "medqa_en:mc::olmo3base",
            "sciq:mc::olmo3base",
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

register(
    Suite(
        name="olmobase:mcqa_non_stem",
        tasks=(
            get_suite("mmlu:humanities:mc::olmo3base"),
            get_suite("mmlu:other:mc::olmo3base"),
            get_suite("mmlu:social_sciences:mc::olmo3base"),
            "csqa:mc::olmo3base",
            "piqa:mc::olmo3base",
            "socialiqa:mc::olmo3base",
            "coqa:mc::olmo3base",
            "drop:mc::olmo3base",
            "jeopardy:mc::olmo3base",
            "naturalqs:mc::olmo3base",
            "squad:mc::olmo3base",
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Generation suite
# =============================================================================

register(
    Suite(
        name="olmobase:gen",
        tasks=(
            "hellaswag:rc::olmo3base",
            "lambada",
            "winogrande:rc::olmo3base",
            get_suite("basic_skills:rc:olmo3base"),
            "drop:gen::olmo3base",
            "jeopardy:gen::olmo3base",
            "squad:gen::olmo3base",
            "coqa:gen::olmo3base",
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Math suite
# =============================================================================

register(
    Suite(
        name="olmobase:math",
        tasks=(
            "gsm8k::olmo3base",
            _GSM_SYMB,
            _MINERVA_MATH,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Easy QA RC suite
# =============================================================================

register(
    Suite(
        name="olmobase:easy:qa:rc",
        tasks=(
            _ARC_RC,
            get_suite("mmlu:rc::olmo3base"),
            "csqa:rc::olmo3base",
            "hellaswag:rc::olmo3base",
            "winogrande:rc::olmo3base",
            "socialiqa:rc::olmo3base",
            "piqa:rc::olmo3base",
            "coqa:rc::olmo3base",
            "drop:rc::olmo3base",
            "jeopardy:rc::olmo3base",
            "naturalqs:rc::olmo3base",
            "squad:rc::olmo3base",
            "sciq:rc::olmo3base",
            "qasper_yesno:rc::olmo3base",
            get_suite("basic_skills:rc:olmo3base"),
            "lab_bench_dbqa::olmo3base",
            "lab_bench_protocolqa::olmo3base",
            "lambada",
            "medmcqa:rc::olmo3base",
            "medqa_en:rc::olmo3base",
            "sciriff_yesno:rc::olmo3base",
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Easy QA BPB suite
# =============================================================================

register(
    Suite(
        name="olmobase:easy:qa:bpb",
        tasks=(
            _ARC_BPB,
            get_suite("mmlu:bpb"),
            "csqa:bpb::olmo3base",
            "hellaswag:bpb::olmo3base",
            "winogrande:bpb::olmo3base",
            "socialiqa:bpb::olmo3base",
            "piqa:bpb::olmo3base",
            "coqa:bpb::olmo3base",
            "drop:bpb::olmo3base",
            "jeopardy:bpb::olmo3base",
            "naturalqs:bpb::olmo3base",
            "squad:bpb::olmo3base",
            "sciq:bpb::olmo3base",
            "qasper_yesno:bpb::olmo3base",
            get_suite("basic_skills:bpb:olmo3base"),
            "lab_bench_dbqa:bpb::olmo3base",
            "lab_bench_protocolqa:bpb::olmo3base",
            "lambada:bpb::olmo3base",
            "medmcqa:bpb::olmo3base",
            "medqa_en:bpb::olmo3base",
            "sciriff_yesno:bpb::olmo3base",
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Easy Math BPB suite
# =============================================================================

register(
    Suite(
        name="olmobase:easy:math:bpb",
        tasks=(get_suite("minerva_math:bpb:olmo3base"),),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

# =============================================================================
# Easy Code BPB suite
# =============================================================================

register(
    Suite(
        name="olmobase:easy:code:bpb",
        tasks=(
            "codex_humaneval:bpb::olmo3base",
            "mbpp:bpb::olmo3base",
            get_suite("mt_mbpp:bpb:olmo3base"),
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

"""OLMo Base evaluation suites.

Aggregate suites for the OLMo Base evaluation regime, organized by task type
(MCQA, generation, math) and scoring method (RC, BPB).
"""

import olmo_eval.evals.suites.mmlu  # noqa: F401 – register MMLU suites
from olmo_eval.evals.suites.biology import _LAB_BENCH_TASKS
from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    Suite,
    get_suite,
    make_suite,
    register,
)
from olmo_eval.evals.tasks.basic_skills import BASIC_SKILLS_SUBTASKS
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS
from olmo_eval.evals.tasks.mmlu import _HUMANITIES, _OTHER, _SOCIAL_SCIENCES, _STEM
from olmo_eval.evals.tasks.multilingual_mbpp import MULTILINGUAL_MBPP_LANGUAGES

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


def _mmlu_mc_tasks(subjects: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}:mc:olmo3base" for s in subjects)


def _mmlu_rc_tasks(subjects: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}:rc:olmo3base" for s in subjects)


# MMLU mc::olmo3base suites
make_suite("mmlu:stem:mc::olmo3base", _mmlu_mc_tasks(_STEM))
make_suite("mmlu:humanities:mc::olmo3base", _mmlu_mc_tasks(_HUMANITIES))
make_suite("mmlu:social_sciences:mc::olmo3base", _mmlu_mc_tasks(_SOCIAL_SCIENCES))
make_suite("mmlu:other:mc::olmo3base", _mmlu_mc_tasks(_OTHER))

# MMLU rc::olmo3base suites
_MMLU_RC_STEM = make_suite("mmlu:stem:rc::olmo3base", _mmlu_rc_tasks(_STEM))
_MMLU_RC_HUMANITIES = make_suite("mmlu:humanities:rc::olmo3base", _mmlu_rc_tasks(_HUMANITIES))
_MMLU_RC_SOCIAL_SCIENCES = make_suite(
    "mmlu:social_sciences:rc::olmo3base", _mmlu_rc_tasks(_SOCIAL_SCIENCES)
)
_MMLU_RC_OTHER = make_suite("mmlu:other:rc::olmo3base", _mmlu_rc_tasks(_OTHER))

register(
    Suite(
        name="mmlu:rc::olmo3base",
        tasks=(
            _MMLU_RC_STEM,
            _MMLU_RC_HUMANITIES,
            _MMLU_RC_SOCIAL_SCIENCES,
            _MMLU_RC_OTHER,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

make_suite(
    "minerva_math_olmo3",
    tuple(f"minerva_math_{t}:olmo3" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
    description="Olmo 3 Base Eval for Minerva",
)

make_suite(
    "minerva_math_olmo3base",
    tuple(f"minerva_math_{t}:olmo3base_gen" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "minerva_math_bpb_olmo3base",
    tuple(f"minerva_math_{t}:bpb::olmo3base" for t in MATH_SUBSETS),
    aggregation=AggregationStrategy.AVERAGE,
)

make_suite(
    "lab_bench:olmo3base",
    tuple(f"{t}:olmo3base" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with RC cloze format (3-shot, logprob MC)",
)

make_suite("arc:mc:olmo3base", ("arc_easy:mc:olmo3base", "arc_challenge:mc:olmo3base"))

make_suite("medmcqa:rc_mc:olmo3base", ("medmcqa:rc:olmo3base", "medmcqa:mc:olmo3base"))

make_suite("medqa_en:rc_mc:olmo3base", ("medqa_en:rc:olmo3base", "medqa_en:mc:olmo3base"))

make_suite("piqa:rc_mc:olmo3base", ("piqa:rc:olmo3base", "piqa:mc:olmo3base"))

make_suite("csqa:rc_mc:olmo3base", ("csqa:rc:olmo3base", "csqa:mc:olmo3base"))

make_suite("socialiqa:rc_mc:olmo3base", ("socialiqa:rc:olmo3base", "socialiqa:mc:olmo3base"))

make_suite("coqa:gen_only:olmo3base", ("coqa:gen:olmo3base",))

make_suite("hellaswag:rc_mc:olmo3base", ("hellaswag:rc:olmo3base", "hellaswag:mc:olmo3base"))

make_suite("jeopardy:gen_only:olmo3base", ("jeopardy:gen:olmo3base",))

make_suite("qasper_yesno:rc_only:olmo3base", ("qasper_yesno:rc:olmo3base",))

make_suite("sciq:rc_mc:olmo3base", ("sciq:rc:olmo3base", "sciq:mc:olmo3base"))

make_suite("sciriff_yesno:rc_only:olmo3base", ("sciriff_yesno:rc:olmo3base",))

make_suite("squad:rc_mc:olmo3base", ("squad:mc:olmo3base", "squad:rc:olmo3base"))

make_suite("winogrande:rc_mc:olmo3base", ("winogrande:rc:olmo3base", "winogrande:mc:olmo3base"))

make_suite("naturalqs:olmo3base", ("naturalqs:mc:olmo3base", "naturalqs:rc:olmo3base"))

make_suite(
    "basic_skills:rc:olmo3base",
    tuple(f"basic_skills_{s}:rc::olmo3base" for s in BASIC_SKILLS_SUBTASKS),
)

make_suite(
    "basic_skills:bpb:olmo3base",
    tuple(f"basic_skills_{s}:bpb::olmo3base" for s in BASIC_SKILLS_SUBTASKS),
)

make_suite(
    "minerva_math:bpb:olmo3base",
    tuple(f"minerva_math_{t}:bpb::olmo3base" for t in MATH_SUBSETS),
)

make_suite(
    "mt_mbpp:bpb:olmo3base",
    tuple(f"mt_mbpp_{lang}:bpb::olmo3base" for lang in MULTILINGUAL_MBPP_LANGUAGES),
)

make_suite(
    "mt_mbpp_v2fix:bpb:olmo3base",
    tuple(f"mt_mbpp_v2fix_{lang}:bpb::olmo3base" for lang in MULTILINGUAL_MBPP_LANGUAGES),
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

# =============================================================================
# OLMeS helper sub-suites
# =============================================================================

make_suite(
    "arc:rc:olmes:full",
    ("arc_easy:rc:olmes:full", "arc_challenge:rc:olmes:full"),
)

make_suite(
    "arc:bpb::olmes:full",
    ("arc_easy:bpb::olmes:full", "arc_challenge:bpb::olmes:full"),
)

make_suite(
    "basic_skills:rc::olmes",
    tuple(f"basic_skills_{s}:rc::olmes" for s in BASIC_SKILLS_SUBTASKS),
)

make_suite(
    "basic_skills:bpb::olmes",
    tuple(f"basic_skills_{s}:bpb::olmes" for s in BASIC_SKILLS_SUBTASKS),
)

make_suite(
    "minerva_math:bpb::olmes",
    tuple(f"minerva_math_{s}:bpb::olmes" for s in MATH_SUBSETS),
)

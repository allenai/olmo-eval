import olmo_eval.evals.suites.mmlu  # noqa: F401 – register MMLU suites
from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    get_suite,
    make_suite,
)
from olmo_eval.evals.tasks.basic_skills import BASIC_SKILLS_SUBTASKS
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS
from olmo_eval.evals.tasks.mmlu import _HUMANITIES, _OTHER, _SOCIAL_SCIENCES, _STEM
from olmo_eval.evals.tasks.multilingual_mbpp import MULTILINGUAL_MBPP_LANGUAGES

# -- ARC ----------------------------------------------------------------------

make_suite(
    name="arc:rc:olmo3base",
    tasks=("arc_challenge:rc:olmo3base", "arc_easy:rc:olmo3base"),
)
make_suite(
    name="arc:bpb:olmo3base",
    tasks=("arc_challenge:bpb:olmo3base", "arc_easy:bpb:olmo3base"),
)
make_suite(
    name="arc:mc:olmo3base",
    tasks=("arc_challenge:mc:olmo3base", "arc_easy:mc:olmo3base"),
)

# -- MMLU ---------------------------------------------------------------------

make_suite(
    name="mmlu:stem:mc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:mc:olmo3base" for subtask in _STEM),
)
make_suite(
    name="mmlu:humanities:mc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:mc:olmo3base" for subtask in _HUMANITIES),
)
make_suite(
    name="mmlu:social_sciences:mc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:mc:olmo3base" for subtask in _SOCIAL_SCIENCES),
)
make_suite(
    name="mmlu:other:mc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:mc:olmo3base" for subtask in _OTHER),
)
make_suite(
    name="mmlu:mc:olmo3base",
    tasks=(
        get_suite("mmlu:stem:mc:olmo3base"),
        get_suite("mmlu:humanities:mc:olmo3base"),
        get_suite("mmlu:social_sciences:mc:olmo3base"),
        get_suite("mmlu:other:mc:olmo3base"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="mmlu:stem:rc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:rc:olmo3base" for subtask in _STEM),
)
make_suite(
    name="mmlu:humanities:rc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:rc:olmo3base" for subtask in _HUMANITIES),
)
make_suite(
    name="mmlu:social_sciences:rc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:rc:olmo3base" for subtask in _SOCIAL_SCIENCES),
)
make_suite(
    name="mmlu:other:rc:olmo3base",
    tasks=tuple(f"mmlu_{subtask}:rc:olmo3base" for subtask in _OTHER),
)
make_suite(
    name="mmlu:rc:olmo3base",
    tasks=(
        get_suite("mmlu:stem:rc:olmo3base"),
        get_suite("mmlu:humanities:rc:olmo3base"),
        get_suite("mmlu:social_sciences:rc:olmo3base"),
        get_suite("mmlu:other:rc:olmo3base"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

# -- Math ---------------------------------------------------------------------

make_suite(
    name="gsm_symb:olmo3base",
    tasks=("gsm_symbolic:olmo3base", "gsm_symbolic:p1:olmo3base", "gsm_symbolic:p2:olmo3base"),
)
make_suite(
    name="minerva_math:olmo3base",
    tasks=tuple(f"minerva_math_{subtask}:olmo3base" for subtask in MATH_SUBSETS),
)
make_suite(
    name="minerva_math:bpb:olmo3base",
    tasks=tuple(f"minerva_math_{subtask}:bpb:olmo3base" for subtask in MATH_SUBSETS),
)

# -- Basic skills -------------------------------------------------------------

make_suite(
    name="basic_skills:rc:olmo3base",
    tasks=tuple(f"basic_skills_{subtask}:rc:olmo3base" for subtask in BASIC_SKILLS_SUBTASKS),
)
make_suite(
    name="basic_skills:bpb:olmo3base",
    tasks=tuple(f"basic_skills_{subtask}:bpb:olmo3base" for subtask in BASIC_SKILLS_SUBTASKS),
)

# -- Code ---------------------------------------------------------------------

make_suite(
    name="mt_mbpp:bpb:olmo3base",
    tasks=tuple(f"mt_mbpp_{lang}:bpb:olmo3base" for lang in MULTILINGUAL_MBPP_LANGUAGES),
)
make_suite(
    name="mt_mbpp_v2fix:bpb:olmo3base",
    tasks=tuple(f"mt_mbpp_v2fix_{lang}:bpb:olmo3base" for lang in MULTILINGUAL_MBPP_LANGUAGES),
)

# =============================================================================
# OlmoBaseEval
# =============================================================================

make_suite(
    name="olmobase:easy:qa:rc",
    tasks=(
        get_suite("arc:rc:olmo3base"),
        get_suite("mmlu:rc:olmo3base"),
        "csqa:rc:olmo3base",
        "hellaswag:rc:olmo3base",
        "winogrande:rc:olmo3base",
        "socialiqa:rc:olmo3base",
        "piqa:rc:olmo3base",
        "coqa:rc:olmo3base",
        "drop:rc:olmo3base",
        "jeopardy:rc:olmo3base",
        "naturalqs:rc:olmo3base",
        "squad:rc:olmo3base",
        "sciq:rc:olmo3base",
        "qasper_yesno:rc:olmo3base",
        get_suite("basic_skills:rc:olmo3base"),
        "lab_bench_dbqa:olmo3base",
        "lab_bench_protocolqa:olmo3base",
        "lambada",
        "medmcqa:rc:olmo3base",
        "medqa_en:rc:olmo3base",
        "sciriff_yesno:rc:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:easy:qa:bpb",
    tasks=(
        get_suite("arc:bpb:olmo3base"),
        get_suite("mmlu:bpb"),
        "csqa:bpb:olmo3base",
        "hellaswag:bpb:olmo3base",
        "winogrande:bpb:olmo3base",
        "socialiqa:bpb:olmo3base",
        "piqa:bpb:olmo3base",
        "coqa:bpb:olmo3base",
        "drop:bpb:olmo3base",
        "jeopardy:bpb:olmo3base",
        "naturalqs:bpb:olmo3base",
        "squad:bpb:olmo3base",
        "sciq:bpb:olmo3base",
        "qasper_yesno:bpb:olmo3base",
        get_suite("basic_skills:bpb:olmo3base"),
        "lab_bench_dbqa:bpb:olmo3base",
        "lab_bench_protocolqa:bpb:olmo3base",
        "lambada:bpb:olmo3base",
        "medmcqa:bpb:olmo3base",
        "medqa_en:bpb:olmo3base",
        "sciriff_yesno:bpb:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:easy:math:bpb",
    tasks=(get_suite("minerva_math:bpb:olmo3base"),),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:easy:code:bpb",
    tasks=(
        "codex_humaneval:bpb:olmo3base",
        "mbpp:bpb:olmo3base",
        get_suite("mt_mbpp:bpb:olmo3base"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:mcqa_stem",
    tasks=(
        get_suite("arc:mc:olmo3base"),
        get_suite("mmlu:stem:mc:olmo3base"),
        "medmcqa:mc:olmo3base",
        "medqa_en:mc:olmo3base",
        "sciq:mc:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:mcqa_non_stem",
    tasks=(
        get_suite("mmlu:humanities:mc:olmo3base"),
        get_suite("mmlu:other:mc:olmo3base"),
        get_suite("mmlu:social_sciences:mc:olmo3base"),
        "csqa:mc:olmo3base",
        "piqa:mc:olmo3base",
        "socialiqa:mc:olmo3base",
        "coqa:mc:olmo3base",
        "drop:mc:olmo3base",
        "jeopardy:mc:olmo3base",
        "naturalqs:mc:olmo3base",
        "squad:mc:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:gen",
    tasks=(
        "hellaswag:rc:olmo3base",
        "lambada",
        "winogrande:rc:olmo3base",
        get_suite("basic_skills:rc:olmo3base"),
        "drop:gen:olmo3base",
        "jeopardy:gen:olmo3base",
        "squad:gen:olmo3base",
        "coqa:gen:olmo3base",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

make_suite(
    name="olmobase:math",
    tasks=(
        "gsm8k:olmo3base",
        get_suite("gsm_symb:olmo3base"),
        get_suite("minerva_math:olmo3base"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
)

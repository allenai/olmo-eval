"""Development and paper benchmark suites (OLMo2, OLMo3)."""

# Import suites from other modules to use in composite suites
from olmo_eval.evals.suites.code import (
    CRUX_EVAL,
    FIM_OLMO3,
    MT_MBPP_V2FIX,
    MULTIPL_E_HE_N32_V2,
    MULTIPL_E_HE_N32_V2_PASS_AT_16,
    MULTIPL_E_MBPP_N32_V2,
    MULTIPL_E_MBPP_N32_V2_PASS_AT_16,
)
from olmo_eval.evals.suites.core_tasks import (
    ARC_BPB_FULL,
    ARC_MC_XLARGE,
    ARC_RC_FULL,
    BASIC_BPB,
    BASIC_RC,
    CORE_MC,
    CORE_RC,
    GEN,
    GEN_XLARGE,
)
from olmo_eval.evals.suites.math import (
    DEEPMIND_MATH_HELDOUT,
    GSM_SYMB_N8_V2,
    GSM_SYMB_N8_V2_PASS_AT_4,
    MINERVA,
    MINERVA_BPB,
    MINERVA_MIDTRAIN,
    MINERVA_N4_V2,
)
from olmo_eval.evals.suites.mmlu import (
    MMLU_BPB,
    MMLU_COT_MIDTRAIN,
    MMLU_COT_THINKER,
    MMLU_HUMANITIES_MC,
    MMLU_MC,
    MMLU_OTHER_MC,
    MMLU_PRO_MC,
    MMLU_RC,
    MMLU_SOCIAL_SCIENCES_MC,
    MMLU_STEM_MC,
)
from olmo_eval.evals.suites.multiturn import (
    OMEGA_0SHOT_CHAT,
    OMEGA_MIDTRAIN,
    STYLED_MATH500_THINKER,
)
from olmo_eval.evals.suites.reasoning import (
    AGI_EVAL,
    AGI_EVAL_MIDTRAIN,
    AGI_EVAL_THINKER,
    BBH_COT_HELDOUT,
    BBH_COT_THINKER,
)
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# =============================================================================
# OLMo3 Dev 1B Suites
# =============================================================================

OLMO3_DEV_1B_MATH_BPB = make_suite(
    "olmo3:dev:1b:math:bpb",
    (MINERVA_BPB,),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B math development tasks (BPB)",
)

OLMO3_DEV_1B_CODE_BPB = make_suite(
    "olmo3:dev:1b:code:bpb",
    ("humaneval:3shot:bpb::none", "mbpp:3shot:bpb::none", MT_MBPP_V2FIX),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B code development tasks (BPB)",
)

OLMO3_DEV_1B_QA_BPB = make_suite(
    "olmo3:dev:1b:qa:bpb",
    (
        ARC_BPB_FULL,
        MMLU_BPB,
        "csqa:bpb::olmes:full",
        "hellaswag:bpb::olmes:full",
        "winogrande:bpb::olmes:full",
        "socialiqa:bpb::olmes:full",
        "piqa:bpb::olmes:full",
        "coqa:bpb::gen2mc",
        "drop:bpb::gen2mc",
        "jeopardy:bpb::gen2mc",
        "naturalqs:bpb::gen2mc",
        "squad:bpb::gen2mc",
        "sciq:bpb::olmo3",
        "qasper_yesno:bpb::olmes",
        BASIC_BPB,
        "lab_bench_dbqa:bpb",
        "lab_bench_protocolqa:bpb",
        "lambada:bpb",
        "medmcqa:bpb::none",
        "medqa_en:bpb::none",
        "sciriff_yesno:bpb::olmes",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B QA development tasks (BPB)",
)

OLMO3_DEV_1B_QA_RC = make_suite(
    "olmo3:dev:1b:qa:rc",
    (
        ARC_RC_FULL,
        MMLU_RC,
        "csqa:rc::olmes:full",
        "hellaswag:rc::olmes:full",
        "winogrande:rc::olmes:full",
        "socialiqa:rc::olmes:full",
        "piqa:rc::olmes:full",
        "coqa:rc::gen2mc",
        "drop:rc::gen2mc",
        "jeopardy:rc::gen2mc",
        "naturalqs:rc::gen2mc",
        "squad:rc::gen2mc",
        "sciq:rc::olmo3",
        "qasper_yesno:rc::olmes",
        BASIC_RC,
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "lambada",
        "medmcqa:rc::none",
        "medqa_en:rc::none",
        "sciriff_yesno:rc::olmes",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B QA development tasks (RC)",
)

OLMO3_DEV_1B_BPB = make_suite(
    "olmo3:dev:1b:bpb",
    (OLMO3_DEV_1B_QA_BPB, MINERVA_BPB, OLMO3_DEV_1B_CODE_BPB),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B combined development tasks (BPB)",
)

OLMO3_DEV_1B_QA_BPB_V2 = make_suite(
    "olmo3:dev:1b:qa:bpb:v2",
    (
        ARC_BPB_FULL,
        MMLU_BPB,
        "csqa:bpb::olmes:full",
        "hellaswag:bpb::olmes:full",
        "winogrande:bpb::olmes:full",
        "socialiqa:bpb::olmes:full",
        "piqa:bpb::olmes:full",
        "coqa:bpb::gen2mc:xlarge",
        "drop:bpb::gen2mc:xlarge",
        "jeopardy:bpb::gen2mc:xlarge",
        "naturalqs:bpb::gen2mc:xlarge",
        "squad:bpb::gen2mc:xlarge",
        "sciq:bpb::olmo3",
        "qasper_yesno:bpb::olmes",
        BASIC_BPB,
        "lab_bench_dbqa:bpb",
        "lab_bench_protocolqa:bpb",
        "lambada:bpb",
        "medmcqa:bpb::none",
        "medqa_en:bpb::none",
        "sciriff_yesno:bpb::olmes",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B QA development tasks (BPB v2)",
)

OLMO3_DEV_1B_QA_RC_V2 = make_suite(
    "olmo3:dev:1b:qa:rc:v2",
    (
        ARC_RC_FULL,
        MMLU_RC,
        "csqa:rc::olmes:full",
        "hellaswag:rc::olmes:full",
        "winogrande:rc::olmes:full",
        "socialiqa:rc::olmes:full",
        "piqa:rc::olmes:full",
        "coqa:rc::gen2mc:xlarge",
        "drop:rc::gen2mc:xlarge",
        "jeopardy:rc::gen2mc:xlarge",
        "naturalqs:rc::gen2mc:xlarge",
        "squad:rc::gen2mc:xlarge",
        "sciq:rc::olmo3",
        "qasper_yesno:rc::olmes",
        BASIC_RC,
        "lab_bench_dbqa",
        "lab_bench_protocolqa",
        "lambada",
        "medmcqa:rc::none",
        "medqa_en:rc::none",
        "sciriff_yesno:rc::olmes",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 1B QA development tasks (RC v2)",
)


# =============================================================================
# OLMo3 Dev 7B Suites
# =============================================================================

OLMO3_DEV_7B_MATH_V2 = make_suite(
    "olmo3:dev:7b:math:v2",
    ("gsm8k::olmo3:n8:v2", GSM_SYMB_N8_V2, MINERVA_N4_V2),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B math development tasks v2",
)

OLMO3_DEV_7B_CODE_GEN_V2 = make_suite(
    "olmo3:dev:7b:code_gen:v2",
    (
        "bigcodebench:3shot::olmo3:v2",
        "humaneval:3shot::olmo3:n32:v2",
        "deepseek_leetcode::olmo3:n32:v2",
        "ds1000:3shot::olmo3:v2",
        "mbpp:3shot::olmo3:n32:v2",
        MULTIPL_E_HE_N32_V2,
        MULTIPL_E_MBPP_N32_V2,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B code generation v2",
)

OLMO3_DEV_7B_CODE_GEN_V2_FAST = make_suite(
    "olmo3:dev:7b:code_gen:v2:fast",
    (
        "bigcodebench:3shot::olmo3:v2",
        "humaneval:3shot::olmo3:n32:v2",
        "ds1000:3shot::olmo3:v2",
        "mbpp:3shot::olmo3:n32:v2",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B code generation v2 (fast subset)",
)

OLMO3_DEV_7B_CODE_FIM = make_suite(
    "olmo3:dev:7b:code_fim",
    (FIM_OLMO3,),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B fill-in-the-middle code",
)

OLMO3_DEV_7B_GEN = make_suite(
    "olmo3:dev:7b:gen",
    (
        "hellaswag:rc::xlarge",
        "winogrande:rc::xlarge",
        "lambada",
        BASIC_RC,
        "drop::xlarge",
        "jeopardy::xlarge",
        "naturalqs::xlarge",
        "squad::xlarge",
        "coqa::xlarge",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B generation tasks",
)

OLMO3_DEV_7B_MCQA_STEM = make_suite(
    "olmo3:dev:7b:mcqa:stem",
    (
        ARC_MC_XLARGE,
        MMLU_STEM_MC,
        "medmcqa:mc::none",
        "medqa_en:mc::none",
        "sciq:mc::xlarge",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B MCQA STEM tasks",
)

OLMO3_DEV_7B_MCQA_NON_STEM = make_suite(
    "olmo3:dev:7b:mcqa:non_stem",
    (
        MMLU_HUMANITIES_MC,
        MMLU_SOCIAL_SCIENCES_MC,
        MMLU_OTHER_MC,
        "csqa:mc::xlarge",
        "piqa:mc::xlarge",
        "socialiqa:mc::xlarge",
        "coqa:mc::gen2mc",
        "drop:mc::gen2mc",
        "jeopardy:mc::gen2mc",
        "naturalqs:mc::gen2mc",
        "squad:mc::gen2mc",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B MCQA non-STEM tasks",
)

OLMO3_DEV_7B_MCQA_NON_STEM_V2 = make_suite(
    "olmo3:dev:7b:mcqa:non_stem:v2",
    (
        MMLU_HUMANITIES_MC,
        MMLU_SOCIAL_SCIENCES_MC,
        MMLU_OTHER_MC,
        "csqa:mc::xlarge",
        "piqa:mc::xlarge",
        "socialiqa:mc::xlarge",
        "coqa:mc::gen2mc:xlarge",
        "drop:mc::gen2mc:xlarge",
        "jeopardy:mc::gen2mc:xlarge",
        "naturalqs:mc::gen2mc:xlarge",
        "squad:mc::gen2mc:xlarge",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B MCQA non-STEM v2",
)

OLMO3_DEV_7B_CODE_GEN_MINI_PASS_AT_16 = make_suite(
    "olmo3:dev:7b:code_gen_mini:v2:n32:pass_at_16",
    (
        "deepseek_leetcode::olmo3:n32:v2:pass_at_16",
        "humaneval:3shot::olmo3:n32:v2:pass_at_16",
        "mbpp:3shot::olmo3:n32:v2:pass_at_16",
        MULTIPL_E_HE_N32_V2_PASS_AT_16,
        MULTIPL_E_MBPP_N32_V2_PASS_AT_16,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B code gen mini with pass@16",
)


# =============================================================================
# OLMo3 Midtrain Suites
# =============================================================================

OLMO3_DEV_MIDTRAIN_V1 = make_suite(
    "olmo3:dev:midtrain:v1",
    (
        "ifeval::hamish_zs_reasoning",
        STYLED_MATH500_THINKER,
        "gsm8k::zs_cot_latex",
        MINERVA,
        "minerva_math_500::hamish_zs_reasoning",
        "aime::hamish_zs_reasoning",
        OMEGA_0SHOT_CHAT,
        "humanevalplus:0-shot-chat::tulu-thinker",
        "mbppplus:0-shot-chat::tulu-thinker",
        "livecodebench_codegeneration::tulu-thinker",
        BBH_COT_THINKER,
        "zebralogic::hamish_zs_reasoning",
        "gpqa:0shot_cot::olmo3:thinker",
        "popqa::olmo3:thinker",
        AGI_EVAL_THINKER,
        MMLU_COT_THINKER,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 midtrain development suite v1",
)

OLMO3_DEV_MIDTRAIN_V2 = make_suite(
    "olmo3:dev:midtrain:v2",
    (
        "ifeval::olmo3:midtrain",
        "gsm8k::olmo3:midtrain",
        MINERVA_MIDTRAIN,
        "minerva_math_500::olmo3:midtrain",
        "aime:2024::olmo3:midtrain",
        "aime:2025::olmo3:midtrain",
        "omega_500::olmo3:midtrain",
        OMEGA_MIDTRAIN,
        # "humanevalplus::olmo3:midtrain", TODO(undfined): Enable once we add code execution
        # "mbppplus::olmo3:midtrain",
        # "livecodebench_codegeneration::olmo3:midtrain",
        # BBH_COT_MIDTRAIN,
        "gpqa::olmo3:midtrain",
        "zebralogic::olmo3:midtrain",
        "popqa::olmo3:midtrain",
        AGI_EVAL_MIDTRAIN,
        MMLU_COT_MIDTRAIN,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 midtrain development suite v2",
)

OLMO3_BASE_HELDOUT = make_suite(
    "olmo3:base_heldout",
    (
        BBH_COT_HELDOUT,
        MMLU_PRO_MC,
        DEEPMIND_MATH_HELDOUT,
        # "lbpp::olmo3" TODO(undfined): Enable once we add code execution
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 base heldout evaluation suite",
)


# =============================================================================
# Display-Only Main Suites
# =============================================================================

OLMO2_PAPER = make_suite(
    "olmo2:paper",
    (
        "arc_challenge:rc::olmes",
        "arc_challenge:mc::olmes",
        "hellaswag:rc::olmes",
        "hellaswag:mc::olmes",
        "winogrande:rc::olmes",
        "winogrande:mc::olmes",
        "naturalqs::olmes",
        "drop::olmes",
        AGI_EVAL,
        "gsm8k::olmes",
        MMLU_MC,
        MMLU_RC,
        CORE_MC,
        MMLU_PRO_MC,
        "triviaqa::olmes",
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo2 paper benchmark suite",
)

OLMO2_DEV_7B = make_suite(
    "olmo2:dev:7b",
    (
        "arc_challenge:mc::olmes",
        "arc_easy:mc::olmes",
        "hellaswag:mc::olmes",
        "naturalqs::olmes",
        "gsm8k::olmo1",
        MMLU_MC,
        CORE_MC,
        GEN,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo2 7B development suite",
)

OLMO2_DEV_1B = make_suite(
    "olmo2:dev:1b",
    (
        "arc_challenge:rc::olmes",
        "arc_easy:rc::olmes",
        "hellaswag:rc::olmes",
        "gsm8k::olmo1",
        MMLU_RC,
        CORE_RC,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo2 1B development suite",
)

OLMO3_DEV_1B_MAIN = make_suite(
    "olmo3:dev:1b:main",
    (
        OLMO3_DEV_1B_MATH_BPB,
        OLMO3_DEV_1B_CODE_BPB,
        OLMO3_DEV_1B_QA_RC,
        ARC_RC_FULL,
        "hellaswag:rc::olmes:full",
        BASIC_RC,
        MT_MBPP_V2FIX,
        MMLU_RC,
        MMLU_BPB,
        "humaneval:3shot:bpb::none",
        "mbpp:3shot:bpb::none",
        MINERVA_BPB,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 1B main development suite",
)

OLMO3_DEV_7B_MAIN_V2 = make_suite(
    "olmo3:dev:7b:main:v2",
    (
        OLMO3_DEV_7B_MCQA_STEM,
        OLMO3_DEV_7B_MCQA_NON_STEM,
        OLMO3_DEV_7B_GEN,
        OLMO3_DEV_7B_MATH_V2,
        OLMO3_DEV_7B_CODE_GEN_V2,
        OLMO3_DEV_7B_CODE_GEN_MINI_PASS_AT_16,
        OLMO3_DEV_7B_CODE_FIM,
        ARC_MC_XLARGE,
        MMLU_MC,
        GEN_XLARGE,
        BASIC_RC,
        "gsm8k::olmo3:n8:v2",
        GSM_SYMB_N8_V2,
        GSM_SYMB_N8_V2_PASS_AT_4,
        MINERVA_N4_V2,
        "minerva_math_500::olmo3:n32:v2",
        "minerva_math_500::olmo3:n32:v2:pass_at_16",
        "humaneval:3shot::olmo3:n32:v2",
        "mbpp:3shot::olmo3:n32:v2",
        MULTIPL_E_HE_N32_V2,
        MULTIPL_E_MBPP_N32_V2,
        CRUX_EVAL,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 7B main development suite v2",
)

OLMO3_DEV_7B_MAIN_V2_FAST = make_suite(
    "olmo3:dev:7b:main:v2:fast",
    (
        OLMO3_DEV_7B_MCQA_STEM,
        OLMO3_DEV_7B_MCQA_NON_STEM,
        OLMO3_DEV_7B_GEN,
        OLMO3_DEV_7B_MATH_V2,
        OLMO3_DEV_7B_CODE_GEN_V2_FAST,
        ARC_MC_XLARGE,
        MMLU_MC,
        GEN_XLARGE,
        BASIC_RC,
        "gsm8k::olmo3:n8:v2",
        GSM_SYMB_N8_V2,
        MINERVA_N4_V2,
        "humaneval:3shot::olmo3:n32:v2",
        "mbpp:3shot::olmo3:n32:v2",
        OLMO3_DEV_1B_CODE_BPB,  # TODO(undfined): Temporary inclusion to cover code for now
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="OLMo3 7B main development suite v2 (fast)",
)

OLMO3_PAPER = make_suite(
    "olmo3:paper",
    (
        # Base easy (1B)
        OLMO3_DEV_1B_MATH_BPB,
        # OLMO3_DEV_1B_CODE_BPB, TODO(undfined): Enable once we add code execution
        OLMO3_DEV_1B_QA_BPB_V2,
        OLMO3_DEV_1B_QA_RC_V2,
        # Base (7B)
        OLMO3_DEV_7B_MCQA_STEM,
        OLMO3_DEV_7B_MCQA_NON_STEM_V2,
        OLMO3_DEV_7B_GEN,
        OLMO3_DEV_7B_MATH_V2,
        # OLMO3_DEV_7B_CODE_GEN_V2, TODO(undfined): Enable once we add code execution
        # OLMO3_DEV_7B_CODE_FIM, TODO(undfined): Enable once we add code execution
        # Base chat
        OLMO3_DEV_MIDTRAIN_V2,
        # Heldout
        OLMO3_BASE_HELDOUT,
    ),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 paper comprehensive benchmark suite",
)


def _build_mixed_priority_tasks() -> tuple[str, ...]:
    """Build olmo3:paper tasks with mixed priorities (high, normal, low)."""
    tasks = OLMO3_PAPER.expand()
    n = len(tasks)
    third = n // 3
    result = []
    for i, task in enumerate(tasks):
        if i < third:
            result.append(f"{task}@high")
        elif i < 2 * third:
            result.append(f"{task}@normal")
        else:
            result.append(f"{task}@low")
    return tuple(result)


OLMO3_PAPER_MIXED_PRIORITY = make_suite(
    "olmo3:paper:mixed-priority",
    _build_mixed_priority_tasks(),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="OLMo3 paper suite with mixed priorities (high/normal/low)",
)

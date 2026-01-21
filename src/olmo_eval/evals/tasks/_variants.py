"""Common task variants registration.

This module registers common variants like :mc (multiple choice) for tasks.
Variants are format modifiers that affect how tasks are evaluated.

For tasks already configured as multiple choice (with MultipleChoiceFormatter),
the :mc variant is a no-op. For generation tasks, :mc would convert them to
multiple choice format.
"""

from olmo_eval.core import (
    BitsPerByteScorer,
    BPBMetric,
    MCQAChatFormatter,
    PPLFormatter,
    SamplingParams,
)

from .core.registry import _tasks, register_variant

# Tasks that are already multiple choice by default
# For these, :mc is a no-op (empty overrides)
_MC_TASKS = [
    # ARC
    "arc_challenge",
    "arc_easy",
    # AGI Eval (subset that are MC)
    "agi_eval_aqua_rat",
    "agi_eval_gaokao_english",
    "agi_eval_logiqa_en",
    "agi_eval_lsat_ar",
    "agi_eval_lsat_lr",
    "agi_eval_lsat_rc",
    "agi_eval_sat_en",
    "agi_eval_sat_en_without_passage",
    "agi_eval_sat_math",
    # Basic Skills
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_pattern",
    "basic_skills_string_operations",
    # BBH (Big-Bench Hard) - all are MC
    "bbh_boolean_expressions",
    "bbh_causal_judgement",
    "bbh_date_understanding",
    "bbh_disambiguation_qa",
    "bbh_dyck_languages",
    "bbh_formal_fallacies",
    "bbh_geometric_shapes",
    "bbh_hyperbaton",
    "bbh_logical_deduction_five_objects",
    "bbh_logical_deduction_seven_objects",
    "bbh_logical_deduction_three_objects",
    "bbh_movie_recommendation",
    "bbh_multistep_arithmetic_two",
    "bbh_navigate",
    "bbh_object_counting",
    "bbh_penguins_in_a_table",
    "bbh_reasoning_about_colored_objects",
    "bbh_ruin_names",
    "bbh_salient_translation_error_detection",
    "bbh_snarks",
    "bbh_sports_understanding",
    "bbh_temporal_sequences",
    "bbh_tracking_shuffled_objects_five_objects",
    "bbh_tracking_shuffled_objects_seven_objects",
    "bbh_tracking_shuffled_objects_three_objects",
    "bbh_web_of_lies",
    "bbh_word_sorting",
    # Core tasks
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    # GPQA
    "gpqa",
    "gpqa_diamond",
    "super_gpqa",
    # Medical
    "medmcqa",
    # QA tasks (can be MC)
    "coqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "squad",
]


def register_mc_variants() -> None:
    """Register :mc variant for all multiple choice tasks.

    For tasks already configured as MC, this is a no-op.
    """
    for task_name in _MC_TASKS:
        if task_name in _tasks:
            # Empty overrides = no-op, just allows the :mc suffix
            register_variant(task_name, "mc")


def register_all_mc_variants() -> None:
    """Register :mc variant for ALL registered tasks.

    This is more permissive - allows :mc suffix on any task.
    Useful when task configs already handle MC format appropriately.
    """
    for task_name in list(_tasks.keys()):
        register_variant(task_name, "mc")


def register_bpb_variants() -> None:
    """Register :bpb variant for ALL registered tasks."""
    for task_name in list(_tasks.keys()):
        register_variant(
            task_name,
            "bpb",
            formatter=PPLFormatter(),
            scorers=(BitsPerByteScorer(),),
            metrics=(BPBMetric()),
        )


def register_rc_variants() -> None:
    """Register :rc variant for ALL registered tasks.

    The :rc variant indicates reading comprehension (cloze) format.
    This uses completion-based scoring with answer continuation.
    """
    for task_name in list(_tasks.keys()):
        register_variant(task_name, "rc")


def register_3shot_variants() -> None:
    """Register :3shot variant for ALL registered tasks.

    The :3shot variant forces 3-shot evaluation regardless of regime.
    """
    for task_name in list(_tasks.keys()):
        register_variant(task_name, "3shot", num_fewshot=3)


# AGI Eval CoT variant configuration
AGI_EVAL_COT_SYSTEM_PROMPT = (
    "Answer the following multiple-choice question by giving the correct answer "
    "letter in parentheses. Provide CONCISE reasoning for the answer, and make "
    "sure to finish the response with 'Therefore, the answer is (ANSWER_LETTER)' "
    "where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc."
)

COT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=131072,
)

_AGI_EVAL_TASKS = [
    "agi_eval_aqua_rat",
    "agi_eval_gaokao_english",
    "agi_eval_logiqa_en",
    "agi_eval_lsat_ar",
    "agi_eval_lsat_lr",
    "agi_eval_lsat_rc",
    "agi_eval_sat_en",
    "agi_eval_sat_en_without_passage",
    "agi_eval_sat_math",
]


def register_cot_variants() -> None:
    """Register :cot variant for AGI Eval tasks.

    The :cot variant uses chat-based chain-of-thought generation
    with answer extraction for multiple choice questions.
    """
    for task_name in _AGI_EVAL_TASKS:
        if task_name in _tasks:
            register_variant(
                task_name,
                "cot",
                formatter=MCQAChatFormatter(system_prompt=AGI_EVAL_COT_SYSTEM_PROMPT),
                sampling_params=COT_SAMPLING_PARAMS,
                num_fewshot=0,
            )


def register_all_variants() -> None:
    """Register all common variants for all tasks."""
    register_all_mc_variants()
    register_bpb_variants()
    register_rc_variants()
    register_3shot_variants()
    register_cot_variants()


# Auto-register all variants for all tasks when this module is imported
# This allows specs like "arc_easy:mc::olmes", "sciq:bpb::olmo3" to work
register_all_variants()

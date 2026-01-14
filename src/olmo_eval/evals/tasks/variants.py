"""Common task variants registration.

This module registers common variants like :mc (multiple choice) for tasks.
Variants are format modifiers that affect how tasks are evaluated.

For tasks already configured as multiple choice (with MultipleChoiceFormatter),
the :mc variant is a no-op. For generation tasks, :mc would convert them to
multiple choice format.
"""

from .registry import _tasks, register_variant

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


# Auto-register :mc variants for all tasks when this module is imported
# This allows specs like "arc_easy:mc::olmes" to work
register_all_mc_variants()

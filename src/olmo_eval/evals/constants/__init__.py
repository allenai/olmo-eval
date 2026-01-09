"""Constants for evaluation benchmarks and tasks.

This module provides centralized access to evaluation-related constants:

- **benchmarks**: Standard evaluation task definitions (MMLU, BBH, etc.)
- **long_context**: HELMET and RULER long-context benchmarks
- **code**: Code generation and FIM evaluation tasks
"""

# Benchmark constants
from olmo_eval.evals.constants.benchmarks import (
    AGI_EVAL_ENGLISH_TASKS,
    ALL_CORE_TASKS,
    ALL_GEN_TASKS,
    ALL_GEN_XLARGE_TASKS,
    ALL_GSM_SYMB_TASKS,
    ALL_MINERVA_TASKS,
    ARC_TASKS,
    BASIC_SKILLS_TASKS,
    BBH_TASKS,
    DEEPMIND_MATH_CATEGORIES,
    IFBENCH_MT_TASKS,
    IFEVAL_MT_TASKS,
    MMLU_CATEGORIES,
    MMLU_PRO_CATEGORIES,
    MMLU_SUBCATEGORIES,
    MT_EVAL_TASKS,
    MULTITURN_ALPACAEVAL_TASKS,
    OMEGA_SUB_CATEGORIES,
    STYLED_TASKS,
    STYLED_TASKS_POPQA,
)

# Code evaluation constants
from olmo_eval.evals.constants.code import (
    ALL_CODEX_TASKS,
    CRUX_EVAL_TASKS,
    DEEPSEEK_CODER_FIM,
    FIM_CONFIGS,
    FIM_TASKS,
    MULTILINGUAL_MBPP_TASKS,
    MULTILINGUAL_MBPP_TASKS_V2,
    MULTIPL_E_HE_TASKS,
    MULTIPL_E_MBPP_TASKS,
    OLMO_FIM,
    SANTACODER_FIM,
    STARCODER_CODEX_TASKS,
    STARCODER_FIM,
    STARCODER_PASS_AT_1_TASKS,
    FIMConfig,
)

# Long-context benchmark constants
from olmo_eval.evals.constants.long_context import (
    HELMET_SUITES,
    RULER_SUITES,
)

__all__ = [
    # Benchmarks
    "AGI_EVAL_ENGLISH_TASKS",
    "ALL_CORE_TASKS",
    "ALL_GEN_TASKS",
    "ALL_GEN_XLARGE_TASKS",
    "ALL_GSM_SYMB_TASKS",
    "ALL_MINERVA_TASKS",
    "ARC_TASKS",
    "BASIC_SKILLS_TASKS",
    "BBH_TASKS",
    "DEEPMIND_MATH_CATEGORIES",
    "IFBENCH_MT_TASKS",
    "IFEVAL_MT_TASKS",
    "MMLU_CATEGORIES",
    "MMLU_PRO_CATEGORIES",
    "MMLU_SUBCATEGORIES",
    "MT_EVAL_TASKS",
    "MULTITURN_ALPACAEVAL_TASKS",
    "OMEGA_SUB_CATEGORIES",
    "STYLED_TASKS",
    "STYLED_TASKS_POPQA",
    # Long-context
    "HELMET_SUITES",
    "RULER_SUITES",
    # Code
    "ALL_CODEX_TASKS",
    "CRUX_EVAL_TASKS",
    "DEEPSEEK_CODER_FIM",
    "FIM_CONFIGS",
    "FIM_TASKS",
    "FIMConfig",
    "OLMO_FIM",
    "MULTILINGUAL_MBPP_TASKS",
    "MULTILINGUAL_MBPP_TASKS_V2",
    "MULTIPL_E_HE_TASKS",
    "MULTIPL_E_MBPP_TASKS",
    "SANTACODER_FIM",
    "STARCODER_CODEX_TASKS",
    "STARCODER_FIM",
    "STARCODER_PASS_AT_1_TASKS",
]

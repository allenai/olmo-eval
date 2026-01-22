"""Common regime registrations.

This module registers common regimes (configuration presets) for tasks.
Regimes are evaluation configurations that can be applied to tasks using
the ::regime suffix (e.g., arc_challenge::olmes).

Regimes are applied after variants, allowing combinations like:
    arc_challenge:mc::olmes  (multiple-choice variant with olmes regime)
"""

from olmo_eval.core import SamplingParams
from olmo_eval.evals.constants.code import HUMANEVAL_STOP_SEQUENCES, MBPP_STOP_SEQUENCES

from .core.registry import _tasks, register_regime


def register_olmes_regimes() -> None:
    """Register the 'olmes' regime for all tasks.

    The 'olmes' regime uses OLMo-style evaluation settings:
    - 5-shot evaluation
    - Consistent fewshot seed (1234)
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmes",
            num_fewshot=5,
            fewshot_seed=1234,
        )


def register_olmes_full_regimes() -> None:
    """Register the 'olmes:full' regime for all tasks.

    The 'olmes:full' regime uses full dataset evaluation:
    - 5-shot evaluation
    - No instance limit (full dataset)
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmes:full",
            num_fewshot=5,
            fewshot_seed=1234,
            limit=None,
        )


def register_gen2mc_regimes() -> None:
    """Register the 'gen2mc' regime for QA tasks.

    The 'gen2mc' regime converts generation tasks to multiple-choice format.
    """
    # QA tasks that support gen2mc conversion
    gen2mc_tasks = [
        "coqa",
        "drop",
        "jeopardy",
        "naturalqs",
        "squad",
    ]
    for task_name in gen2mc_tasks:
        if task_name in _tasks:
            register_regime(
                task_name,
                "gen2mc",
                num_fewshot=5,
                fewshot_seed=1234,
            )


def register_gen2mc_xlarge_regimes() -> None:
    """Register the 'gen2mc:xlarge' regime for QA tasks.

    The 'gen2mc:xlarge' regime uses larger sample sizes.
    """
    gen2mc_tasks = [
        "coqa",
        "drop",
        "jeopardy",
        "naturalqs",
        "squad",
    ]
    for task_name in gen2mc_tasks:
        if task_name in _tasks:
            register_regime(
                task_name,
                "gen2mc:xlarge",
                num_fewshot=5,
                fewshot_seed=1234,
            )


def register_zero_shot_regime() -> None:
    """Register the 'zero' regime for zero-shot evaluation."""
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "zero",
            num_fewshot=0,
        )


def register_olmo3_regimes() -> None:
    """Register the 'olmo3' regime for all tasks.

    The 'olmo3' regime uses OLMo3-style evaluation settings:
    - 5-shot evaluation
    - Consistent fewshot seed (42)
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmo3",
            num_fewshot=5,
            fewshot_seed=42,
        )


def register_olmo3_midtrain_regimes() -> None:
    """Register the 'olmo3:midtrain' regime for all tasks.

    The 'olmo3:midtrain' regime uses zero-shot evaluation
    for mid-training checkpoint evaluations.
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmo3:midtrain",
            num_fewshot=0,
        )


def register_xlarge_regimes() -> None:
    """Register the 'xlarge' regime for all tasks.

    The 'xlarge' regime uses full dataset evaluation:
    - 5-shot evaluation
    - No instance limit (full dataset)
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "xlarge",
            num_fewshot=5,
            fewshot_seed=42,
            limit=None,
        )


def register_none_regime() -> None:
    """Register the 'none' regime for zero-shot baseline evaluation."""
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "none",
            num_fewshot=0,
        )


def register_olmo3_n32_v2_regimes() -> None:
    """Register the 'olmo3:n32:v2' regime for code tasks.

    The 'olmo3:n32:v2' regime is for code generation with:
    - Zero-shot evaluation
    - 32 samples per problem (for pass@k)
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmo3:n32:v2",
            num_fewshot=0,
        )


def register_olmo3_v2_regimes() -> None:
    """Register the 'olmo3:v2' regime for code tasks.

    The 'olmo3:v2' regime is for code benchmarks with:
    - 3-shot evaluation
    """
    for task_name in list(_tasks.keys()):
        register_regime(
            task_name,
            "olmo3:v2",
            num_fewshot=3,
            fewshot_seed=42,
        )


def register_code_pass_at_k_regimes() -> None:
    """Register pass@k regimes for code generation tasks.

    These regimes configure sampling parameters for pass@k evaluation:
    - Temperature sampling (0.8) for diversity
    - Multiple samples (configurable k)
    - Appropriate stop sequences
    """
    # HumanEval tasks
    humaneval_tasks = [
        "humaneval",
        "humanevalplus",
    ]

    for task_name in humaneval_tasks:
        if task_name in _tasks:
            # pass@1 with temperature 0.8, 16 samples
            register_regime(
                task_name,
                "pass@1",
                num_fewshot=0,
                sampling_params=SamplingParams(
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.95,
                    num_samples=16,
                    stop_sequences=HUMANEVAL_STOP_SEQUENCES,
                ),
            )
            # pass@10 with 32 samples
            register_regime(
                task_name,
                "pass@10",
                num_fewshot=0,
                sampling_params=SamplingParams(
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.95,
                    num_samples=32,
                    stop_sequences=HUMANEVAL_STOP_SEQUENCES,
                ),
            )

    # MBPP tasks
    mbpp_tasks = ["mbpp", "mbpp_plus", "mbppplus"]

    for task_name in mbpp_tasks:
        if task_name in _tasks:
            # pass@1 with temperature 0.8, 16 samples
            register_regime(
                task_name,
                "pass@1",
                num_fewshot=0,
                sampling_params=SamplingParams(
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.95,
                    num_samples=16,
                    stop_sequences=MBPP_STOP_SEQUENCES,
                ),
            )
            # pass@10 with 32 samples
            register_regime(
                task_name,
                "pass@10",
                num_fewshot=0,
                sampling_params=SamplingParams(
                    max_tokens=1024,
                    temperature=0.8,
                    top_p=0.95,
                    num_samples=32,
                    stop_sequences=MBPP_STOP_SEQUENCES,
                ),
            )


def register_all_regimes() -> None:
    """Register all common regimes."""
    register_olmes_regimes()
    register_olmes_full_regimes()
    register_gen2mc_regimes()
    register_gen2mc_xlarge_regimes()
    register_zero_shot_regime()
    register_olmo3_regimes()
    register_olmo3_midtrain_regimes()
    register_xlarge_regimes()
    register_none_regime()
    register_olmo3_n32_v2_regimes()
    register_olmo3_v2_regimes()
    register_code_pass_at_k_regimes()


# Auto-register all regimes when this module is imported
register_all_regimes()

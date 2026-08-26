"""Olmo 3.5 post-training suites: the per-checkpoint dev tier and the AAI tier.

Two-tier structure. ``olmo35_dev`` is the decision suite — cheap enough to run
per checkpoint, chosen for being unsaturated and discriminative rather than for
index coverage. ``olmo35_aai`` is the external-scorecard tier: the Artificial
Analysis Intelligence Index components we can run locally. Run it at
milestones, and never select checkpoints on it — its value as a held-out
signal depends on nobody optimizing against it. The two tiers deliberately
share no task except none: GPQA Diamond, the one dev-bundle candidate that is
also an AAI component, lives only in the AAI tier for that reason.

The AAI tier tracks Intelligence Index v4.1.1. Task-suite members here cover
only part of the index; the coding/agentic components are external evals with
their own runner (``olmo-eval run-external``):

    scicode           implemented (external)
    terminal_bench    implemented (external; AAI pins Terminal-Bench v2.1 —
                      verify the pin before comparing to published index scores)
    tau2              implemented (external) but the WRONG GENERATION for
                      v4.1.1, which uses τ³-Banking; τ²'s domains are
                      airline/retail/telecom
    HLE, CritPt, AA-LCR, AA-Omniscience, GDPval-AA v2 — not implemented

Launch notes: safety suites (``safety_instruct``/``safety_thinking``) are the
companion tier, not members here — their judge-scored tasks need
``-o auxiliary_providers.wg_judge.kind=vllm_server``
``-o auxiliary_providers.wg_judge.model=allenai/wildguard`` after ``--harness``
and 2 GPUs, so they launch separately. ``mmlu:cot`` and ``popqa:chat`` are the
expensive members; per-checkpoint runs can cap them at launch
(``-o limit=5000`` retains decision power on popqa; see the tracking issue).
"""

from olmo_eval.evals.suites.mmlu import MMLU_COT
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register

OLMO35_DEV = register(
    Suite(
        name="olmo35_dev",
        tasks=(
            "aime_2024:pass_at_32",
            "aime_2025:pass_at_32",
            "omega_500",
            "gpqa_main:cot",
            # zebralogic:chat currently scores in completion mode (#313);
            # numbers will shift once the formatter fix lands.
            "zebralogic:chat",
            "ifeval",
            # ifbench members inlined; ifeval_ood needs the IFBench_test
            # dataset switch to run without a fine-grained HF token.
            "ifeval_ood",
            "ifeval_mt_wildchat_unused_withRewrite",
            "ifeval_mt_ood_wildchat_unused_withRewrite",
            "popqa:chat",
            MMLU_COT,
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description="Olmo 3.5 per-checkpoint decision suite (dev tier)",
    )
)

OLMO35_AAI = make_suite(
    "olmo35_aai",
    ("gpqa_diamond:cot",),
    aggregation=AggregationStrategy.AVERAGE,
    description=(
        "Artificial Analysis Intelligence Index v4.1.1 components runnable as "
        "tasks. Milestones only; never select checkpoints on this tier. "
        "Coding/agentic components run via `olmo-eval run-external` "
        "(scicode, terminal_bench, tau2 — see module docstring for version "
        "caveats); HLE, CritPt, AA-LCR, AA-Omniscience, GDPval-AA not yet "
        "implemented."
    ),
)

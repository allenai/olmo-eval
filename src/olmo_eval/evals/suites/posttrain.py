"""Olmo 3.5 post-training dev and test suites, one pair per model size.

Membership follows the Olmo eval tracker's Eval list: Track is dev, Publish is test.
Dev suites are for hill-climbing and checkpoint selection; test suites are read only
for release reports. No task is in both suites of one size.

Left out: tracker rows with no olmo-eval task yet (CritPt, FrontierCode, BrowseComp-Plus,
AutomationBench, DeepSearchQA, Terminal-Bench-Science, FIRM, OpenAgentSafety, the
decontaminated paper-MATH), externally run evals (SWE-bench Verified, Terminal-Bench,
tau2-bench, AA-Briefcase, DeepScholar Bench), and the multimodal rows, which keep their own
suites. Safety tasks use the thinking-model judge variants and need the judge providers
shown in ``safety.py``. Set one generation budget per comparison with ``-o max_tokens``:
several task defaults are sized for non-reasoning models.
"""

# Defining modules, imported so their suites exist whatever the discovery order.
from olmo_eval.evals.suites import (  # noqa: F401
    arena_hard,
    bfcl,
    ctc,
    helmet,
    mmlu,
    mmlu_pro,
    ruler_plus,
    science,
)
from olmo_eval.evals.suites.math import OMEGA_DEV
from olmo_eval.evals.suites.registry import AggregationStrategy, get_suite, make_suite

_HMMT = ("hmmt_feb_2025:pass_at_32", "hmmt_nov_2025:pass_at_32", "hmmt_feb_2026:pass_at_32")
_AIME = ("aime_2025:pass_at_32", "aime_2026:pass_at_32")

_DEV_SHARED = (
    "gpqa_diamond:cot",
    get_suite("mmlu:cot"),
    "popqa",
    OMEGA_DEV,
    "math500",
    *_HMMT,
    "livecodebench:lite",
    "humaneval_plus",
    get_suite("bfcl"),
    "ifeval",
    "ifeval_ood",
    get_suite("ruler_plus_all__65536"),
    get_suite("helmet_all__65536"),
    "longbench_v2",
    get_suite("ctc:r64k"),
)

_TEST_SHARED = (
    get_suite("mmlu_pro:cot"),
    "omniscience",
    "zebralogic",
    *_AIME,
    "livecodebench_hidden",
    "do_anything_now:wg_judge_thinking",
    "wildguardtest:wg_judge_thinking",
    "wildjailbreak:wg_judge_thinking",
    "wmdp:mcq",
    "bbq:mcq",
    "strongreject:sr_judge_thinking",
)

make_suite(
    "posttrain:tiny:dev",
    (*_DEV_SHARED, "gpqa_main:cot", "mbpp_plus"),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Tiny post-training dev set (the tracker's Track column).",
)

make_suite(
    "posttrain:tiny:test",
    _TEST_SHARED,
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Tiny post-training test set (the tracker's Publish column).",
)

make_suite(
    "posttrain:small:dev",
    (
        *_DEV_SHARED,
        "mt_bench",
        "multi_if",
        "multichallenge",
        "harmbench:wg_judge_thinking",
        "xstest:wg_judge_thinking",
    ),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Small post-training dev set (the tracker's Track column).",
)

make_suite(
    "posttrain:small:test",
    (
        *_TEST_SHARED,
        "gpqa_main:cot",
        "hle:text:verified",
        get_suite("arena_hard_v2"),
        get_suite("frontierscience"),
    ),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Small post-training test set (the tracker's Publish column).",
)

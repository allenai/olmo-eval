"""Olmo 3.5 post-training dev and test suites, one pair per model size.

Membership follows the Olmo eval tracker's Eval list: Track is dev, Publish is test.
Dev suites are for hill-climbing and checkpoint selection; test suites are read only
for release reports. No task is in both suites of one size. Dev has to run fast, so it
reads fixed seeded samples of the two largest sets, popqa (1,000 of about 14,000 items)
and MMLU:cot (18 items per subject, 1,026 in all), and of RULER+ at 128K and 256K (128
items per task, against 512). HMMT draws 8 samples per problem, against AIME's 32 in test.

Left out: tracker rows with no olmo-eval task yet (CritPt, FrontierCode, BrowseComp-Plus,
AutomationBench, DeepSearchQA, Terminal-Bench-Science, FIRM, OpenAgentSafety, the
decontaminated paper-MATH), externally run evals (SWE-bench Verified, Terminal-Bench,
tau2-bench, AA-Briefcase, DeepScholar Bench), and the multimodal rows, which keep their own
suites. Also left out until olmo-eval has a configuration that lets a reasoning model
finish (their variants cap generation at 1,024 or 128 tokens, most with stop sequences):
MATH-500, HumanEval+, MBPP+ and LongBench v2.

Sampled entries run at the Olmo 3.5 models' suggested sampling, temperature 1.0 and top_p
0.95 (the ledger's setting), through ``:..._t10`` variants that keep each task's budget and
sample count; olmo-eval's own default for most of them is 0.6. Do not override sampling or
``max_tokens`` for the whole run. Dev reads RULER+ at 64K (full), 128K and 256K, and HELMET
and CTC at 64K. All three run greedy in completion mode with direct-answer budgets (CTC's
docs say a reasoning model may need a larger budget for CTC alone), and BFCL keeps its own
settings (temperature 0.001, 4,096 tokens), as the ledger's BFCL reads do. Safety tasks use
the thinking-model judge variants and need the judge providers shown in ``safety.py``;
Omniscience is graded by its default judge, gpt-4o-mini, through the OpenAI API.

Tiny dev needs no API key: it reads ``helmet_nojudge__65536``, which leaves out HELMET's
three API-judged tasks (NarrativeQA and both summarization tasks, so the summarization
category too). Small dev reads all of HELMET, since MT-Bench and MultiChallenge need an
API judge anyway. Every other suite needs ``OPENAI_API_KEY`` for its API-judged tasks.
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
from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    Suite,
    get_suite,
    make_suite,
    suite_exists,
)
from olmo_eval.evals.tasks.common import register_sampling_variant

# The Olmo 3.5 models' suggested sampling; each task keeps its budget and sample count.
MODEL_SAMPLING = {"temperature": 1.0, "top_p": 0.95}
# Entries that keep their own sampling: greedy direct-answer long context, and BFCL.
OWN_SAMPLING = frozenset(
    {
        "ruler_plus_all__65536",
        "ruler_plus_all__131072:hillclimb",
        "ruler_plus_all__262144:hillclimb",
        "helmet_all__65536",
        "helmet_nojudge__65536",
        "ctc:r64k",
        "bfcl",
    }
)


# Each ``_t10`` spec in these suites, mapped to the spec it was made from.
SAMPLED_FROM: dict[str, str] = {}


def _at_model_sampling(entry: str | Suite) -> str | Suite:
    """The entry at ``MODEL_SAMPLING`` (``:..._t10``), unless it keeps its own sampling."""
    if isinstance(entry, Suite):
        if entry.name in OWN_SAMPLING:
            return entry
        name = f"{entry.name}:t10"
        if suite_exists(name):
            return get_suite(name)
        return make_suite(
            name,
            tuple(_at_model_sampling(task) for task in entry.tasks),
            aggregation=entry.aggregation,
            description=f"{entry.description} At temperature 1.0 and top_p 0.95.".strip(),
        )
    if entry in OWN_SAMPLING:
        return entry
    spec = register_sampling_variant(entry, "t10", **MODEL_SAMPLING)
    SAMPLED_FROM[spec] = entry
    return spec


def _sampled(entries: tuple) -> tuple:
    return tuple(_at_model_sampling(entry) for entry in entries)


_HMMT = ("hmmt_feb_2025:pass_at_8", "hmmt_nov_2025:pass_at_8", "hmmt_feb_2026:pass_at_8")
_AIME = ("aime_2025:pass_at_32", "aime_2026:pass_at_32")


def _dev_shared(helmet_suite: str) -> tuple:
    return (
        "gpqa_diamond:cot",
        "gpqa_main:cot",
        get_suite("mmlu:cot:hillclimb"),
        "popqa:chat:hillclimb",
        OMEGA_DEV,
        *_HMMT,
        "livecodebench:lite",
        get_suite("bfcl"),
        "ifeval",
        "ifeval_ood",
        get_suite("ruler_plus_all__65536"),
        get_suite("ruler_plus_all__131072:hillclimb"),
        get_suite("ruler_plus_all__262144:hillclimb"),
        get_suite(helmet_suite),
        get_suite("ctc:r64k"),
    )


_TEST_SHARED = (
    get_suite("mmlu_pro:cot"),
    "omniscience:judge",
    "zebralogic:chat",
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
    _sampled(_dev_shared("helmet_nojudge__65536")),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Tiny post-training dev set (the tracker's Track column).",
)

make_suite(
    "posttrain:tiny:test",
    _sampled(_TEST_SHARED),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Tiny post-training test set (the tracker's Publish column).",
)

make_suite(
    "posttrain:small:dev",
    _sampled(
        (
            *_dev_shared("helmet_all__65536"),
            "mt_bench",
            "multi_if",
            "multichallenge",
            "harmbench:wg_judge_thinking",
            "xstest:wg_judge_thinking",
        )
    ),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Small post-training dev set (the tracker's Track column).",
)

make_suite(
    "posttrain:small:test",
    _sampled(
        (
            *_TEST_SHARED,
            "hle:text:verified",
            get_suite("arena_hard_v2"),
            get_suite("frontierscience"),
        )
    ),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Small post-training test set (the tracker's Publish column).",
)

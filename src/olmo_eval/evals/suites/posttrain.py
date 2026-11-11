"""Olmo 3.5 post-training dev and test suites, one pair per model size.

Membership follows the Olmo eval tracker's Eval list: Track is dev, Publish is test.
Dev suites are for hill-climbing and checkpoint selection; test suites are read only
for release reports. No task is in both suites of one size. Dev has to run fast, so it
reads fixed seeded samples of the two largest sets, popqa (1,000 of about 14,000 items)
and MMLU:cot (18 items per subject, 1,026 in all), and of RULER+ at 128K and 256K (128
items per task, against 512). HMMT draws 8 samples per problem, against AIME's 32 in test.

Dev also leaves out four tracker Track rows. GPQA is read in neither variant: at 32,768
output tokens about 60% of a reasoning model's generations stop mid-thought, and those
truncated samples score at chance, so the entry measures the cutoff more than the model
(three Olmo 3.5 Tiny SFT checkpoints, temperature 1.0). MMLU:cot guards knowledge
instead. HELMET's recall tasks repeat RULER+, and its QA tasks are chat
requests capped at upstream's 10 tokens, so a model that reasons first is scored on where
the cap cut it. CTC's 15 tasks at 64K report heterogeneous metrics with no aggregate, and
its own README says a reasoning model needs more than their direct-answer budgets (64
tokens for retrieval). Long context is read from RULER+ at 64K, 128K and 256K.

Small dev also reads the tracker's agentic Track rows, which are external evals: a suite
cannot hold them, so each runs as its own ``olmo-eval run-external`` job beside the suite.
They are ``openthoughts_tblite`` (OpenThoughts-TBLite, 100 tasks) and ``terminal_bench_2``
(Terminal-Bench 2.1, 89 tasks) for the Terminal-Bench row, and ``swe_bench_verified``.
Tiny leaves them out: the tracker deprioritizes them there.

Left out: tracker rows with no olmo-eval task yet (CritPt, FrontierCode, BrowseComp-Plus,
AutomationBench, DeepSearchQA, Terminal-Bench-Science, FIRM, OpenAgentSafety, the
decontaminated paper-MATH), the test rows that are external evals (tau2-bench,
AA-Briefcase, DeepScholar Bench), and the multimodal rows, which keep their own suites.
Also left out until olmo-eval has a configuration that lets a reasoning model
finish (their variants cap generation at 1,024 or 128 tokens, most with stop sequences):
MATH-500, HumanEval+, MBPP+ and LongBench v2.

Sampled entries run at the ledger's setting: Olmo 3.5 Tiny's suggested temperature 1.0 and
top_p 0.95 (Small uses them too, assumed from Tiny, not measured), with 32,768 output
tokens. They are ``:..._t10_32k`` variants that keep each task's sample count. olmo-eval's
own default temperature for most of them is 0.6, and most set no ``max_tokens``, so their
budget would otherwise be whatever context they are served at. Do not override sampling or
``max_tokens`` for the whole run. RULER+ runs greedy in completion mode with direct-answer
budgets, and BFCL keeps its own settings (temperature 0.001, 4,096 tokens), as the ledger's
BFCL reads do. Safety tasks use
the thinking-model judge variants and need the judge providers shown in ``safety.py``
(``wg_judge`` in every suite, ``sr_judge`` in test). Each local judge takes its own GPU, so
dev needs at least 2 GPUs and test 3. Omniscience is graded by its default judge,
gpt-4o-mini, through the OpenAI API.

Tiny dev needs no API key. Every other suite needs ``OPENAI_API_KEY`` for its API-judged
tasks (MT-Bench and MultiChallenge in Small dev).
"""

# Defining modules, imported so their suites exist whatever the discovery order.
from olmo_eval.evals.suites import (  # noqa: F401
    arena_hard,
    bfcl,
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

# The ledger's sampling: Olmo 3.5 Tiny's suggested temperature and top_p (assumed for Small),
# and a fixed budget, so scores do not depend on the served context length.
MODEL_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}
SAMPLED_VARIANT = "t10_32k"
# Entries that keep their own sampling: greedy direct-answer RULER+, and BFCL.
OWN_SAMPLING = frozenset(
    {
        "ruler_plus_all__65536",
        "ruler_plus_all__131072:hillclimb",
        "ruler_plus_all__262144:hillclimb",
        "bfcl",
    }
)


# Each ``_t10_32k`` spec in these suites, mapped to the spec it was made from.
SAMPLED_FROM: dict[str, str] = {}


def _at_model_sampling(entry: str | Suite) -> str | Suite:
    """The entry at ``MODEL_SAMPLING`` (``:..._t10_32k``), unless it keeps its own sampling."""
    if isinstance(entry, Suite):
        if entry.name in OWN_SAMPLING:
            return entry
        name = f"{entry.name}:{SAMPLED_VARIANT}"
        if suite_exists(name):
            return get_suite(name)
        return make_suite(
            name,
            tuple(_at_model_sampling(task) for task in entry.tasks),
            aggregation=entry.aggregation,
            description=(
                f"{entry.description} At temperature 1.0, top_p 0.95 and 32,768 tokens."
            ).strip(),
        )
    if entry in OWN_SAMPLING:
        return entry
    spec = register_sampling_variant(entry, SAMPLED_VARIANT, **MODEL_SAMPLING)
    SAMPLED_FROM[spec] = entry
    return spec


def _sampled(entries: tuple) -> tuple:
    return tuple(_at_model_sampling(entry) for entry in entries)


_HMMT = ("hmmt_feb_2025:pass_at_8", "hmmt_nov_2025:pass_at_8", "hmmt_feb_2026:pass_at_8")
_AIME = ("aime_2025:pass_at_32", "aime_2026:pass_at_32")


_DEV_SHARED = (
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
    "harmbench:wg_judge_thinking",
    "xstest:wg_judge_thinking",
    "wildguardtest:wg_judge_thinking",
)


_TEST_SHARED = (
    get_suite("mmlu_pro:cot"),
    "omniscience:judge",
    "zebralogic:chat",
    *_AIME,
    "livecodebench_hidden",
    "do_anything_now:wg_judge_thinking",
    "wildjailbreak:wg_judge_thinking",
    "wmdp:mcq",
    "bbq:mcq",
    "strongreject:sr_judge_thinking",
)

make_suite(
    "posttrain:tiny:dev",
    _sampled(_DEV_SHARED),
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
            *_DEV_SHARED,
            "mt_bench",
            "multi_if",
            "multichallenge",
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

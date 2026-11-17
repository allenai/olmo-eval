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

The tracker deprioritizes those rows for Tiny, but without an agentic read Tiny dev cannot
show whether a change helps or hurts agentic skill, so it reads two on which Tiny scores
above zero. BFCL multi-turn is ``posttrain:tiny:dev:bfcl_multi_turn``, a suite of its own
because it needs the ``bfcl_multi_turn`` scaffold (``--harness bfcl_multi_turn``), at the
ledger's setting for it: temperature 1.0, top_p 0.95 and 16,384 tokens per turn. SWE-bench
Verified reads only the 194 of its 500 instances annotated as under 15 minutes to fix
(``-a difficulties='<15 min fix'``), which hold most of what a Tiny checkpoint resolves (17
of 26 for one Tiny SFT checkpoint). It runs as its own job at the ledger's setting:
``-a task_frame=swesmith -a temperature=1.0 -a top_p=0.95 -a max_tokens=16384`` and
``-K max_model_len=65536``, with the vanillux scaffold's 64 turns.

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
(``wg_judge`` in every suite but the multi-turn and chat RULER+ ones, ``sr_judge`` in
test). Each local judge takes its own GPU, so
dev needs at least 2 GPUs and test 3. Omniscience is graded by its default judge,
gpt-4o-mini, through the OpenAI API.

Small dev is two suites, run as separate jobs, and so is Tiny dev (``posttrain:tiny:dev`` and
``posttrain:tiny:dev:bfcl_multi_turn``). MT-Bench and Multi-IF need the
``multi_turn_chat`` scaffold (``--harness multi_turn_chat``), which drops tool calls and has
no sandbox, so they are ``posttrain:small:dev:multi_turn``; the rest is
``posttrain:small:dev``.

Each dev suite also has a chat RULER+ suite, ``posttrain:{tiny,small}:dev:chat_ruler``:
RULER+ at 64K on the 128-item sample, sent as one chat turn, greedy, with 512 output tokens.
Completion RULER+ understates what post-training costs in the format the model is served
in: on the original RULER at 64K, Olmo 3.5 SFT checkpoints scored 4 to 12 points lower in
chat than in completion on the same items. Run it as its own job with thinking off, by
serving a think-off chat template (``-o provider.tokenizer=<dir>``). With thinking on, 512
tokens cut the reasoning off, and even at 8,192 tokens 70% of answers hit the cap.

Tiny dev needs no API key. Every other suite needs ``OPENAI_API_KEY`` for its API-judged
tasks (MultiChallenge in Small dev, MT-Bench in its multi-turn suite).
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
        "ruler_plus_all__65536:hillclimb:chat_mt512",
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
    _sampled((*_DEV_SHARED, "multichallenge")),
    aggregation=AggregationStrategy.NONE,
    description="Olmo 3.5 Small post-training dev set (the tracker's Track column).",
)

# BFCL multi-turn, Tiny dev's agentic read: it needs the bfcl_multi_turn scaffold, so it is
# a suite of its own. It runs at the ledger's setting for these tasks (temperature 1.0, top_p
# 0.95, 16,384 tokens per model turn). At the tasks' own 4,096 tokens, nearly half of a Tiny
# SFT checkpoint's rollouts hit the cap.
BFCL_MULTI_TURN_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 16384}
BFCL_MULTI_TURN_VARIANT = "t10_16k"
_bfcl_multi_turn = get_suite("bfcl:multi_turn")
make_suite(
    "posttrain:tiny:dev:bfcl_multi_turn",
    (
        make_suite(
            f"bfcl:multi_turn:{BFCL_MULTI_TURN_VARIANT}",
            tuple(
                register_sampling_variant(task, BFCL_MULTI_TURN_VARIANT, **BFCL_MULTI_TURN_SAMPLING)
                for task in _bfcl_multi_turn.expand()
            ),
            aggregation=_bfcl_multi_turn.aggregation,
            description=(
                f"{_bfcl_multi_turn.description} At temperature 1.0, top_p 0.95 and "
                "16,384 tokens per turn."
            ),
        ),
    ),
    aggregation=AggregationStrategy.NONE,
    description=(
        "Olmo 3.5 Tiny post-training dev set, BFCL multi-turn under --harness bfcl_multi_turn."
    ),
)

# The rest of Small dev, which needs the multi_turn_chat scaffold. That scaffold drops tool
# calls and has no sandbox, so BFCL and LiveCodeBench cannot share its job.
make_suite(
    "posttrain:small:dev:multi_turn",
    _sampled(("mt_bench", "multi_if")),
    aggregation=AggregationStrategy.NONE,
    description=(
        "Olmo 3.5 Small post-training dev set, the tasks that run under --harness multi_turn_chat."
    ),
)

# Chat RULER+, run as its own job with thinking off (see the module docstring).
for _size in ("tiny", "small"):
    make_suite(
        f"posttrain:{_size}:dev:chat_ruler",
        _sampled((get_suite("ruler_plus_all__65536:hillclimb:chat_mt512"),)),
        aggregation=AggregationStrategy.NONE,
        description=(
            f"Olmo 3.5 {_size.capitalize()} post-training dev set, RULER+ 64K in chat format; "
            "serve with thinking off."
        ),
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

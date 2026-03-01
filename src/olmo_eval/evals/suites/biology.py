"""Biology evaluation suite."""

from olmo_eval.evals.suites.registry import make_suite

# =============================================================================
# LAB-Bench Suite
# =============================================================================

_LAB_BENCH_TASKS = (
    "lab_bench_litqa2",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "lab_bench_protocolqa",
    "lab_bench_suppqa",
    "lab_bench_cloning_scenarios",
)

LAB_BENCH = make_suite(
    "lab_bench",
    _LAB_BENCH_TASKS,
    description="LAB-Bench biology research benchmark (futurehouse/lab-bench)",
)

LAB_BENCH_MC = make_suite(
    "lab_bench:mc",
    tuple(f"{t}:mc" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with logprob-based MC scoring",
)

LAB_BENCH_BPB = make_suite(
    "lab_bench:bpb",
    tuple(f"{t}:bpb" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with bits-per-byte evaluation",
)

make_suite(
    "medmcqa",
    ("medmcqa:mc", "medmcqa:rc"),
)

make_suite(
    "medmcqa:mc",
    ("medmcqa:mc",),
)

make_suite(
    "medmcqa:rc",
    ("medmcqa:rc",),
)

make_suite(
    "medqa_en",
    ("medqa_en:rc", "medqa_en:mc"),
)

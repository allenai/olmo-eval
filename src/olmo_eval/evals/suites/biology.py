"""Biology evaluation suite."""

from olmo_eval.evals.suites.registry import make_suite

# =============================================================================
# LAB-Bench Suite
# =============================================================================

LAB_BENCH = make_suite(
    "lab_bench",
    ("lab_bench_litqa2",),
    description="LAB-Bench biology research benchmark (futurehouse/lab-bench)",
)

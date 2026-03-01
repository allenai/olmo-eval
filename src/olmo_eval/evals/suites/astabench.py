"""AstaBench evaluation suite."""

from olmo_eval.evals.suites.registry import make_suite

ASTABENCH = make_suite(
    "astabench",
    ("astabench_scholarqa",),
    description="AstaBench scientific evaluation suite (allenai/asta-bench)",
)

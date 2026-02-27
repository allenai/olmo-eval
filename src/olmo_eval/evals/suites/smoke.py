"""Smoke test suites for basic model sanity checks."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# =============================================================================
# Smoke Test Suites
# =============================================================================


OLMO_INSTRUCT_SMOKE = make_suite(
    name="olmo:instruct:smoke",
    tasks=("smoke_hello", "smoke_identity_olmo", "smoke_toolcall"),
    aggregation=AggregationStrategy.NONE,
    description="Smoke tests for Olmo3 instruct models",
)

OLMO_THINK_SMOKE = make_suite(
    name="olmo:think:smoke",
    tasks=("smoke_hello", "smoke_identity_olmo", "smoke_reasoning"),
    aggregation=AggregationStrategy.NONE,
    description="Smoke tests for Olmo3 think models",
)

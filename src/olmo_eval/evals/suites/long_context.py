"""Long-context benchmark suites (HELMET, RULER)."""

from olmo_eval.evals.constants.long_context import HELMET_SUITES, RULER_SUITES
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite

# =============================================================================
# HELMET Benchmark Suites
# =============================================================================


def _build_helmet_suite(length: int) -> Suite:
    """Build HELMET suite for a specific context length."""
    tasks = tuple(
        task
        for suite_name, suite_tasks in HELMET_SUITES.items()
        for task in suite_tasks
        if suite_name.endswith(f"__{length}::suite") and not suite_name.startswith("helmet_all")
    )
    return make_suite(
        f"helmet:{length // 1024}k",
        tasks,
        description=f"HELMET long-context benchmark at {length // 1024}k tokens",
    )


# Register HELMET suites for 8k, 16k, 32k, 64k, 128k
HELMET_SUITES_BY_LENGTH: dict[int, Suite] = {}
for _length in (8192, 16384, 32768, 65536, 131072):
    HELMET_SUITES_BY_LENGTH[_length] = _build_helmet_suite(_length)


# =============================================================================
# RULER Benchmark Suites
# =============================================================================


def _build_ruler_suite(length: int) -> Suite:
    """Build RULER suite for a specific context length."""
    tasks = tuple(
        task
        for suite_name, suite_tasks in RULER_SUITES.items()
        for task in suite_tasks
        if suite_name.endswith(f"__{length}::suite") and not suite_name.startswith("ruler_all")
    )
    return make_suite(
        f"ruler:{length // 1024}k",
        tasks,
        description=f"RULER long-context benchmark at {length // 1024}k tokens",
    )


# Register RULER suites for 4k, 8k, 16k, 32k, 64k, 128k
RULER_SUITES_BY_LENGTH: dict[int, Suite] = {}
for _length in (4096, 8192, 16384, 32768, 65536, 131072):
    RULER_SUITES_BY_LENGTH[_length] = _build_ruler_suite(_length)


# RULER 4k-64k suite for OLMo3
OLMO3_RULER_V1 = make_suite(
    "olmo3:ruler:v1",
    tuple(RULER_SUITES_BY_LENGTH[2**i] for i in range(12, 17)),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="RULER 4k-64k suite for OLMo3",
)

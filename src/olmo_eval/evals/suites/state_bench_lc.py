from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.state_bench_lc import (
    STATE_BENCH_10PCT_TASKS,
    STATE_BENCH_CONFIGS,
    STATE_BENCH_STRATA_BY_CONFIG,
    STATE_BENCH_TASKS,
    STATE_BENCH_TOKEN_STRATA,
    state_bench_10pct_task_name,
    state_bench_task_name,
)

make_suite(
    name="state_bench",
    tasks=STATE_BENCH_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Long-context StateBench tasks across all formats and complexity classes.",
)

make_suite(
    name="state_bench_10pct",
    tasks=STATE_BENCH_10PCT_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Deterministic 10% sample of the long-context StateBench tasks.",
)

for _complexity in ("aperiodic", "periodic", "r-trivial"):
    make_suite(
        name=f"state_bench:{_complexity.replace('-', '_')}",
        tasks=tuple(
            state_bench_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.endswith(f"--{_complexity}")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks in the {_complexity} complexity class.",
    )
    make_suite(
        name=f"state_bench_10pct:{_complexity.replace('-', '_')}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.endswith(f"--{_complexity}")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Deterministic 10% StateBench sample in the {_complexity} complexity class.",
    )

for _formatter in (
    "cube-painting",
    "integer-code",
    "people-in-rooms",
    "ruler",
    "spreadsheet-cells",
    "status-lights",
):
    make_suite(
        name=f"state_bench:{_formatter.replace('-', '_')}",
        tasks=tuple(
            state_bench_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.startswith(f"{_formatter}--")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks using the {_formatter} format.",
    )
    make_suite(
        name=f"state_bench_10pct:{_formatter.replace('-', '_')}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if config_name.startswith(f"{_formatter}--")
            for token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Deterministic 10% StateBench sample using the {_formatter} format.",
    )
    for _token_stratum in STATE_BENCH_TOKEN_STRATA:
        make_suite(
            name=(f"state_bench:{_formatter.replace('-', '_')}:{_token_stratum}"),
            tasks=tuple(
                state_bench_task_name(config_name, _token_stratum)
                for config_name in STATE_BENCH_CONFIGS
                if config_name.startswith(f"{_formatter}--")
                and _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
            ),
            aggregation=AggregationStrategy.AVERAGE,
            description=(
                f"Long-context StateBench {_formatter} tasks in the {_token_stratum} context tier."
            ),
        )
        make_suite(
            name=(f"state_bench_10pct:{_formatter.replace('-', '_')}:{_token_stratum}"),
            tasks=tuple(
                state_bench_10pct_task_name(config_name, _token_stratum)
                for config_name in STATE_BENCH_CONFIGS
                if config_name.startswith(f"{_formatter}--")
                and _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
            ),
            aggregation=AggregationStrategy.AVERAGE,
            description=(
                f"Deterministic 10% StateBench {_formatter} sample in the "
                f"{_token_stratum} context tier."
            ),
        )

for _token_stratum in STATE_BENCH_TOKEN_STRATA:
    make_suite(
        name=f"state_bench:{_token_stratum}",
        tasks=tuple(
            state_bench_task_name(config_name, _token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Long-context StateBench tasks in the {_token_stratum} context tier.",
    )
    make_suite(
        name=f"state_bench_10pct:{_token_stratum}",
        tasks=tuple(
            state_bench_10pct_task_name(config_name, _token_stratum)
            for config_name in STATE_BENCH_CONFIGS
            if _token_stratum in STATE_BENCH_STRATA_BY_CONFIG[config_name]
        ),
        aggregation=AggregationStrategy.AVERAGE,
        description=f"Deterministic 10% StateBench sample in the {_token_stratum} context tier.",
    )

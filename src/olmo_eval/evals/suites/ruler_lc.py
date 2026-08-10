"""RULER long-context (ruler-lc) task suites organized by category and context size.

Mirrors suites/ruler.py's structure (per-category suites plus a combined
"all" suite per context size, matching the paper's "average of all 13 tasks"
methodology) for the ruler-lc task family.
"""

from olmo_eval.data.ruler_lc_tasks import CONTEXT_SIZES as RULER_LC_CONTEXT_SIZES
from olmo_eval.data.ruler_lc_tasks import RULER_LC_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Task categories (tags)
CATEGORIES = ["niah", "multi_hop_tracing", "aggregation", "qa"]


# Create suites for each (category, context_size) combination
for size in RULER_LC_CONTEXT_SIZES:
    all_tasks: list[str] = []

    for category in CATEGORIES:
        # Find all tasks with this category and context size
        tasks = [
            f"ruler_lc_{task_name}"
            for task_name, task_config in RULER_LC_TASKS.items()
            if task_config["tag"] == category and task_name.endswith(f"__{size}")
        ]

        if len(tasks) == 0:
            continue

        # Register category-specific suite: ruler_lc_niah__4096
        suite = Suite(
            name=f"ruler_lc_{category}__{size}",
            tasks=tuple(tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"RULER long-context {category} tasks at {size} context length",
        )
        register(suite)
        all_tasks.extend(tasks)

    # Create combined suite: flat average of all 13 tasks, matching the paper
    if len(all_tasks) > 0:
        all_tasks_suite = Suite(
            name=f"ruler_lc_all__{size}",
            tasks=tuple(all_tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=(
                f"All RULER long-context tasks at {size} context length (flat average of all tasks)"
            ),
        )
        register(all_tasks_suite)

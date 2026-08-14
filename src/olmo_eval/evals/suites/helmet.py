"""HELMET-plus task suites organized by category and context size.

Mirrors the suite structure in `suites/ruler.py`:
- Per-category suites for each size: helmet_recall__262144, etc.
- Combined suites for each size: helmet_all__262144, etc.
"""

from olmo_eval.data.helmet_tasks import CONTEXT_SIZES, HELMET_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Task categories (tags)
CATEGORIES = ["recall"]


# Create suites for each (category, context_size) combination
for size in CONTEXT_SIZES:
    all_tasks: list[str] = []

    for category in CATEGORIES:
        tasks = [
            f"helmet_{task_name}"
            for task_name, task_config in HELMET_TASKS.items()
            if task_config["tag"] == category and task_name.endswith(f"__{size}")
        ]

        if len(tasks) == 0:
            continue

        suite = Suite(
            name=f"helmet_{category}__{size}",
            tasks=tuple(tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"HELMET-plus {category} tasks at {size} context length",
        )
        register(suite)
        all_tasks.extend(tasks)

    if len(all_tasks) > 0:
        all_tasks_suite = Suite(
            name=f"helmet_all__{size}",
            tasks=tuple(all_tasks),
            aggregation=AggregationStrategy.AVERAGE,
            description=f"All HELMET-plus tasks at {size} context length (flat average)",
        )
        register(all_tasks_suite)

"""HELMET-plus task suites organized by category and context size.

Mirrors the suite structure in `suites/ruler.py`:
- Per-category suites for each size: helmet_recall__262144, helmet_icl__4096, etc.
- Combined suites for each size: helmet_all__4096, etc.

Two deliberate differences from `suites/ruler.py`:

1. `helmet_all__*` aggregates the *category* suites rather than flat-averaging
   every task, matching how the HELMET paper reports an overall number. A flat
   average would let a category's task count set its weight -- ICL contributes
   five tasks and recall one, so recall would be worth a sixth of ICL.

2. `helmet_all__*` is only registered where more than one category exists at
   that length. Above 128k only `json_kv` extends, so a combined suite there
   would be a rename of `helmet_recall__*` whose composition silently differs
   from the same suite at shorter lengths -- exactly the thing that turns a
   length sweep into a misleading trend line. Use `helmet_recall__*` for the
   extended tiers.
"""

from olmo_eval.data.helmet_tasks import CONTEXT_SIZES, HELMET_TASKS
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

# Task categories (tags), in the order HELMET reports them
CATEGORIES = ["recall", "icl"]


# Create suites for each (category, context_size) combination
for size in CONTEXT_SIZES:
    category_suites: list[Suite] = []

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
        category_suites.append(suite)

    if len(category_suites) > 1:
        all_tasks_suite = Suite(
            name=f"helmet_all__{size}",
            tasks=tuple(category_suites),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
            description=(
                f"All HELMET-plus tasks at {size} context length "
                f"(mean over category averages: {', '.join(CATEGORIES)})"
            ),
        )
        register(all_tasks_suite)

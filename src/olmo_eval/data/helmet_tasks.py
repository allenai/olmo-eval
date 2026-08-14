"""HELMET-plus task configurations.

This module defines the long-context (up to 2m tokens) HELMET task variants
published as the `allenai/helmet-plus` dataset. Only synthetic subsets can be
extended this far past standard HELMET's 128k ceiling, since real-document
tasks (LongQA, Summ, RAG, ...) are capped by how long the source documents
actually are; currently that's just the `json_kv` recall task.

Tasks are generated programmatically from base configurations, following the
same pattern as `ruler_tasks.py`.
"""

# Length tiers published in helmet-plus, mapping the manifest key to its
# resolved token length (see HELMET's scripts/generate_json_kv_data.py).
LENGTH_NAMES = {
    262144: "256k",
    524288: "512k",
    1048576: "1m",
    2097152: "2m",
}

CONTEXT_SIZES = list(LENGTH_NAMES)

_DEFAULT_MAX_GEN_TOKS = 100
_DEFAULT_SHOTS = 2

# Base task configurations, keyed by HELMET task type.
_BASE_TASKS = {
    "json_kv": {
        "tag": "recall",
    },
}


def _generate_helmet_tasks() -> dict:
    """Generate HELMET_TASKS dictionary from base configurations.

    Creates task definitions for all combinations of base tasks and context sizes.

    Returns:
        Dictionary mapping task names to their configurations.
    """
    tasks = {}

    for task_type, base_config in _BASE_TASKS.items():
        for size in CONTEXT_SIZES:
            task_name = f"{task_type}__{size}"
            tasks[task_name] = {
                "length_name": LENGTH_NAMES[size],
                "shots": base_config.get("shots", _DEFAULT_SHOTS),
                "max_gen_toks": base_config.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS),
                "tag": base_config["tag"],
            }

    return tasks


# Generate the full HELMET_TASKS dictionary
HELMET_TASKS = _generate_helmet_tasks()

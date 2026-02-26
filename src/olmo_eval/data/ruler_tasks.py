"""RULER task configurations.

This module defines all RULER task variants across different context sizes.
Tasks are generated programmatically from base configurations.
"""

# Context sizes to generate tasks for
CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072]

# Default configuration values
_DEFAULT_MAX_GEN_TOKS = 50

# Base task configurations
# Each entry defines a task type with its default settings
# Only specify max_gen_toks if different from default (50)
_BASE_TASKS = {
    # NIAH (Needle in a Haystack) - Single variants
    "niah_s_1": {
        "data_template": "data/ruler/niah_single_1/validation_{size}.jsonl",
        "tag": "niah",
    },
    "niah_s_2": {
        "data_template": "data/ruler/niah_single_2/validation_{size}.jsonl",
        "tag": "niah",
    },
    "niah_s_3": {
        "data_template": "data/ruler/niah_single_3/validation_{size}.jsonl",
        "tag": "niah",
    },
    # NIAH - Multi-key variants
    "niah_mk_1": {
        "data_template": "data/ruler/niah_multikey_1/validation_{size}.jsonl",
        "tag": "niah",
    },
    "niah_mk_2": {
        "data_template": "data/ruler/niah_multikey_2/validation_{size}.jsonl",
        "tag": "niah",
    },
    "niah_mk_3": {
        "data_template": "data/ruler/niah_multikey_3/validation_{size}.jsonl",
        "max_gen_toks": 100,  # Non-default
        "tag": "niah",
    },
    # NIAH - Multi-value variant
    "niah_mv": {
        "data_template": "data/ruler/niah_multivalue/validation_{size}.jsonl",
        "max_gen_toks": {4096: 300, 8192: 250, 16384: 250, 32768: 250, 65536: 250, 131072: 250},
        "tag": "niah",
    },
    # NIAH - Multi-query variant
    "niah_mq": {
        "data_template": "data/ruler/niah_multiquery/validation_{size}.jsonl",
        "max_gen_toks": 100,  # Non-default
        "tag": "niah",
    },
    # Multi-hop tracing - Variable tracking
    "vt": {
        "data_template": "data/ruler/vt/validation_{size}.jsonl",
        "tag": "multi_hop_tracing",
    },
    # Aggregation - Common word extraction
    "cwe": {
        "data_template": "data/ruler/cwe/validation_{size}.jsonl",
        "max_gen_toks": 100,  # Non-default
        "tag": "aggregation",
    },
    # Aggregation - Frequency word extraction
    "fwe": {
        "data_template": "data/ruler/fwe/validation_{size}.jsonl",
        "tag": "aggregation",
    },
    # Question Answering
    "qa_1": {
        "data_template": "data/ruler/qa_1/validation_{size}.jsonl",
        "tag": "qa",
        "metrics": [
            "substring_exact_match",
            "exact_match",
            "f1",
            "rougeL_f1",
            "rougeL_recall",
            "rougeLsum_f1",
            "rougeLsum_recall",
        ],
    },
    "qa_2": {
        "data_template": "data/ruler/qa_2/validation_{size}.jsonl",
        "tag": "qa",
        "metrics": [
            "substring_exact_match",
            "exact_match",
            "f1",
            "rougeL_f1",
            "rougeL_recall",
            "rougeLsum_f1",
            "rougeLsum_recall",
        ],
    },
}


def _generate_ruler_tasks() -> dict:
    """Generate RULER_TASKS dictionary from base configurations.

    Creates task definitions for all combinations of base tasks and context sizes.

    Returns:
        Dictionary mapping task names to their configurations.
    """
    tasks = {}

    for task_type, base_config in _BASE_TASKS.items():
        for size in CONTEXT_SIZES:
            task_name = f"{task_type}__{size}"

            # Resolve max_gen_toks (default if not specified, or size-specific from dict)
            max_gen_toks = base_config.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS)
            if isinstance(max_gen_toks, dict):
                max_gen_toks = max_gen_toks[size]

            # Build task configuration
            tasks[task_name] = {
                "data": str(base_config["data_template"]).format(size=size),
                "examples": None,
                "num_shots": 0,
                "max_gen_toks": max_gen_toks,
                "use_chat_template": False,
                "stop_new_line": False,
                "tag": base_config["tag"],
                "metrics": base_config.get("metrics", ["recall"]),
            }

    return tasks


# Generate the full RULER_TASKS dictionary
RULER_TASKS = _generate_ruler_tasks()

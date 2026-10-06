"""HELMET-plus data loading utilities.

Loads HELMET (https://github.com/princeton-nlp/HELMET) data from the
ai2-internal `allenai/helmet-plus` dataset on the Hub, which re-hosts
HELMET's pre-generated and pre-retrieved files so consumers can fetch them
per task. The synthetic `json_kv` tiers are regenerated with HELMET's own
generator, calibrated against the Olmo 3 tokenizer.

Downloads are pinned to a fixed revision of the dataset so that results stay
reproducible as the dataset grows.
"""

import json
import logging
import os
from typing import Any

from olmo_eval.data.hub import download_dataset_file
from olmo_eval.data.jsonl import sample_jsonl_rows

logger = logging.getLogger(__name__)

HELMET_PLUS_REPO_ID = "allenai/helmet-plus"
HELMET_PLUS_REVISION = "f46550958fafdff9340d4c24c50a98aaccd5d202"


def download_helmet_plus_file(filename: str) -> str:
    """Download a single file from the helmet-plus dataset repo, using the HF cache.

    Args:
        filename: Path within the `allenai/helmet-plus` repo (e.g. "json_kv/manifest.json").

    Returns:
        Local path to the downloaded file.
    """
    return download_dataset_file(HELMET_PLUS_REPO_ID, filename, revision=HELMET_PLUS_REVISION)


def load_json_kv_manifest() -> dict[str, dict[str, Any]]:
    """Download and parse the json_kv long-context manifest.

    The manifest maps length names (e.g. "256k", "2m") to the resolved
    `num_kvs` and data file used to generate that length tier, so the
    exact file path never needs to be hardcoded here.
    """
    manifest_path = download_helmet_plus_file("json_kv/manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


# Matches HELMET's own load_json_kv prompt, from
# https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/prompts/kv_retrieval.prompt
_USER_TEMPLATE = (
    "{context}\n\n"
    "Extract the value corresponding to the specified key in the JSON object below.\n\n"
    "{demos}Key: {question}"
)
_SYSTEM_TEMPLATE = "Corresponding value:"
_DEMO_TEMPLATE = "Key: {key}\nCorresponding value: {value}"


def load_json_kv_dataset(
    length_name: str, shots: int = 2, max_samples: int | None = None, seed: int = 42
) -> dict[str, Any]:
    """Load a helmet-plus json_kv dataset for a specific long-context length tier.

    Args:
        length_name: Manifest key identifying the length tier (e.g. "256k", "2m").
        shots: Number of few-shot key/value demos to prepend to each example.
        max_samples: Maximum number of examples to load (for testing).
        seed: Random seed for sampling.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_json_kv_manifest()
    if length_name not in manifest:
        raise ValueError(
            f"Unknown helmet-plus json_kv length '{length_name}'. Available: {sorted(manifest)}"
        )

    remote_path = f"json_kv/{os.path.basename(manifest[length_name]['test_file'])}"
    data_path = download_helmet_plus_file(remote_path)

    def process_example(example: dict[str, Any]) -> dict[str, Any]:
        demos = example.get("demos", [])[:shots]
        demo_text = "\n\n".join(
            _DEMO_TEMPLATE.format(key=key, value=value) for key, value in demos
        ) + ("\n\n" if demos else "")
        return {**example, "demos": demo_text}

    data = [process_example(record) for record in sample_jsonl_rows(data_path, max_samples, seed)]

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }

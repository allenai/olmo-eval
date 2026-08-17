"""HELMET-plus data loading utilities.

Loads the ai2-internal long-context extension of HELMET
(https://github.com/princeton-nlp/HELMET), published as the
`allenai/helmet-plus` dataset on the Hub. Standard HELMET tops out at 128k
tokens; helmet-plus extends select synthetic subsets (currently the
`json_kv` recall task) up to 2m tokens by generating additional
calibrated-length examples rather than relying on real documents, which
can't be scaled arbitrarily.
"""

import json
import logging
import os
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars
from huggingface_hub.utils import silent_tqdm

logger = logging.getLogger(__name__)

HELMET_PLUS_REPO_ID = "allenai/helmet-plus"


def _disable_helmet_progress_bars() -> None:
    """Avoid HF tqdm `_lock` failures in the helmet-plus download path."""
    disable_hf_hub_progress_bars()


def download_helmet_plus_file(filename: str) -> str:
    """Download a single file from the helmet-plus dataset repo, using the HF cache.

    Args:
        filename: Path within the `allenai/helmet-plus` repo (e.g. "json_kv/manifest.json").

    Returns:
        Local path to the downloaded file.
    """
    _disable_helmet_progress_bars()
    return hf_hub_download(  # ty: ignore[no-matching-overload]
        repo_id=HELMET_PLUS_REPO_ID,
        filename=filename,
        repo_type="dataset",
        tqdm_class=silent_tqdm,
    )


def load_json_kv_manifest() -> dict[str, dict[str, Any]]:
    """Download and parse the json_kv long-context manifest.

    The manifest maps length names (e.g. "256k", "2m") to the resolved
    `num_kvs` and data file used to generate that length tier, so the
    exact file path never needs to be hardcoded here.
    """
    manifest_path = download_helmet_plus_file("json_kv/manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSONL in {path}:{line_num}: {err}") from err
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_num}, got {type(record).__name__}"
                )
            records.append(record)
    return records


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

    data = [process_example(record) for record in _load_jsonl(data_path)]

    if max_samples is not None:
        permutation = np.random.default_rng(seed).permutation(len(data))
        data = [data[int(idx)] for idx in permutation[: min(len(data), max_samples)]]

    return {
        "data": data,
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }

"""HuggingFace-backed data loading for the RULER-plus (ruler-plus) dataset.

Unlike ruler_loader.py's tarball-based RULER release, ruler-plus is generated
with https://github.com/jopetty/RULER (scripts/generate-data.sh) and published
to the allenai/ruler-plus HuggingFace dataset repo. Each (context_size, task)
condition is its own JSONL shard, some of which are many GB (contexts run up
to 2097152 tokens), so only the single shard a task actually needs is
downloaded (rather than the whole repo), and it is read with reservoir
sampling so peak memory scales with the requested sample count rather than
the shard's full size.
"""

import json
import logging
import os

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars

logger = logging.getLogger(__name__)

RULER_PLUS_REPO_ID = "allenai/ruler-plus"
RULER_PLUS_DATA_DIR_ENV_VAR = "OLMO_EVAL_RULER_PLUS_DATA_DIR"


def get_ruler_plus_data_file(relative_path: str) -> str:
    """Return the local path to a single ruler-plus validation shard.

    ``relative_path`` is relative to the dataset root, e.g.
    ``"4096/niah_single_1/validation.jsonl"`` (see ``data_template`` in
    ruler_plus_tasks.py). Downloads (and caches) just that one shard from the
    allenai/ruler-plus HuggingFace dataset repo; override
    OLMO_EVAL_RULER_PLUS_DATA_DIR to point at a local checkout instead.
    """
    override_dir = os.environ.get(RULER_PLUS_DATA_DIR_ENV_VAR)
    if override_dir:
        file_path = os.path.join(os.path.expanduser(override_dir), relative_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"RULER-plus data file not found: {file_path}. Check {RULER_PLUS_DATA_DIR_ENV_VAR}."
            )
        return file_path

    disable_hf_hub_progress_bars()
    filename = f"data/{relative_path}"
    logger.info(f"Fetching {filename} from {RULER_PLUS_REPO_ID}...")
    return hf_hub_download(repo_id=RULER_PLUS_REPO_ID, repo_type="dataset", filename=filename)


def load_ruler_plus_shard(
    data_path: str, max_samples: int | None = None, seed: int = 42
) -> list[dict]:
    """Load a ruler-plus JSONL shard, reservoir-sampling up to ``max_samples`` records.

    Streams the shard line-by-line using reservoir sampling (Algorithm R) rather
    than reading the full file into memory before subsampling, so peak memory
    stays proportional to ``max_samples`` instead of the shard's full size.
    """
    if max_samples is None:
        with open(data_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rng = np.random.default_rng(seed)
    reservoir: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        seen = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < max_samples:
                reservoir.append(json.loads(line))
            else:
                j = int(rng.integers(0, seen + 1))
                if j < max_samples:
                    reservoir[j] = json.loads(line)
            seen += 1

    return reservoir

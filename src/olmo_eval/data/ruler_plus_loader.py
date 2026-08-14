"""HuggingFace-backed data loading for the RULER-plus (ruler-plus) dataset.

Unlike ruler_loader.py's tarball-based RULER release, ruler-plus is generated
with https://github.com/jopetty/RULER (scripts/generate-data.sh) and published
to the allenai/ruler-plus HuggingFace dataset repo, from which it is downloaded
on demand. Record parsing/prompt-template lookup is otherwise unchanged, so
ruler_loader.load_ruler_dataset is reused as-is once the data root is resolved.
"""

import logging
import os

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars

logger = logging.getLogger(__name__)

RULER_PLUS_REPO_ID = "allenai/ruler-plus"
RULER_PLUS_DATA_DIR_ENV_VAR = "OLMO_EVAL_RULER_PLUS_DATA_DIR"


def get_ruler_plus_data_root() -> str:
    """Return the root directory containing the ruler-plus validation JSONL files.

    Downloads (and caches) the allenai/ruler-plus dataset repo from HuggingFace;
    override with OLMO_EVAL_RULER_PLUS_DATA_DIR to point at a local checkout instead.
    """
    override_dir = os.environ.get(RULER_PLUS_DATA_DIR_ENV_VAR)
    if override_dir:
        data_dir = os.path.expanduser(override_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"RULER-plus data directory not found: {data_dir}. "
                f"Check {RULER_PLUS_DATA_DIR_ENV_VAR}."
            )
        return data_dir

    disable_hf_hub_progress_bars()
    logger.info(f"Fetching RULER-plus data from {RULER_PLUS_REPO_ID}...")
    repo_dir = snapshot_download(repo_id=RULER_PLUS_REPO_ID, repo_type="dataset")
    # The repo nests the {size}/{task}/validation.jsonl files under a top-level
    # "data" directory (see data_template in ruler_plus_tasks.py).
    return os.path.join(repo_dir, "data")

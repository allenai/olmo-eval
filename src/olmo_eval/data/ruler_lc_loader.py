"""Local data loading for the RULER long-context (ruler-lc) dataset.

Unlike ruler_loader.py's HuggingFace-hosted RULER data, ruler-lc is generated
with https://github.com/jopetty/RULER (scripts/generate-data.sh) and read
directly from local disk (e.g. Weka) rather than downloaded on demand.
Record parsing/prompt-template lookup is otherwise unchanged, so
ruler_loader.load_ruler_dataset is reused as-is once the data root is resolved.
"""

import os

RULER_LC_DATA_DIR_ENV_VAR = "OLMO_EVAL_RULER_LC_DATA_DIR"

# Ai2 cluster path where jopetty/RULER's generate-data.sh was run with its
# default output_dir (see olmo-eval's RULER long-context integration notes).
_DEFAULT_RULER_LC_DATA_DIR = "/weka/oe-training-default/jacksonp/RULER/scripts/benchmark_root/data"


def get_ruler_lc_data_root() -> str:
    """Return the root directory containing the ruler-lc validation JSONL files.

    Defaults to the Ai2 cluster path documented above; override with
    OLMO_EVAL_RULER_LC_DATA_DIR to point at a different checkout or output_dir.
    """
    data_dir = os.path.expanduser(
        os.environ.get(RULER_LC_DATA_DIR_ENV_VAR, _DEFAULT_RULER_LC_DATA_DIR)
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"RULER long-context data directory not found: {data_dir}. "
            f"Set {RULER_LC_DATA_DIR_ENV_VAR} to override."
        )
    return data_dir

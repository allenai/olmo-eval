"""Downloading files from Hugging Face Hub dataset repos."""

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars, silent_tqdm


def download_dataset_file(repo_id: str, filename: str, *, revision: str) -> str:
    """Download one file from a Hub dataset repo, using the HF cache.

    The revision is required, so every caller pins the data it reads.

    Args:
        repo_id: Dataset repo, e.g. "allenai/helmet-plus".
        filename: Path within the repo.
        revision: Commit SHA to read the file at.

    Returns:
        Local path to the downloaded file.
    """
    # Progress bars here have hit tqdm `_lock` failures under the async runner.
    disable_progress_bars()
    return hf_hub_download(  # ty: ignore[no-matching-overload]
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        tqdm_class=silent_tqdm,
    )

"""Hugging Face cache helpers for inference providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from olmo_eval.common.logging import get_logger

logger = get_logger(__name__)


def _looks_like_remote_hf_model(path: str) -> bool:
    """Return True when path should be resolved through the Hugging Face Hub."""
    if not path:
        return False

    expanded = Path(path).expanduser()
    if expanded.exists():
        return False

    if path.startswith(("/", "./", "../", "~")):
        return False

    scheme = urlparse(path).scheme
    return not (scheme and scheme not in {"hf"})


def refresh_hf_cache(
    model_or_tokenizer: str | None,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    token: bool | str | None = None,
    force_download: bool = False,
) -> str | None:
    """Force-refresh a Hugging Face Hub snapshot in the local cache.

    Returns the local snapshot path when a refresh was attempted, otherwise None.
    Local paths and non-HF URI schemes are ignored so local checkpoints keep their
    normal loading behavior.
    """
    if not force_download or not model_or_tokenizer:
        return None

    repo_id = str(model_or_tokenizer)
    if repo_id.startswith("hf://"):
        repo_id = repo_id.removeprefix("hf://")

    if not _looks_like_remote_hf_model(repo_id):
        return None

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to force-refresh Hugging Face cache entries."
        ) from e

    logger.info(
        "Force-refreshing Hugging Face cache for %s%s",
        repo_id,
        f" at revision {revision}" if revision else "",
    )
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "force_download": True,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if token is not None:
        kwargs["token"] = token

    return snapshot_download(**kwargs)

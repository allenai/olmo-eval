"""HuggingFace Hub dataset backend."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource


class HuggingFaceBackend:
    """Load datasets from HuggingFace Hub.

    Supports all HuggingFace datasets accessible via the `datasets` library.
    The path can be in org/repo format or prefixed with hf://.

    Examples:
        >>> backend = HuggingFaceBackend()
        >>> source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from HuggingFace Hub.

        Args:
            source: The data source with HuggingFace dataset path.
            streaming: Whether to stream the dataset.

        Yields:
            Raw document dictionaries from the dataset.
        """
        import os

        from datasets import load_dataset

        # Remove hf:// prefix if present
        path = source.path.removeprefix("hf://")

        # Use HF_TOKEN for authentication if available
        token = os.getenv("HF_TOKEN")

        kwargs: dict[str, Any] = {}
        if source.data_files is not None:
            kwargs["data_files"] = source.data_files
        if source.revision is not None:
            kwargs["revision"] = source.revision

        try:
            dataset = load_dataset(
                path,
                name=source.subset,
                split=source.split,
                streaming=streaming,
                token=token,
                **kwargs,
            )
        except ValueError as exc:
            # The datasets library silently falls back to its cache module when
            # the Hub is unreachable.  If the cache has *some* configs but not
            # the one we need, the cache builder raises a confusing ValueError.
            # Retry with force-redownload so the library hits the Hub directly
            # instead of using the stale/partial cache.
            if "Couldn't find cache" not in str(exc):
                raise
            from datasets import DownloadMode

            dataset = load_dataset(
                path,
                name=source.subset,
                split=source.split,
                streaming=streaming,
                token=token,
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
                **kwargs,
            )

        yield from dataset

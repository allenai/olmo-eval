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
        from datasets import load_dataset

        # Remove hf:// prefix if present
        path = source.path.removeprefix("hf://")

        dataset = load_dataset(
            path,
            name=source.subset,
            split=source.split,
            streaming=streaming,
            trust_remote_code=True,
        )

        yield from dataset

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
        except (RuntimeError, ValueError) as exc:
            err = str(exc)
            is_script_error = "Dataset scripts are no longer supported" in err
            is_cache_error = "Couldn't find cache" in err
            if not is_script_error and not is_cache_error:
                raise
            # datasets v4+ rejects repos that contain a legacy loading script.
            # The RuntimeError is silently caught by the library's fallback
            # logic and replaced with a confusing cache-miss ValueError.
            # Work around both by loading the data files directly from the Hub.
            dataset = self._load_from_hub_files(path, source, streaming, token, **kwargs)

        yield from dataset

    @staticmethod
    def _load_from_hub_files(
        path: str,
        source: DataSource,
        streaming: bool,
        token: str | None,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset directly from Hub data files, bypassing the module factory."""
        from datasets import load_dataset
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo_files = api.list_repo_files(path, repo_type="dataset")

        # Find data files matching the subset name
        subset = source.subset or ""
        candidates = [
            f
            for f in repo_files
            if subset in f and f.rsplit(".", 1)[-1] in ("jsonl", "json", "parquet", "csv")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No data files matching subset '{subset}' in {path}. "
                f"This dataset has a legacy loading script that is no longer supported."
            )

        ext = candidates[0].rsplit(".", 1)[-1]
        module = "json" if ext in ("json", "jsonl") else ext
        data_urls = [f"hf://datasets/{path}/{f}" for f in candidates]

        return load_dataset(
            module,
            data_files={source.split: data_urls},
            split=source.split,
            streaming=streaming,
            token=token,
        )

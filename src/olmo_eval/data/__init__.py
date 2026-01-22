"""Data loading abstraction for evaluation datasets.

This module provides a unified interface for loading datasets from multiple sources:
- HuggingFace Hub
- Local files (JSONL, Parquet, CSV)
- Amazon S3
- Google Cloud Storage

Example usage:
    >>> from olmo_eval.data import DataSource, DataLoader
    >>>
    >>> # Load from HuggingFace
    >>> loader = DataLoader()
    >>> source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
    >>> for doc in loader.load(source):
    ...     print(doc)
    >>>
    >>> # Load from local file
    >>> source = DataSource(path="/path/to/data.jsonl")
    >>> for doc in loader.load(source):
    ...     print(doc)
    >>>
    >>> # Load from URI string
    >>> for doc in loader.load("s3://bucket/data.jsonl"):
    ...     print(doc)
"""

from olmo_eval.data.loader import DataLoader, get_default_loader, load_dataset
from olmo_eval.data.sources import DataSource, SourceType

__all__ = [
    "DataLoader",
    "DataSource",
    "SourceType",
    "get_default_loader",
    "load_dataset",
]

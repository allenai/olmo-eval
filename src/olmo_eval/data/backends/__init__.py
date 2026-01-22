"""Dataset loading backends for different source types."""

from olmo_eval.data.backends.base import DataBackend
from olmo_eval.data.backends.gcs import GCSBackend
from olmo_eval.data.backends.huggingface import HuggingFaceBackend
from olmo_eval.data.backends.local import LocalBackend
from olmo_eval.data.backends.s3 import S3Backend

__all__ = [
    "DataBackend",
    "GCSBackend",
    "HuggingFaceBackend",
    "LocalBackend",
    "S3Backend",
]

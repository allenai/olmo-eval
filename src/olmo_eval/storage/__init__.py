"""Storage backends for evaluation results."""

from olmo_eval.storage.base import (
    EvalResult,
    StorageBackend,
    TaskResult,
    convert_runner_results,
)

__all__ = [
    "EvalResult",
    "StorageBackend",
    "TaskResult",
    "convert_runner_results",
    "get_backend",
]


def get_backend(
    backend_type: str,
    **kwargs,
) -> StorageBackend:
    """Create a storage backend by type.

    Args:
        backend_type: The type of backend ("s3", "postgres").
        **kwargs: Backend-specific configuration options.

    Returns:
        A configured storage backend instance.

    Raises:
        ValueError: If backend_type is not recognized.
        ImportError: If required dependencies are not installed.
    """
    if backend_type == "s3":
        try:
            from olmo_eval.storage.s3 import S3Backend
        except ImportError as e:
            raise ImportError(
                "S3 backend requires boto3. Install with: pip install olmo-eval-internal[s3]"
            ) from e
        return S3Backend(**kwargs)
    elif backend_type == "postgres":
        try:
            from olmo_eval.storage.postgres import PostgresBackend
        except ImportError as e:
            raise ImportError(
                "Postgres backend requires psycopg. "
                "Install with: pip install olmo-eval-internal[postgres]"
            ) from e
        return PostgresBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported types: s3, postgres")

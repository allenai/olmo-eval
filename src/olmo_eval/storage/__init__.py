"""Storage backends for evaluation results."""

from olmo_eval.storage.base import (
    StorageBackend,
    compute_model_hash,
    convert_runner_results,
)

__all__ = [
    "StorageBackend",
    "compute_model_hash",
    "convert_runner_results",
    "get_backend",
]


def get_backend(
    backend_type: str,
    **kwargs,
):
    """Create a storage backend by type.

    Args:
        backend_type: The type of backend ("postgres").
        **kwargs: Backend-specific configuration options.

    Returns:
        A configured storage backend instance.
        - "postgres": PostgresBackend

    Raises:
        ValueError: If backend_type is not recognized.
        ImportError: If required dependencies are not installed.
    """
    if backend_type == "postgres":
        try:
            from olmo_eval.storage.backends.postgres import PostgresBackend
        except ImportError as e:
            raise ImportError(
                "Postgres backend requires psycopg (the PostgreSQL driver for SQLAlchemy). "
                "Install with: uv pip install 'olmo-eval[postgres]'"
            ) from e
        return PostgresBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported types: postgres")

"""Centralized logging configuration for olmo-eval."""

import logging
import os
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Package-wide logger
PACKAGE_LOGGER_NAME = "olmo_eval"

# Suppress noisy third-party library output BEFORE they are imported.
# These must be set at module load time to take effect.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure root logging for olmo-eval.

    Called once at CLI entry points (run.py, beaker/launch.py).
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level),
    )

    # Suppress noisy third-party loggers
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # Set environment variables for third-party libraries
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the olmo_eval namespace."""
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_worker_logging(worker_id: str) -> logging.Logger:
    """Configure logging for a worker subprocess.

    Called at the start of each worker process. Creates a logger
    with worker identification in the format string.

    Args:
        worker_id: Unique worker identifier (e.g., "OLMo-2-7B-w0")

    Returns:
        Configured logger for this worker
    """
    logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.worker.{worker_id}")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [{worker_id}] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def get_worker_id(model_name: str, worker_index: int) -> str:
    """Generate a short worker ID from model name and index.

    Examples:
        "allenai/OLMo-2-7B", 0 -> "OLMo-2-7B-w0"
        "meta-llama/Llama-3.1-8B", 1 -> "Llama-3.1-8B-w1"
    """
    # Extract last component of path
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    # Truncate if too long
    if len(short_name) > 20:
        short_name = short_name[:17] + "..."
    return f"{short_name}-w{worker_index}"

"""Centralized logging configuration for olmo-eval."""

import logging
import os
import re
import warnings
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Package-wide logger
PACKAGE_LOGGER_NAME = "olmo_eval"

# ANSI color codes for terminal output
_COLORS = (
    "\033[36m",  # Cyan
    "\033[33m",  # Yellow
    "\033[35m",  # Magenta
    "\033[32m",  # Green
    "\033[34m",  # Blue
    "\033[91m",  # Light Red
    "\033[96m",  # Light Cyan
    "\033[93m",  # Light Yellow
    "\033[95m",  # Light Magenta
    "\033[92m",  # Light Green
    "\033[94m",  # Light Blue
)
_RESET = "\033[0m"


def _get_color_for_owner(owner: str) -> str:
    """Get a consistent color for an owner string based on its hash."""
    return _COLORS[hash(owner) % len(_COLORS)]


# Pattern to match [owner] at the start of a message
_OWNER_PATTERN = re.compile(r"^(\[[\w.-]+\])")


def _color_owner_in_message(message: str) -> str:
    """Color any [owner] pattern at the start of a log message."""
    match = _OWNER_PATTERN.match(message)
    if match:
        owner_tag = match.group(1)
        owner = owner_tag[1:-1]  # Remove brackets
        color = _get_color_for_owner(owner)
        return f"{color}{owner_tag}{_RESET}{message[match.end() :]}"
    return message


class OwnerColoringFormatter(logging.Formatter):
    """Formatter that colors [owner] patterns in log messages."""

    def format(self, record: logging.LogRecord) -> str:
        # Color any [owner] pattern in the message
        record.msg = _color_owner_in_message(str(record.msg))
        return super().format(record)


# Suppress noisy third-party library output BEFORE they are imported.
# These must be set at module load time to take effect.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("LITELLM_LOG", "ERROR")


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit.

    In multiprocessing subprocesses, stdout/stderr may be line-buffered or
    fully buffered, causing logs to appear only after the process completes.
    This handler forces a flush after each log message to ensure real-time output.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def filter_warnings() -> None:
    """Suppress noisy deprecation warnings from third-party libraries."""
    # PyTorch deprecation warnings
    warnings.filterwarnings("ignore", message=r".*torch\.distributed\.\w+_base.*")
    warnings.filterwarnings("ignore", message=r".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*DTensor.*")
    warnings.filterwarnings("ignore", message=r".*TORCH_NCCL_AVOID_RECORD_STREAMS.*")
    warnings.filterwarnings("ignore", message=r".*weights_only=False.*")
    warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.custom_fwd.*")

    # Flash-attention warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"flash_attn.*")

    # Transformers/HuggingFace warnings
    warnings.filterwarnings("ignore", message=r".*resume_download.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*clean_up_tokenization_spaces.*")

    # Pydantic warnings
    warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*")


def _is_debug_mode() -> bool:
    """Check if debug mode is enabled via OLMO_EVAL_DEBUG environment variable."""
    return os.environ.get("OLMO_EVAL_DEBUG", "").lower() in ("1", "true")


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure root logging for olmo-eval.

    Called once at CLI entry points (run.py, beaker/launch.py).

    Set OLMO_EVAL_DEBUG=1 to bypass all log silencing and warning filters.
    """
    root_logger = logging.getLogger()

    # In debug mode, use DEBUG level and don't silence anything
    if _is_debug_mode():
        handler = logging.StreamHandler()
        handler.setFormatter(
            OwnerColoringFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        OwnerColoringFormatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level))

    # Suppress noisy third-party loggers
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # Allow swerex (sandbox) logs through at DEBUG level
    logging.getLogger("swerex").setLevel(logging.DEBUG)

    # Set environment variables for third-party libraries
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    # Filter noisy deprecation warnings
    filter_warnings()


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
        handler = FlushingStreamHandler()
        color = _get_color_for_owner(worker_id)
        colored_id = f"{color}[{worker_id}]{_RESET}"
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [%(levelname)s] {colored_id} %(message)s",
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

"""vLLM OpenAI-compatible server management for agent tasks.

This module provides utilities to start and manage a vLLM server that
exposes an OpenAI-compatible API, enabling agent tasks to use any
HuggingFace or local model via the OpenAI Agents SDK.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default timeout for server startup
DEFAULT_STARTUP_TIMEOUT = 300  # 5 minutes for large models
DEFAULT_HEALTH_CHECK_INTERVAL = 2  # seconds


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


ProgressCallback = Callable[[str], None]


def _wait_for_server(
    url: str,
    timeout: float = DEFAULT_STARTUP_TIMEOUT,
    interval: float = DEFAULT_HEALTH_CHECK_INTERVAL,
    process: subprocess.Popen | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, Exception | None, str | None]:
    """Wait for vLLM server to be ready.

    Args:
        url: Base URL of the server (e.g., "http://localhost:8000/v1")
        timeout: Maximum time to wait in seconds
        interval: Time between health checks in seconds
        process: Optional subprocess to monitor for early exit
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (success, last_error, process_output):
        - success: True if server is ready, False if timeout exceeded or process died
        - last_error: The last error encountered, or None
        - process_output: Captured output if process died early, or None
    """
    import urllib.error
    import urllib.request

    health_url = url.rstrip("/").replace("/v1", "") + "/health"
    models_url = url.rstrip("/") + "/models"

    start_time = time.time()
    last_error: Exception | None = None
    last_progress_time = 0.0
    progress_interval = 10.0  # Log progress every 10 seconds

    if progress_callback:
        progress_callback(f"Waiting for vLLM server (0/{int(timeout)}s)...")

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        # Log progress periodically
        if progress_callback and elapsed - last_progress_time >= progress_interval:
            progress_callback(f"Waiting for vLLM server ({int(elapsed)}/{int(timeout)}s)...")
            last_progress_time = elapsed

        # Check if the subprocess has died
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                # Process exited - read its output
                output = None
                if process.stdout:
                    with suppress(Exception):
                        output = process.stdout.read().decode("utf-8", errors="replace")
                logger.error(f"vLLM server process exited with code {exit_code}")
                return False, RuntimeError(f"Process exited with code {exit_code}"), output

        try:
            # Try health endpoint first
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    if progress_callback:
                        progress_callback(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        try:
            # Fallback to models endpoint
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    if progress_callback:
                        progress_callback(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        time.sleep(interval)

    logger.error(f"vLLM server failed to start within {timeout}s. Last error: {last_error}")
    return False, last_error, None


def _infer_tool_call_parser(model_name: str) -> str:
    """Infer the appropriate tool call parser based on model name.

    Args:
        model_name: Model name or path.

    Returns:
        Tool call parser name for vLLM.
    """
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "llama3_json"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        # Default for Qwen, OLMo, and other models
        return "hermes"


def _build_server_command(
    model_name: str,
    port: int,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    tokenizer: str | None = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str | None = None,
    enable_prefix_caching: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Build the vLLM server command.

    Args:
        model_name: HuggingFace model ID or local path
        port: Port to serve on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type for model weights
        tokenizer: Custom tokenizer (defaults to model_name)
        enable_auto_tool_choice: Enable automatic tool choice
        tool_call_parser: Parser for tool calls (auto-detected if not specified
            when enable_auto_tool_choice is True)
        enable_prefix_caching: Enable prefix caching for faster inference (default: True)
        chat_template_kwargs: Extra kwargs for chat template (e.g., {"enable_thinking": false})
        **kwargs: Additional vLLM server arguments

    Returns:
        Command list for subprocess
    """
    import json

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]

    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])

    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    # Tool calling support
    if enable_auto_tool_choice:
        cmd.append("--enable-auto-tool-choice")
        # Auto-detect parser if not specified
        parser = tool_call_parser or _infer_tool_call_parser(model_name)
        cmd.extend(["--tool-call-parser", parser])

    # Prefix caching (enabled by default for faster inference)
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    # Chat template kwargs (e.g., for Qwen3 enable_thinking)
    if chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", json.dumps(chat_template_kwargs)])

    # Disable tqdm loading bar by default, enable with --debug-provider
    if is_debug_provider():
        cmd.append("--use-tqdm-on-load")
    else:
        cmd.append("--no-use-tqdm-on-load")

    # Add any extra kwargs as CLI args
    for key, value in kwargs.items():
        if value is not None:
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            else:
                cmd.extend([arg_name, str(value)])

    return cmd


class VLLMServerProcess:
    """Manages a vLLM server subprocess."""

    def __init__(
        self,
        model_name: str,
        port: int | None = None,
        tensor_parallel_size: int = 1,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        log_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the server manager.

        Args:
            model_name: HuggingFace model ID or local path
            port: Port to serve on (auto-assigned if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            startup_timeout: Maximum time to wait for server startup
            log_dir: Directory to write server logs to (if set, logs are persisted)
            **kwargs: Additional arguments passed to vLLM server
        """
        self.model_name = model_name
        self.port = port or _find_free_port()
        self.tensor_parallel_size = tensor_parallel_size
        self.startup_timeout = startup_timeout
        self.log_dir = log_dir
        self.server_kwargs = kwargs
        self._process: subprocess.Popen | None = None
        self._log_file: Any | None = None
        self._started = False

    @property
    def base_url(self) -> str:
        """Get the base URL for the OpenAI-compatible API."""
        return f"http://localhost:{self.port}/v1"

    def start(self, progress_callback: ProgressCallback | None = None) -> str:
        """Start the vLLM server.

        Args:
            progress_callback: Optional callback for progress updates during startup.

        Returns:
            The base URL for the OpenAI-compatible API

        Raises:
            RuntimeError: If server fails to start
        """
        if self._started:
            return self.base_url

        cmd = _build_server_command(
            model_name=self.model_name,
            port=self.port,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.server_kwargs,
        )

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        if progress_callback:
            progress_callback(f"Starting vLLM server for {self.model_name}...")

        # Start the server process
        env = os.environ.copy()

        # Enable verbose vLLM logging when debugging
        if is_debug_provider():
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            logger.info("vLLM debug logging enabled (OLMO_EVAL_DEBUG_PROVIDER=1)")

        # Determine output handling:
        # 1. If log_dir is set, write to file for persistence
        # 2. If debugging, stream to stderr
        # 3. Otherwise capture for error reporting
        if self.log_dir:
            import pathlib

            log_path = pathlib.Path(self.log_dir) / "vllm_server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "w")  # noqa: SIM115
            logger.info(f"vLLM server logs will be written to {log_path}")
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        elif is_debug_provider():
            self._process = subprocess.Popen(
                cmd,
                stdout=None,  # Inherit stdout
                stderr=None,  # Inherit stderr
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        else:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

        # Register cleanup on exit
        atexit.register(self.stop)

        # Wait for server to be ready
        success, last_error, process_output = _wait_for_server(
            self.base_url,
            timeout=self.startup_timeout,
            process=self._process,
            progress_callback=progress_callback,
        )
        if not success:
            # Try to capture any remaining output before stopping
            if process_output is None and self._process and self._process.stdout:
                with suppress(Exception):
                    process_output = self._process.stdout.read().decode("utf-8", errors="replace")

            # Log the captured output for debugging
            if process_output:
                logger.error(f"vLLM server output:\n{process_output}")

            self.stop()

            # Build a detailed error message
            error_msg = (
                f"vLLM server failed to start for model {self.model_name} "
                f"within {self.startup_timeout}s"
            )
            if last_error:
                error_msg += f". Error: {last_error}"
            if process_output:
                # Include a truncated version of the output in the exception
                max_output_len = 2000
                if len(process_output) > max_output_len:
                    truncated_output = "...[truncated]...\n" + process_output[-max_output_len:]
                else:
                    truncated_output = process_output
                error_msg += f"\n\nServer output:\n{truncated_output}"

            raise RuntimeError(error_msg)

        self._started = True
        return self.base_url

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        logger.info("Stopping vLLM server...")

        try:
            # Try graceful shutdown first
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
                self._process.wait(timeout=5)

        except (ProcessLookupError, OSError):
            # Process already dead
            pass
        finally:
            self._process = None
            self._started = False
            atexit.unregister(self.stop)
            # Close log file if open
            if self._log_file is not None:
                with suppress(Exception):
                    self._log_file.close()
                self._log_file = None

        logger.info("vLLM server stopped")

    def __enter__(self) -> str:
        """Context manager entry - start server and return URL."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - stop server."""
        self.stop()


@contextmanager
def vllm_server_context(
    model_name: str,
    port: int | None = None,
    tensor_parallel_size: int = 1,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    progress_callback: ProgressCallback | None = None,
    log_dir: str | None = None,
    **kwargs: Any,
) -> Generator[str, None, None]:
    """Context manager to start and stop a vLLM server.

    This is the recommended way to use a vLLM server for agent tasks.
    The server is automatically started and stopped.

    Args:
        model_name: HuggingFace model ID or local path
        port: Port to serve on (auto-assigned if None)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        startup_timeout: Maximum time to wait for server startup
        progress_callback: Optional callback for progress updates during startup
        log_dir: Directory to write server logs to (if set, logs are persisted)
        **kwargs: Additional arguments passed to vLLM server

    Yields:
        The base URL for the OpenAI-compatible API (e.g., "http://localhost:8000/v1")

    Example:
        with vllm_server_context("meta-llama/Llama-3.1-8B-Instruct") as url:
            # url is "http://localhost:<port>/v1"
            client = AsyncOpenAI(base_url=url, api_key="EMPTY")
            ...
    """
    server = VLLMServerProcess(
        model_name=model_name,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        startup_timeout=startup_timeout,
        log_dir=log_dir,
        **kwargs,
    )

    try:
        yield server.start(progress_callback=progress_callback)
    finally:
        server.stop()

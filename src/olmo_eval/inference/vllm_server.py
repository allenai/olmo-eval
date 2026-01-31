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
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from olmo_eval.core.debug import is_debug_provider

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


def _wait_for_server(
    url: str,
    timeout: float = DEFAULT_STARTUP_TIMEOUT,
    interval: float = DEFAULT_HEALTH_CHECK_INTERVAL,
    process: subprocess.Popen | None = None,
) -> tuple[bool, Exception | None, str | None]:
    """Wait for vLLM server to be ready.

    Args:
        url: Base URL of the server (e.g., "http://localhost:8000/v1")
        timeout: Maximum time to wait in seconds
        interval: Time between health checks in seconds
        process: Optional subprocess to monitor for early exit

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

    while time.time() - start_time < timeout:
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
                    logger.info(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        try:
            # Fallback to models endpoint
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        time.sleep(interval)

    logger.error(f"vLLM server failed to start within {timeout}s. Last error: {last_error}")
    return False, last_error, None


def _build_server_command(
    model_name: str,
    port: int,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    tokenizer: str | None = None,
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
        **kwargs: Additional vLLM server arguments

    Returns:
        Command list for subprocess
    """
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
        **kwargs: Any,
    ) -> None:
        """Initialize the server manager.

        Args:
            model_name: HuggingFace model ID or local path
            port: Port to serve on (auto-assigned if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            startup_timeout: Maximum time to wait for server startup
            **kwargs: Additional arguments passed to vLLM server
        """
        self.model_name = model_name
        self.port = port or _find_free_port()
        self.tensor_parallel_size = tensor_parallel_size
        self.startup_timeout = startup_timeout
        self.server_kwargs = kwargs
        self._process: subprocess.Popen | None = None
        self._started = False

    @property
    def base_url(self) -> str:
        """Get the base URL for the OpenAI-compatible API."""
        return f"http://localhost:{self.port}/v1"

    def start(self) -> str:
        """Start the vLLM server.

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

        # Start the server process
        env = os.environ.copy()

        # Enable verbose vLLM logging when debugging
        if is_debug_provider():
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            logger.info("vLLM debug logging enabled (OLMO_EVAL_DEBUG_PROVIDER=1)")

        # When debugging, stream output to stderr; otherwise capture for error reporting
        if is_debug_provider():
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
            self.base_url, timeout=self.startup_timeout, process=self._process
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
        logger.info(f"vLLM server started at {self.base_url}")
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
        **kwargs,
    )

    try:
        yield server.start()
    finally:
        server.stop()

"""Tests for VLLMServerProcess startup and timeout handling."""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestVLLMServerStartupTimeout:
    """Tests for VLLMServerProcess startup timeout behavior."""

    @pytest.mark.anyio
    async def test_timeout_does_not_block_on_stdout(self):
        """Startup timeout terminates process before reading stdout to avoid blocking."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        # Create a mock process that never becomes healthy
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.stdout = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.wait.return_value = 0

        # Track if stop() is called before any stdout.read()
        call_order = []

        def track_killpg(*args):
            call_order.append("killpg")

        def track_read():
            call_order.append("read")
            # Simulate blocking read that would hang
            time.sleep(10)
            return b"output"

        mock_process.stdout.read.side_effect = track_read

        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(False, "Connection refused", None),
            ),
            patch("atexit.register"),
            patch("atexit.unregister"),
            patch("os.killpg", side_effect=track_killpg),
            patch("os.getpgid", return_value=12345),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                startup_timeout=1,  # Short timeout
            )

            with pytest.raises(RuntimeError, match="failed to start"):
                server.start()

            # Verify killpg was called (stop() uses killpg on Unix)
            # and stdout.read was never called (would have blocked)
            assert "killpg" in call_order
            assert "read" not in call_order

    @pytest.mark.anyio
    async def test_timeout_reads_log_file_after_stop(self, tmp_path):
        """After timeout, log file is read for error details."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0

        # Create a log file with error content
        log_file = log_dir / "vllm_server.log"
        log_content = "CUDA out of memory error"
        log_file.write_text(log_content)

        # Mock the log file handle
        mock_log_file = MagicMock()

        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(False, "Connection refused", None),
            ),
            patch("builtins.open", return_value=mock_log_file),
            patch("atexit.register"),
            patch("atexit.unregister"),
            patch("os.killpg"),
            patch("os.getpgid", return_value=12345),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                startup_timeout=1,
                log_dir=str(log_dir),
            )

            with pytest.raises(RuntimeError, match="failed to start") as exc_info:
                server.start()

            # Log content should be included in error message
            assert "CUDA out of memory" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_successful_startup_returns_url(self):
        """Successful startup returns the server URL."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ),
            patch("atexit.register"),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
            )

            url = server.start()
            assert url in ("http://localhost:8000/v1", "http://127.0.0.1:8000/v1")

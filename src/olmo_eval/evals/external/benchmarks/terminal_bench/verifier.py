"""Terminal-Bench task verification."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox.executor import SandboxExecutor

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of task verification.

    Attributes:
        reward: 0.0 for failure, 1.0 for success.
        test_output: Output from running the test script.
        test_exit_code: Exit code from the test script.
    """

    reward: float
    test_output: str
    test_exit_code: int


class TerminalBenchVerifier:
    """Verifies Terminal-Bench task completion."""

    async def inject_tests(
        self,
        executor: SandboxExecutor,
        test_files: dict[str, bytes],
    ) -> None:
        """Inject test files into the container.

        Creates /tests/ and /logs/verifier/ directories, then writes all
        test files to /tests/.

        Args:
            executor: The sandbox executor.
            test_files: Mapping of relative paths to file content.
        """
        # Create directories
        result = await executor.execute_command("mkdir -p /tests /logs/verifier", timeout=30.0)
        if not result.success:
            logger.warning(f"Failed to create test directories: {result.output}")

        # Inject each file
        for rel_path, content in test_files.items():
            # Ensure parent directory exists
            parent_dir = str(Path(rel_path).parent)
            if parent_dir != ".":
                await executor.execute_command(
                    f"mkdir -p /tests/{parent_dir}",
                    timeout=30.0,
                )

            # Write file via base64 (safe for both text and binary)
            b64 = base64.b64encode(content).decode()

            # Split large base64 strings into chunks to avoid command line limits
            if len(b64) > 50000:
                # Write in chunks for large files
                chunk_size = 50000
                for i in range(0, len(b64), chunk_size):
                    chunk = b64[i : i + chunk_size]
                    if i == 0:
                        # First chunk: create file
                        await executor.execute_command(
                            f"echo -n '{chunk}' > /tmp/_tb_chunk",
                            timeout=60.0,
                        )
                    else:
                        # Subsequent chunks: append
                        await executor.execute_command(
                            f"echo -n '{chunk}' >> /tmp/_tb_chunk",
                            timeout=60.0,
                        )
                # Decode the complete base64
                await executor.execute_command(
                    f"base64 -d /tmp/_tb_chunk > /tests/{rel_path} && rm /tmp/_tb_chunk",
                    timeout=60.0,
                )
            else:
                await executor.execute_command(
                    f"echo '{b64}' | base64 -d > /tests/{rel_path}",
                    timeout=60.0,
                )

        logger.info(f"Injected {len(test_files)} test files")

    async def run_verification(
        self,
        executor: SandboxExecutor,
        timeout: float,
        working_dir: str = "/app",
        task_id: str | None = None,
    ) -> VerificationResult:
        """Run verification tests.

        Executes /tests/test.sh and reads /logs/verifier/reward.txt for result.

        Args:
            executor: The sandbox executor.
            timeout: Timeout for test execution in seconds.
            working_dir: Working directory for running tests.
            task_id: Task identifier for log prefix.

        Returns:
            VerificationResult with reward, output, and exit code.
        """
        log_prefix = f"{task_id}_verifier" if task_id else "terminal_bench_verifier"

        # Run test.sh
        test_result = await executor.execute_command(
            f"cd {working_dir} && bash /tests/test.sh",
            timeout=timeout,
            stream=True,
            log_prefix=log_prefix,
        )

        logger.info(f"Test script exit code: {test_result.exit_code}")

        # Read reward
        reward_result = await executor.execute_command(
            "cat /logs/verifier/reward.txt",
            timeout=30.0,
        )

        reward = 0.0
        if reward_result.success:
            try:
                reward = float(reward_result.output.strip())
                logger.info(f"Reward: {reward}")
            except ValueError:
                logger.warning(f"Failed to parse reward: {reward_result.output[:100]}")
        else:
            logger.warning(f"Failed to read reward.txt: {reward_result.output[:100]}")

        return VerificationResult(
            reward=reward,
            test_output=test_result.output,
            test_exit_code=test_result.exit_code,
        )

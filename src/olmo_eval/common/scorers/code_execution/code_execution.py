"""Code execution scorer for evaluating generated code against test cases."""

from __future__ import annotations

import logging
import math
import shlex
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.types import Instance, LMOutput

from ..execution import ExecutionScorer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment, ExecutionResult


@dataclass(frozen=True, slots=True)
class CodeExecutionScorer(ExecutionScorer):
    """Score code by executing it against test cases in a sandbox.

    This scorer requires a sandboxed execution environment (inherited from
    ExecutionScorer). Configure sandbox in HarnessConfig to use this scorer.

    The instance metadata must contain a 'test' key with the test code to run,
    and the output.extracted_answer must contain the complete code to execute.

    Example instance metadata:
        {
            "test": "assert add(1, 2) == 3\\nassert add(0, 0) == 0",
            ...
        }
    """

    name: str = "code_exec"
    timeout: float = 20.0
    language: str = "python"
    separator: str = "\n\n"

    async def execute_python_script(
        self,
        execution_env: ExecutionEnvironment,
        code: str,
        *,
        timeout: float,
        python_executable: str = "python3",
        filename: str = "script.py",
        virtual_memory_limit_kb: int | None = None,
        data_limit_kb: int | None = None,
        stack_limit_kb: int | None = None,
    ) -> ExecutionResult:
        """Execute a Python script from a fresh temp directory."""
        tmp_dir = f"/tmp/{uuid.uuid4().hex}"
        script_path = f"{tmp_dir}/{filename}"
        quoted_tmp_dir = shlex.quote(tmp_dir)
        quoted_script_path = shlex.quote(script_path)
        cleanup_cmd = f"rm -rf {quoted_tmp_dir} || true"

        limit_cmds: list[str] = []
        if virtual_memory_limit_kb is not None:
            limit_cmds.append(f"ulimit -v {int(virtual_memory_limit_kb)}")
        if data_limit_kb is not None:
            limit_cmds.append(f"ulimit -d {int(data_limit_kb)}")
        if stack_limit_kb is not None:
            limit_cmds.append(f"ulimit -s {int(stack_limit_kb)}")

        inner_cmd_parts = [
            *limit_cmds,
            f"exec {shlex.quote(python_executable)} {shlex.quote(filename)}",
        ]
        run_cmd = (
            f"timeout {math.ceil(timeout)} bash -c {shlex.quote(' && '.join(inner_cmd_parts))}"
        )

        command = "\n".join(
            [
                f"mkdir -p {quoted_tmp_dir} || exit $?",
                (
                    f"printf '%s' {shlex.quote(code)} > {quoted_script_path}"
                    f" || {{ status=$?; {cleanup_cmd}; exit $status; }}"
                ),
                f"cd {quoted_tmp_dir} || {{ status=$?; {cleanup_cmd}; exit $status; }}",
                "status=0",
                f"{run_cmd} || status=$?",
                cleanup_cmd,
                "exit $status",
            ]
        )

        return await execution_env.execute_command(command, timeout=timeout + 1.0)

    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        """Score by executing code + tests in the sandbox.

        Args:
            instance: The instance being scored. Must have 'test' in metadata.
            output: The model output to score. extracted_answer should contain code.
            execution_env: The execution environment for running code.

        Returns:
            1.0 if all tests pass, 0.0 otherwise.
        """
        if output.extracted_answer is None:
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0

        full_code = f"{output.extracted_answer}{self.separator}{test_code}"

        result = await execution_env.execute_code(
            full_code,
            language=self.language,
            timeout=self.timeout,
        )
        if not result.success and result.error:
            instance_id = instance.metadata.get("id", "?")
            logger.warning(f"Code execution failed [{instance_id}]: {result.error}")
        return 1.0 if result.success else 0.0

"""MULTIPL_E scorer for evaluating generated code across multiple languages."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.types import Instance, LMOutput

from ..execution import ExecutionScorer
from .languages import get_evaluator
from .languages.base import EvalResult, ExecutionStatus

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment, ExecutionResult


@dataclass(frozen=True, slots=True)
class MultiplEScorer(ExecutionScorer):
    """Score MULTIPL_E code by compiling/executing against test cases.

    This scorer handles file-based compilation and execution using language-specific
    evaluators. Each language evaluator provides:
    - Compile and run commands
    - Language-specific error detection (syntax errors, exceptions, etc.)
    - Appropriate default timeouts

    Currently supported languages:
    - cpp: Compiles with g++ and executes
    - java: Compiles with javac and executes with java
    - js: Executes with node
    - php: Executes with php
    - rs: Compiles with rustc and executes
    - sh: Executes with bash

    The instance metadata must contain:
    - 'test': Test code to append after the generated code
    - 'language': One of the supported languages
    """

    name: str = "multipl_e"
    timeout: float | None = None  # None = use language-specific default
    language: str = "cpp"

    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        """Score by compiling/executing code + tests in the sandbox.

        Args:
            instance: The instance being scored. Must have 'test' in metadata.
            output: The model output to score. extracted_answer should contain code.
            execution_env: The execution environment for running code.

        Returns:
            1.0 if all tests pass, 0.0 otherwise.
        """
        if output.extracted_answer is None:
            output.metadata["execution_result"] = {"success": False, "error": "No extracted answer"}
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            output.metadata["execution_result"] = {"success": False, "error": "No test code"}
            return 0.0

        # Combine generated code with tests (single newline matches oe-eval-internal)
        full_code = f"{output.extracted_answer}\n{test_code}"

        result = await self._execute_with_evaluator(execution_env, full_code)

        # Store execution details (truncate long output)
        MAX_OUTPUT_LEN = 4000
        output.metadata["execution_result"] = {
            "success": result.success,
            "status": result.status.value,
            "exit_code": result.exit_code,
            "stdout": result.stdout[:MAX_OUTPUT_LEN] if result.stdout else "",
            "stderr": result.stderr[:MAX_OUTPUT_LEN] if result.stderr else "",
        }

        return 1.0 if result.success else 0.0

    async def _execute_with_evaluator(self, env: ExecutionEnvironment, code: str) -> EvalResult:
        """Execute code using the language-specific evaluator."""
        try:
            evaluator = get_evaluator(self.language)
        except ValueError as e:
            return EvalResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stderr=str(e),
            )

        # Use explicit timeout if set, otherwise use total timeout (compile + run + buffer).
        # Per-step timeouts are enforced by the `timeout` command in the shell (see
        # build_eval_command), so this outer timeout is a safety net.
        timeout = self.timeout if self.timeout is not None else evaluator.get_total_timeout()

        # Use a unique temp directory per execution to avoid conflicts with parallel runs
        tmp_dir = f"/tmp/{uuid.uuid4().hex}"

        # Build the evaluation command
        cmd = evaluator.build_eval_command(tmp_dir, code)

        try:
            exec_result: ExecutionResult = await env.execute_command(cmd, timeout=timeout)

            # Determine if execution timed out — either at the execution-env level
            # or via the `timeout` shell command (exit code 124)
            timed_out = (
                exec_result.error == "timeout"
                or exec_result.exit_code == 124  # `timeout` command exit code
                or "timed out" in (exec_result.error or "").lower()
                or "timed out" in (exec_result.output or "").lower()
            )

            return evaluator.categorize_result(
                exit_code=exec_result.exit_code,
                stdout=exec_result.output or "",
                stderr="",  # Our shell command combines stderr into stdout
                timed_out=timed_out,
            )
        except Exception as e:
            return EvalResult(
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stderr=str(e),
            )

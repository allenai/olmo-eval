"""MULTIPL_E scorer for evaluating generated code across multiple languages."""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.types import Instance, LMOutput

from .execution import ExecutionScorer

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment, ExecutionResult


@dataclass(frozen=True, slots=True)
class MultiplEScorer(ExecutionScorer):
    """Score MULTIPL_E code by compiling/executing against test cases.

    This scorer handles file-based compilation and execution for 6 languages:
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
    timeout: float = 10.0
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
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0

        # Combine generated code with tests
        full_code = f"{output.extracted_answer}\n\n{test_code}"

        result = await self._execute_language(execution_env, full_code)
        return 1.0 if result.success else 0.0

    async def _execute_language(self, env: ExecutionEnvironment, code: str) -> ExecutionResult:
        """Execute code in the appropriate language.

        Args:
            env: The execution environment.
            code: The full code including tests.

        Returns:
            ExecutionResult with success status.
        """
        # Use a unique temp directory per execution to avoid conflicts with parallel runs
        tmp_dir = f"/tmp/{uuid.uuid4().hex}"
        quoted_code = shlex.quote(code)

        match self.language:
            case "sh":
                # MULTIPL_E uses bash-specific syntax (e.g., [[ ]])
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/code.sh && "
                    f"/bin/bash {tmp_dir}/code.sh"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case "js":
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/code.js && "
                    f"node {tmp_dir}/code.js"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case "php":
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/code.php && "
                    f"php {tmp_dir}/code.php"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case "cpp":
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/code.cpp && "
                    f"g++ -o {tmp_dir}/a.out {tmp_dir}/code.cpp && {tmp_dir}/a.out"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case "java":
                # MULTIPL_E Java uses Problem class and requires javatuples
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/Problem.java && "
                    f"cd {tmp_dir} && javac -cp '/runtime/java/*' Problem.java && "
                    "java -cp '/runtime/java/*:.' Problem"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case "rs":
                cmd = (
                    f"mkdir -p {tmp_dir} && echo {quoted_code} > {tmp_dir}/code.rs && "
                    f"rustc -o {tmp_dir}/a.out {tmp_dir}/code.rs && {tmp_dir}/a.out"
                )
                return await env.execute_command(cmd, timeout=self.timeout)
            case _:
                from olmo_eval.common.execution import ExecutionResult

                return ExecutionResult(
                    success=False, error=f"Unsupported language: {self.language}"
                )

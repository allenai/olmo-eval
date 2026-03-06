"""MULTIPL_E scorer for evaluating generated code across multiple languages."""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from olmo_eval.common.types import Instance, LMOutput

from .execution import ExecutionScorer

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment, ExecutionResult


class LangConfig(NamedTuple):
    filename: str
    compile_cmd: str | None  # None for interpreted languages
    run_cmd: str
    timeout: float = 10.0  # Default timeout in seconds


# Language configurations: (filename, compile_cmd, run_cmd, timeout)
# {d} = tmp_dir, {f} = full file path
LANG_CONFIGS: dict[str, LangConfig] = {
    "sh": LangConfig("code.sh", None, "/bin/bash {f}"),
    "js": LangConfig("code.js", None, "node {f}"),
    "php": LangConfig("code.php", None, "php {f}"),
    "cpp": LangConfig("code.cpp", "g++ -o {d}/a.out {f}", "{d}/a.out", timeout=30.0),
    "rs": LangConfig("code.rs", "rustc -o {d}/a.out {f}", "{d}/a.out", timeout=30.0),
    "java": LangConfig(
        "Problem.java",
        "cd {d} && javac -cp '/runtime/java/*' Problem.java",
        "cd {d} && java -cp '/runtime/java/*:.' Problem",
        timeout=30.0,
    ),
}


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

        # Combine generated code with tests
        full_code = f"{output.extracted_answer}\n\n{test_code}"

        result = await self.exec_for_lang(execution_env, full_code)

        # Store execution details (truncate long output)
        MAX_OUTPUT_LEN = 4000
        output.metadata["execution_result"] = {
            "success": result.success,
            "exit_code": result.exit_code,
            "output": result.output[:MAX_OUTPUT_LEN] if result.output else "",
            "error": result.error,
        }

        return 1.0 if result.success else 0.0

    async def exec_for_lang(self, env: ExecutionEnvironment, code: str) -> ExecutionResult:
        """Execute code in the appropriate language."""
        from olmo_eval.common.execution import ExecutionResult

        config = LANG_CONFIGS.get(self.language)
        if config is None:
            return ExecutionResult(success=False, error=f"Unsupported language: {self.language}")

        # Use explicit timeout if set, otherwise use language default
        effective_timeout = self.timeout if self.timeout is not None else config.timeout

        # Use a unique temp directory per execution to avoid conflicts with parallel runs
        tmp_dir = f"/tmp/{uuid.uuid4().hex}"
        file_path = f"{tmp_dir}/{config.filename}"

        # Use shlex.quote to safely escape code for shell
        quoted_code = shlex.quote(code)

        # Build command: create dir, write code, compile (if needed), run
        parts = [
            f"mkdir -p {tmp_dir}",
            f"echo {quoted_code} > {file_path}",
        ]
        if config.compile_cmd:
            parts.append(config.compile_cmd.format(d=tmp_dir, f=file_path))
        parts.append(config.run_cmd.format(d=tmp_dir, f=file_path))

        cmd = " && ".join(parts)

        try:
            return await env.execute_command(cmd, timeout=effective_timeout)
        except Exception as e:
            return ExecutionResult(success=False, output="", exit_code=-1, error=str(e))

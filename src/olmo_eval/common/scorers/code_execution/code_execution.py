"""Code execution scorer for evaluating generated code against test cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_eval.common.types import Instance, LMOutput

from ..execution import ExecutionScorer

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment


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

        # Combine generated code with tests
        full_code = f"{output.extracted_answer}\n\n{test_code}"

        result = await execution_env.execute_code(
            full_code,
            language=self.language,
            timeout=self.timeout,
        )
        return 1.0 if result.success else 0.0

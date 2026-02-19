"""Base class for scorers that require sandboxed execution."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from olmo_eval.common.types import Instance, LMOutput

from .base import Scorer

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment


class SandboxRequiredError(RuntimeError):
    """Raised when a scorer requires a sandbox but none is available."""

    pass


@dataclass(frozen=True)
class ExecutionScorer(Scorer):
    """Base class for scorers that require sandboxed code execution.

    Subclasses must implement `ascore()` which receives an execution environment.
    The task runner is responsible for providing a valid execution environment.

    Example subclass:
        @dataclass(frozen=True, slots=True)
        class MyCodeScorer(ExecutionScorer):
            name: str = "my_scorer"

            async def ascore(
                self,
                instance: Instance,
                output: LMOutput,
                execution_env: ExecutionEnvironment,
            ) -> float:
                result = await execution_env.execute_code(output.text)
                return 1.0 if result.success else 0.0
    """

    requires_async: ClassVar[bool] = True

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Sync scoring is not supported for execution scorers.

        Raises:
            SandboxRequiredError: Always, since execution requires async sandbox.
        """
        raise SandboxRequiredError(
            f"{self.__class__.__name__} requires a sandbox execution environment. "
            "Configure sandboxes in HarnessConfig."
        )

    @abstractmethod
    async def ascore(
        self,
        instance: Instance,
        output: LMOutput,
        execution_env: ExecutionEnvironment,
    ) -> float:
        """Score using the execution environment.

        Subclasses implement this method with their scoring logic.

        Args:
            instance: The instance being scored.
            output: The model output to score.
            execution_env: The execution environment for running code.

        Returns:
            Score as a float (typically 0.0 to 1.0).
        """
        ...

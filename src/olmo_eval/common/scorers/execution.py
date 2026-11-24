"""Base classes for scorers that require async execution or context."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from olmo_eval.common.types import Instance, LMOutput, Response

from .base import Scorer

if TYPE_CHECKING:
    from olmo_eval.common.execution import ExecutionEnvironment, ScoringContext


class SandboxRequiredError(RuntimeError):
    """Raised when a scorer requires a sandbox but none is available."""

    pass


class ScoringIncompleteError(RuntimeError):
    """Raised when a scorer could not reach a verdict for an output.

    Distinguishes a score that was never produced, such as a grader that did
    not run or a payload that could not be read, from a solution that was
    graded and failed. The runner records these as lost scores rather than
    folding them into the metric as zeros.
    """


class SandboxInfrastructureError(RuntimeError):
    """A sandbox operation could not be completed because infrastructure failed."""


def format_scoring_error(exc: Exception, *, phase: str) -> dict[str, str]:
    """Describe a scoring failure for an output's ``scoring_errors``.

    ``infrastructure`` is set to ``"true"`` when the failure came from the
    environment rather than from the output: a judge or grader that gave no
    verdict (``ScoringIncompleteError``), no sandbox (``SandboxRequiredError``),
    or a sandbox that failed (``SandboxInfrastructureError``). The flag is a
    diagnostic; every scoring failure counts against the task's failure budget.
    """
    error = {"phase": phase, "type": type(exc).__qualname__}
    message = str(exc).strip()
    if message:
        error["message"] = message
    if isinstance(exc, (ScoringIncompleteError, SandboxRequiredError, SandboxInfrastructureError)):
        error["infrastructure"] = "true"
    return error


@dataclass(frozen=True)
class ContextScorer(Scorer):
    """Base class for scorers that require access to ScoringContext.

    Subclasses must implement `ascore_with_context()` which receives the full
    scoring context including inference pool and execution environment.

    Example subclass:
        @dataclass(frozen=True, slots=True)
        class MyJudgeScorer(ContextScorer):
            name: str = "my_judge"
            provider_name: str = "judge"

            async def ascore_with_context(
                self,
                instance: Instance,
                output: LMOutput,
                context: ScoringContext,
            ) -> float:
                provider = context.get_provider(self.provider_name)
                response = await provider.generate(...)
                return parse_score(response)
    """

    requires_async: ClassVar[bool] = True

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Sync scoring is not supported for context scorers.

        Raises:
            RuntimeError: Always, since context scorers require async execution.
        """
        raise RuntimeError(
            f"{self.__class__.__name__} requires async execution with ScoringContext. "
            "Ensure the task runner provides a ScoringContext."
        )

    @abstractmethod
    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        """Score using the scoring context.

        Subclasses implement this method with their scoring logic.

        Args:
            instance: The instance being scored.
            output: The model output to score.
            context: The scoring context with inference pool and execution env.

        Returns:
            Score as a float (typically 0.0 to 1.0).
        """
        ...


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


@dataclass(frozen=True)
class ResponseBatchScorer(Scorer):
    """Base class for scorers that grade a batch of responses in one call.

    Subclasses declare ``batch_size`` (number of responses per call, or ``None``
    to grade all responses in a bucket together) and implement
    :meth:`ascore_batch`, which receives a sequence of :class:`Response`
    objects and a :class:`ScoringContext` and returns one score per response
    (or one sequence of per-output scores per response) in the same order.

    Subclasses whose grading protocol groups responses before batching (for
    example CharXiv descriptive grading, which batches up to 5 responses sharing
    the same question template id) may override :meth:`batch_key`.
    """

    requires_async: ClassVar[bool] = True
    batch_size: int | None = None

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Sync single-output scoring is not supported for response-batch scorers."""
        raise RuntimeError(
            f"{self.__class__.__name__} requires batch async execution via ascore_batch(). "
            "Use Task.score_responses() with a ScoringContext."
        )

    def batch_key(self, response: Response) -> Hashable:
        """Group key for batching responses; different keys are batched separately."""
        return None

    @abstractmethod
    async def ascore_batch(
        self,
        responses: Sequence[Response],
        context: ScoringContext,
    ) -> Sequence[float | list[float | None] | tuple[float | None, ...] | None]:
        """Score a batch of responses in one call.

        Args:
            responses: Non-empty batch of responses to grade together.
            context: The scoring context with concurrency and runtime settings.

        Returns:
            A sequence aligned with ``responses``. Each element may be a single
            ``float`` (applied to the response's outputs), ``None`` (unscored),
            or a sequence of per-output ``float | None`` values aligned with
            ``response.outputs``.
        """
        ...

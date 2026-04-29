"""Tests for olmo_eval.tasks.base module."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import pytest

from olmo_eval.common.execution.environment import ExecutionResult, ScoringContext
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import ExactMatchScorer
from olmo_eval.common.scorers.execution import ExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, Split
from olmo_eval.evals.tasks.common import OutputScoreAggregation, Task, TaskConfig


class ConcreteTask(Task):
    """A concrete task implementation for testing."""

    def __init__(self, config: TaskConfig, instances_data: list[Instance] | None = None):
        super().__init__(config)
        self._instances_data = instances_data or [
            Instance(question="What is 2+2?", gold_answer="4"),
            Instance(question="What is 3+3?", gold_answer="6"),
        ]

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._instances_data

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_custom_values(self):
        """Test creating config with custom values."""
        from olmo_eval.data import DataSource

        config = TaskConfig(
            name="custom",
            data_source=DataSource(path="test/dataset", subset="subset1"),
            num_fewshot=5,
            fewshot_seed=123,
            limit=100,
            split=Split.VALIDATION,
        )
        assert config.data_source.subset == "subset1"
        assert config.num_fewshot == 5
        assert config.fewshot_seed == 123
        assert config.limit == 100
        assert config.split == Split.VALIDATION

    def test_config_with_metrics(self):
        """Test config with metrics."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)

        config = TaskConfig(
            name="scored",
            data_source="test/dataset",
            metrics=(metric,),
        )
        assert len(config.metrics) == 1


class TestTask:
    """Tests for Task base class."""

    def test_format_request(self):
        """Test format_request produces LMRequest."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        instance = Instance(question="Test question?", gold_answer="answer")
        request = task.format_request(instance)

        assert isinstance(request, LMRequest)
        assert request.prompt == "Test question?"
        assert request.request_type == RequestType.COMPLETION

    def test_extract_answer(self):
        """Test extract_answer extracts from output."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        output = LMOutput(text="  extracted answer  ")
        answer = task.extract_answer(output)

        assert answer == "extracted answer"

    def test_get_fewshot_default_empty(self):
        """Test that default get_fewshot returns empty list."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        fewshot = task.get_fewshot()
        assert fewshot == []

    def test_get_fewshot_cached(self):
        """Test that fewshot examples are cached."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        fewshot1 = task.get_fewshot()
        fewshot2 = task.get_fewshot()

        assert fewshot1 is fewshot2  # Same object (cached)


@pytest.mark.anyio
class TestTaskScoring:
    """Tests for Task scoring functionality."""

    def _make_request(self, prompt: str) -> LMRequest:
        """Helper to create a simple LMRequest."""
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    async def test_score_responses_extracts_answers(self):
        """Test that score_responses extracts answers from outputs."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="4")
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=[output],
        )

        scored = await task.score_responses([response])

        assert len(scored) == 1
        assert scored[0].outputs[0].extracted_answer == "4"

    async def test_score_responses_applies_scorers(self):
        """Test that score_responses applies scorers from metrics."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="4")
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=[output],
        )

        scored = await task.score_responses([response])

        assert "exact_match" in scored[0].scores
        assert scored[0].scores["exact_match"] == 1.0

    async def test_score_responses_incorrect_answer(self):
        """Test scoring with incorrect answer."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="5")
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=[output],
        )

        scored = await task.score_responses([response])

        assert scored[0].scores["exact_match"] == 0.0

    async def test_score_responses_multiple_outputs_takes_max(self):
        """Test that scoring takes max score across multiple outputs."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        outputs = [
            LMOutput(text="3"),  # Wrong
            LMOutput(text="4"),  # Correct
            LMOutput(text="5"),  # Wrong
        ]
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=outputs,
        )

        scored = await task.score_responses([response])

        assert scored[0].scores["exact_match"] == 1.0  # Max of [0, 1, 0]

    async def test_score_responses_multiple_outputs_can_take_first(self):
        """Test that scoring can use the first sampled output like oe-eval."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
            output_score_aggregation=OutputScoreAggregation.FIRST,
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        outputs = [
            LMOutput(text="3"),  # Wrong first sample
            LMOutput(text="4"),  # Correct later sample
            LMOutput(text="5"),  # Wrong
        ]
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=outputs,
        )

        scored = await task.score_responses([response])

        assert scored[0].scores["exact_match"] == 0.0
        assert scored[0].outputs[0].metadata["score:exact_match"] == 0.0
        assert scored[0].outputs[1].metadata["score:exact_match"] == 1.0


@pytest.mark.anyio
class TestTaskMetrics:
    """Tests for Task metrics computation."""

    def _make_request(self, prompt: str) -> LMRequest:
        """Helper to create a simple LMRequest."""
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    async def test_compute_metrics(self):
        """Test compute_metrics aggregates scores."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
        )
        task = ConcreteTask(config)

        # Create responses with mixed results
        responses = [
            Response(
                instance=Instance(question="Q1", gold_answer="A"),
                request=self._make_request("Q1"),
                outputs=[LMOutput(text="A")],
            ),
            Response(
                instance=Instance(question="Q2", gold_answer="B"),
                request=self._make_request("Q2"),
                outputs=[LMOutput(text="B")],
            ),
            Response(
                instance=Instance(question="Q3", gold_answer="C"),
                request=self._make_request("Q3"),
                outputs=[LMOutput(text="X")],
            ),
        ]

        # Score first
        scored = await task.score_responses(responses)

        # Compute metrics (returns nested structure: {metric: {scorer: value}})
        metrics = task.compute_metrics(scored)

        assert "accuracy" in metrics
        assert "exact_match" in metrics["accuracy"]
        assert metrics["accuracy"]["exact_match"] == pytest.approx(2 / 3)

    async def test_compute_metrics_first_sample_accuracy_with_pass_at_1(self):
        """Accuracy can use the first sample while pass@1 still uses all samples."""
        accuracy = AccuracyMetric(scorer=ExactMatchScorer)
        pass_at_1 = PassAtKMetric(k=1, scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(accuracy, pass_at_1),
            output_score_aggregation=OutputScoreAggregation.FIRST,
        )
        task = ConcreteTask(config)

        response = Response(
            instance=Instance(question="What is 2+2?", gold_answer="4"),
            request=self._make_request("What is 2+2?"),
            outputs=[
                LMOutput(text="3"),
                LMOutput(text="4"),
                LMOutput(text="5"),
            ],
        )

        scored = await task.score_responses([response])
        metrics = task.compute_metrics(scored)

        assert metrics["accuracy"]["exact_match"] == 0.0
        assert metrics["pass_at_1"]["exact_match"] == pytest.approx(1 / 3)

    def test_task_config_normalizes_string_output_aggregation(self):
        """Legacy string config values should normalize to the enum."""
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            output_score_aggregation="first",
        )

        assert config.output_score_aggregation == OutputScoreAggregation.FIRST

    def test_compute_metrics_empty_responses(self):
        """Test compute_metrics with empty responses."""
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            metrics=(metric,),
        )
        task = ConcreteTask(config)

        metrics = task.compute_metrics([])

        # Returns nested structure: {metric: {scorer: value}}
        assert "accuracy" in metrics
        assert "exact_match" in metrics["accuracy"]
        assert metrics["accuracy"]["exact_match"] == 0.0


# ── Helpers for execution concurrency tests ─────────────────────────────────


@dataclass(frozen=True)
class _ConcurrencyTracker:
    """Track peak concurrent calls (mutable internals, frozen wrapper)."""

    _current: list[int]  # [current_count] — single-element list for mutability
    _peak: list[int]  # [peak_count]
    _lock: asyncio.Lock

    @staticmethod
    def create() -> _ConcurrencyTracker:
        return _ConcurrencyTracker(_current=[0], _peak=[0], _lock=asyncio.Lock())

    async def enter(self) -> None:
        async with self._lock:
            self._current[0] += 1
            if self._current[0] > self._peak[0]:
                self._peak[0] = self._current[0]

    async def exit(self) -> None:
        async with self._lock:
            self._current[0] -= 1

    @property
    def peak(self) -> int:
        return self._peak[0]


@dataclass(frozen=True, slots=True)
class _MockExecutionScorer(ExecutionScorer):
    """ExecutionScorer that tracks concurrency and returns a fixed score."""

    name: str = "mock_exec"
    tracker: _ConcurrencyTracker | None = None
    requires_async: ClassVar[bool] = True

    async def ascore(self, instance, output, execution_env) -> float:
        if self.tracker:
            await self.tracker.enter()
        try:
            await asyncio.sleep(0.01)  # force overlap
            return 1.0
        finally:
            if self.tracker:
                await self.tracker.exit()


@dataclass(frozen=True, slots=True)
class _FailingExecutionScorer(ExecutionScorer):
    """ExecutionScorer that can fail for selected outputs."""

    name: str = "failing_exec"
    fail_text: str = "boom"
    requires_async: ClassVar[bool] = True

    async def ascore(self, instance, output, execution_env) -> float:
        if output.text == self.fail_text:
            raise RuntimeError("intentional scorer failure")
        return 1.0


class _MockExecutionEnv:
    """Minimal mock that satisfies the ExecutionEnvironment protocol + semaphore."""

    def __init__(self, semaphore: asyncio.Semaphore | None = None) -> None:
        self._semaphore = semaphore

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None):
        return ExecutionResult(success=True)

    async def execute_code(self, code: str, language: str = "python", timeout: float | None = None):
        return ExecutionResult(success=True)

    def get_executor(self, required_capabilities):
        return self

    def get_execution_semaphore(self, required_capabilities):
        return self._semaphore


def _make_responses(n_responses: int, n_outputs: int) -> list[Response]:
    """Create test responses with the given shape."""
    responses = []
    for i in range(n_responses):
        inst = Instance(question=f"Q{i}", gold_answer="A")
        outputs = [LMOutput(text=f"out{j}") for j in range(n_outputs)]
        req = LMRequest(request_type=RequestType.COMPLETION, prompt=f"Q{i}")
        responses.append(Response(instance=inst, request=req, outputs=outputs))
    return responses


# ── Execution concurrency tests ─────────────────────────────────────────────


@pytest.mark.anyio
class TestExecutionConcurrency:
    """Tests that the shared execution semaphore properly bounds concurrency."""

    async def test_shared_semaphore_bounds_concurrency(self):
        """Peak concurrent sandbox calls must not exceed the semaphore limit."""
        tracker = _ConcurrencyTracker.create()
        semaphore_limit = 2
        env = _MockExecutionEnv(semaphore=asyncio.Semaphore(semaphore_limit))

        scorer = _MockExecutionScorer(tracker=tracker)
        metric = PassAtKMetric(k=1, scorer=lambda: scorer)
        config = TaskConfig(name="test", data_source="test/dataset", metrics=(metric,))
        task = ConcreteTask(config)

        # 3 responses x 8 outputs = 24 scoring coroutines
        responses = _make_responses(n_responses=3, n_outputs=8)
        context = ScoringContext(execution_env=env, scoring_concurrency=100)

        await task.score_responses(responses, context=context)

        assert tracker.peak <= semaphore_limit
        # Verify all outputs were scored
        for resp in responses:
            for output in resp.outputs:
                assert output.metadata is not None
                assert "score:mock_exec" in output.metadata

    async def test_no_semaphore_still_works(self):
        """When execution env has no get_execution_semaphore, scoring completes."""
        tracker = _ConcurrencyTracker.create()
        env = _MockExecutionEnv(semaphore=None)

        scorer = _MockExecutionScorer(tracker=tracker)
        metric = PassAtKMetric(k=1, scorer=lambda: scorer)
        config = TaskConfig(name="test", data_source="test/dataset", metrics=(metric,))
        task = ConcreteTask(config)

        responses = _make_responses(n_responses=2, n_outputs=4)
        context = ScoringContext(execution_env=env, scoring_concurrency=100)

        await task.score_responses(responses, context=context)

        # All outputs scored — concurrency was unbounded
        for resp in responses:
            for output in resp.outputs:
                assert output.metadata is not None
                assert "score:mock_exec" in output.metadata
        assert tracker.peak > 0

    async def test_env_without_get_execution_semaphore(self):
        """Execution env that lacks get_execution_semaphore method still works."""

        class _BareEnv:
            """Env without semaphore support."""

            is_running = True

            async def execute(self, command, timeout=None):
                return ""

            async def execute_command(self, command, timeout=None):
                return ExecutionResult(success=True)

            async def execute_code(self, code, language="python", timeout=None):
                return ExecutionResult(success=True)

        env = _BareEnv()
        scorer = _MockExecutionScorer()
        metric = PassAtKMetric(k=1, scorer=lambda: scorer)
        config = TaskConfig(name="test", data_source="test/dataset", metrics=(metric,))
        task = ConcreteTask(config)

        responses = _make_responses(n_responses=1, n_outputs=2)
        context = ScoringContext(execution_env=env, scoring_concurrency=10)

        await task.score_responses(responses, context=context)

        for resp in responses:
            assert "mock_exec" in resp.scores


@pytest.mark.anyio
class TestAsyncScoringFailures:
    """Tests that async scoring failures are recorded instead of dropping metrics."""

    async def test_async_scorer_failure_becomes_zero_with_diagnostics(self):
        env = _MockExecutionEnv(semaphore=asyncio.Semaphore(4))
        scorer = _FailingExecutionScorer()
        metric = PassAtKMetric(k=1, scorer=lambda: scorer)
        config = TaskConfig(name="test", data_source="test/dataset", metrics=(metric,))
        task = ConcreteTask(config)

        responses = _make_responses(n_responses=2, n_outputs=1)
        responses[0].outputs[0].text = "ok"
        responses[1].outputs[0].text = "boom"
        context = ScoringContext(execution_env=env, scoring_concurrency=8)

        scored = await task.score_responses(responses, context=context)

        assert scored[0].scores["failing_exec"] == 1.0
        assert scored[0].outputs[0].metadata["score:failing_exec"] == 1.0
        assert "scoring_errors" not in scored[0].outputs[0].metadata

        failing_output = scored[1].outputs[0]
        assert scored[1].scores["failing_exec"] == 0.0
        assert failing_output.metadata["score:failing_exec"] == 0.0
        assert failing_output.metadata["scoring_errors"] == {
            "failing_exec": {
                "phase": "execution",
                "type": "RuntimeError",
                "message": "intentional scorer failure",
            }
        }

    async def test_partial_async_failures_preserve_passing_outputs(self):
        env = _MockExecutionEnv(semaphore=asyncio.Semaphore(4))
        scorer = _FailingExecutionScorer()
        metric = PassAtKMetric(k=1, scorer=lambda: scorer)
        config = TaskConfig(name="test", data_source="test/dataset", metrics=(metric,))
        task = ConcreteTask(config)

        response = _make_responses(n_responses=1, n_outputs=2)[0]
        response.outputs[0].text = "ok"
        response.outputs[1].text = "boom"
        context = ScoringContext(execution_env=env, scoring_concurrency=8)

        scored = await task.score_responses([response], context=context)

        assert scored[0].scores["failing_exec"] == 1.0
        assert scored[0].outputs[0].metadata["score:failing_exec"] == 1.0
        assert scored[0].outputs[1].metadata["score:failing_exec"] == 0.0
        assert scored[0].outputs[1].metadata["scoring_errors"]["failing_exec"]["type"] == (
            "RuntimeError"
        )

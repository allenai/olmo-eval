"""Tests for olmo_eval.tasks.base module."""

from collections.abc import Iterator

import pytest

from olmo_eval.core import (
    AccuracyMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    Split,
)
from olmo_eval.evals.tasks import Task, TaskConfig


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

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = TaskConfig(name="test", data_source="test/dataset")
        assert config.name == "test"
        assert config.data_source == "test/dataset"

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TaskConfig(name="test", data_source="test/dataset")
        assert config.formatter is None
        assert config.scorers == ()
        assert config.metrics == ()
        assert config.num_fewshot == 0
        assert config.fewshot_seed == 42
        assert config.limit is None
        assert config.split == Split.TEST
        assert config.primary_metric is None

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

    def test_config_with_scorers_and_metrics(self):
        """Test config with scorers and metrics."""
        scorer = ExactMatchScorer()
        metric = AccuracyMetric(scorer=ExactMatchScorer)

        config = TaskConfig(
            name="scored",
            data_source="test/dataset",
            scorers=(scorer,),
            metrics=(metric,),
        )
        assert len(config.scorers) == 1
        assert len(config.metrics) == 1


class TestTask:
    """Tests for Task base class."""

    def test_task_initialization(self):
        """Test task initialization stores config."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)
        assert task.config is config

    def test_instances_iterator(self):
        """Test that instances returns an iterator."""
        config = TaskConfig(name="test", data_source="test/dataset")
        task = ConcreteTask(config)

        instances = list(task.instances)
        assert len(instances) == 2
        assert all(isinstance(i, Instance) for i in instances)

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


class TestTaskScoring:
    """Tests for Task scoring functionality."""

    def _make_request(self, prompt: str) -> LMRequest:
        """Helper to create a simple LMRequest."""
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def test_score_responses_extracts_answers(self):
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

        scored = task.score_responses([response])

        assert len(scored) == 1
        assert scored[0].outputs[0].extracted_answer == "4"

    def test_score_responses_applies_scorers(self):
        """Test that score_responses applies all scorers."""
        scorer = ExactMatchScorer()
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            scorers=(scorer,),
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="4")
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=[output],
        )

        scored = task.score_responses([response])

        assert "exact_match" in scored[0].scores
        assert scored[0].scores["exact_match"] == 1.0

    def test_score_responses_incorrect_answer(self):
        """Test scoring with incorrect answer."""
        scorer = ExactMatchScorer()
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            scorers=(scorer,),
        )
        task = ConcreteTask(config)

        instance = Instance(question="What is 2+2?", gold_answer="4")
        output = LMOutput(text="5")
        response = Response(
            instance=instance,
            request=self._make_request("What is 2+2?"),
            outputs=[output],
        )

        scored = task.score_responses([response])

        assert scored[0].scores["exact_match"] == 0.0

    def test_score_responses_multiple_outputs_takes_max(self):
        """Test that scoring takes max score across multiple outputs."""
        scorer = ExactMatchScorer()
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            scorers=(scorer,),
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

        scored = task.score_responses([response])

        assert scored[0].scores["exact_match"] == 1.0  # Max of [0, 1, 0]


class TestTaskMetrics:
    """Tests for Task metrics computation."""

    def _make_request(self, prompt: str) -> LMRequest:
        """Helper to create a simple LMRequest."""
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def test_compute_metrics(self):
        """Test compute_metrics aggregates scores."""
        scorer = ExactMatchScorer()
        metric = AccuracyMetric(scorer=ExactMatchScorer)
        config = TaskConfig(
            name="test",
            data_source="test/dataset",
            scorers=(scorer,),
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
        scored = task.score_responses(responses)

        # Compute metrics
        metrics = task.compute_metrics(scored)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(2 / 3)

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

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.0

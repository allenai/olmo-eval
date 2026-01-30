"""Tests for TaskConfig integration with DataSource."""

import pytest

from olmo_eval.data import DataSource, SourceType
from olmo_eval.evals.tasks.core import TaskConfig


class TestTaskConfigDataSource:
    """Tests for TaskConfig data_source field."""

    def test_data_source_with_datasource_object(self):
        config = TaskConfig(
            name="test_task",
            data_source=DataSource(path="cais/mmlu", subset="math"),
        )
        source = config.get_data_source()
        assert source.path == "cais/mmlu"
        assert source.subset == "math"
        assert source.source_type == SourceType.HF

    def test_data_source_with_string_uri(self):
        config = TaskConfig(
            name="test_task",
            data_source="hf://cais/mmlu?subset=math",
        )
        source = config.get_data_source()
        assert source.path == "cais/mmlu"
        assert source.subset == "math"

    def test_get_data_source_with_split_override(self):
        config = TaskConfig(
            name="test_task",
            data_source=DataSource(path="cais/mmlu", split="test"),
        )
        source = config.get_data_source(split="validation")
        assert source.split == "validation"

    def test_get_data_source_uses_config_split_by_default(self):
        from olmo_eval.core.types import Split

        config = TaskConfig(
            name="test_task",
            data_source=DataSource(path="cais/mmlu"),
            split=Split.VALIDATION,
        )
        source = config.get_data_source()
        assert source.split == "validation"

    def test_get_data_source_raises_when_not_configured(self):
        config = TaskConfig(name="test_task")
        with pytest.raises(ValueError, match="No data source configured"):
            config.get_data_source()


class TestTaskConfigFewshotSource:
    """Tests for TaskConfig fewshot_source field."""

    def test_fewshot_source_with_datasource(self):
        config = TaskConfig(
            name="test_task",
            data_source=DataSource(path="cais/mmlu"),
            fewshot_source=DataSource(path="cais/mmlu", split="dev"),
        )
        source = config.get_fewshot_source()
        assert source is not None
        assert source.split == "dev"

    def test_fewshot_source_falls_back_to_main_source(self):
        config = TaskConfig(
            name="test_task",
            data_source=DataSource(path="cais/mmlu"),
        )
        source = config.get_fewshot_source(split="dev")
        assert source is not None
        assert source.path == "cais/mmlu"
        assert source.split == "dev"

    def test_fewshot_source_returns_none_when_no_source(self):
        config = TaskConfig(name="test_task")
        source = config.get_fewshot_source()
        assert source is None

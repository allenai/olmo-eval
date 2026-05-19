"""Tests for AIME task registration and document processing."""

import pytest

from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_ALL_TASKS = ("aime_2024", "aime_2025", "aime_2026")
_ALL_VARIANTS = ("pass_at_32", "pass_at_32_rlzero")
_SIXTEEN_K_TASKS = ("aime_2025", "aime_2026")


class TestAIMERegistration:
    """Tests for AIME task registration."""

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_task_registered(self, task_name):
        assert task_name in list_tasks()

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task.config.name == task_name

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    @pytest.mark.parametrize("variant", _ALL_VARIANTS)
    def test_variants_registered(self, task_name, variant):
        task = get_task(f"{task_name}:{variant}")
        assert task is not None

    @pytest.mark.parametrize("task_name", _SIXTEEN_K_TASKS)
    def test_pass_at_32_16k_variant_overrides_only_max_tokens(self, task_name):
        base = get_task(f"{task_name}:pass_at_32")
        variant = get_task(f"{task_name}:pass_at_32:16k")

        assert variant.config.formatter == base.config.formatter
        assert variant.config.metrics == base.config.metrics
        assert variant.config.primary_metric == base.config.primary_metric

        base_sampling = base.config.sampling_params
        variant_sampling = variant.config.sampling_params
        assert base_sampling is not None
        assert variant_sampling is not None
        assert base_sampling.max_tokens == 32768
        assert variant_sampling.max_tokens == 16384
        assert variant_sampling.temperature == base_sampling.temperature == 0.6
        assert variant_sampling.top_p == base_sampling.top_p == 0.95
        assert variant_sampling.num_samples == base_sampling.num_samples == 32

    def test_aime_2026_uses_matharena_dataset(self):
        task = get_task("aime_2026")
        assert isinstance(task.config.data_source, DataSource)
        assert task.config.data_source.path == "MathArena/aime_2026"
        assert task.config.get_data_source().split == "train"


class TestAIMEProcessDoc:
    """Tests for AIME document conversion."""

    def test_aime_2025_filters_year_and_normalizes_answer(self):
        task = get_task("aime_2025")

        accepted = task.process_doc(
            {
                "year": 2025,
                "problem": "Find the value of x.",
                "answer": "007",
                "id": "2025-I-1",
                "problem_number": 1,
            },
            index=0,
        )
        rejected = task.process_doc(
            {
                "year": 2024,
                "problem": "Wrong year.",
                "answer": "123",
            },
            index=0,
        )

        assert accepted is not None
        assert accepted.question == "Find the value of x."
        assert accepted.gold_answer == "7"
        assert accepted.metadata == {
            "id": "2025-I-1",
            "year": 2025,
            "problem_number": 1,
            "all_gold_answers": ["7"],
        }
        assert rejected is None

    def test_aime_2026_maps_matharena_schema(self):
        task = get_task("aime_2026")

        instance = task.process_doc(
            {
                "problem_idx": 6,
                "problem": "What is the answer?",
                "answer": 29,
            },
            index=3,
        )

        assert instance is not None
        assert instance.question == "What is the answer?"
        assert instance.gold_answer == "29"
        assert instance.metadata == {
            "id": 6,
            "year": 2026,
            "problem_number": 6,
            "all_gold_answers": ["29"],
        }

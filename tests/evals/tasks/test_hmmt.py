"""Tests for HMMT task registration and document processing."""

import pytest

from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_ALL_TASKS = ("hmmt_feb_2025", "hmmt_nov_2025", "hmmt_feb_2026")
_ALL_VARIANTS = ("pass_at_32", "pass_at_32_rlzero")
_SIXTEEN_K_TASKS = ("hmmt_nov_2025", "hmmt_feb_2026")
_DATASETS_BY_TASK = {
    "hmmt_feb_2025": "MathArena/hmmt_feb_2025",
    "hmmt_nov_2025": "MathArena/hmmt_nov_2025",
    "hmmt_feb_2026": "MathArena/hmmt_feb_2026",
}


class TestHMMTRegistration:
    """Tests for HMMT task registration."""

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

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_task_uses_matharena_dataset(self, task_name):
        task = get_task(task_name)
        assert isinstance(task.config.data_source, DataSource)
        assert task.config.data_source.path == _DATASETS_BY_TASK[task_name]
        assert task.config.get_data_source().split == "train"


class TestHMMTProcessDoc:
    """Tests for HMMT document conversion."""

    def test_hmmt_feb_2026_maps_matharena_schema(self):
        task = get_task("hmmt_feb_2026")

        instance = task.process_doc(
            {
                "problem_idx": 17,
                "problem": "Find the exact value.",
                "answer": r" \frac{2}{5} ",
                "problem_type": [" Combinatorics ", " Geometry "],
            },
            index=3,
        )

        assert instance is not None
        assert instance.question == "Find the exact value."
        assert instance.gold_answer == r"\frac{2}{5}"
        assert instance.metadata == {
            "id": 17,
            "year": 2026,
            "season": "feb",
            "competition": "feb_2026",
            "date": "2026-02-14",
            "problem_number": 17,
            "problem_types": ["Combinatorics", "Geometry"],
            "all_gold_answers": [r"\frac{2}{5}"],
        }

    def test_hmmt_nov_2025_handles_missing_problem_types(self):
        task = get_task("hmmt_nov_2025")

        instance = task.process_doc(
            {
                "problem_idx": 4,
                "problem": "Compute the answer.",
                "answer": "42",
            },
            index=1,
        )

        assert instance is not None
        assert instance.question == "Compute the answer."
        assert instance.gold_answer == "42"
        assert instance.metadata == {
            "id": 4,
            "year": 2025,
            "season": "nov",
            "competition": "nov_2025",
            "date": "2025-11-08",
            "problem_number": 4,
            "problem_types": None,
            "all_gold_answers": ["42"],
        }

    def test_hmmt_filters_out_other_competitions(self):
        task = get_task("hmmt_feb_2026")

        rejected = task.process_doc(
            {
                "dataset_path": "MathArena/hmmt_nov_2025",
                "problem_idx": 1,
                "problem": "Wrong competition.",
                "answer": "1",
            },
            index=0,
        )

        assert rejected is None

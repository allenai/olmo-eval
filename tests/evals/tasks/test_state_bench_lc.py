"""Tests for the StateBench long-context tasks and suites."""

import pytest

from olmo_eval.common.types import RequestType
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.state_bench_lc import STATE_BENCH_V1, STATE_BENCH_V2

SOLVABLE_TASK = "state_bench_v2_integer_code_modular_solvable_tokens_under_2k"


def _solvable_row() -> dict:
    return {
        "example_id": "0135531c140f93435dcb3194",
        "instance_id": "4ee9f3e5eb86fdfbeedc",
        "formatter": "integer-code-modular",
        "complexity": "solvable",
        "prefix": "a <- 97; TMP <- 45; TMP <- (TMP + 52) mod 100; a <- TMP;\n\n",
        "choices": ["a ?= 45; TMP ?= 97;", "a ?= 97; TMP ?= 97;"],
        "correct_choice_index": 1,
        "token_stratum": "tokens_under_2k",
    }


class TestReleases:
    def test_v1_keeps_its_revision_and_lacks_solvable(self):
        task = get_task("state_bench_integer_code_aperiodic_tokens_under_2k")
        assert task.data_source.revision == "683d249128a0d387eda27c8cfc8c1f9df0389495"
        assert STATE_BENCH_V1.complexities == ("aperiodic", "periodic", "r-trivial")
        assert not any("solvable" in name for name in list_tasks() if "v2" not in name)

    def test_v2_has_four_complexity_classes(self):
        assert STATE_BENCH_V2.complexities == ("aperiodic", "periodic", "r-trivial", "solvable")

    @pytest.mark.parametrize(
        "config_name",
        ["integer-code-modular--solvable", "spreadsheet-cells-modular--solvable"],
    )
    def test_solvable_tasks_load_their_config_at_the_v2_revision(self, config_name: str):
        task = get_task(STATE_BENCH_V2.task_name(config_name, "tokens_64k"))
        assert task.data_source.path == "allenai/state-bench"
        assert task.data_source.subset == config_name
        assert task.data_source.split == "test__tokens_64k"
        assert task.data_source.revision == STATE_BENCH_V2.revision

    def test_v1_and_v2_namesakes_are_distinct_specs(self):
        v1 = get_task(STATE_BENCH_V1.task_name("cube-painting--periodic", "tokens_8k"))
        v2 = get_task(STATE_BENCH_V2.task_name("cube-painting--periodic", "tokens_8k"))
        assert v1.data_source.revision != v2.data_source.revision
        assert compute_task_hash(v1.config.to_dict()) != compute_task_hash(v2.config.to_dict())

    def test_2m_plus_is_registered_only_where_staged(self):
        assert STATE_BENCH_V2.configs_for_stratum("tokens_2m_plus") == (
            "spreadsheet-cells-modular--solvable",
        )
        assert "state_bench_v2_spreadsheet_cells_modular_solvable_tokens_2m_plus" in list_tasks()
        assert "state_bench_v2_integer_code_modular_solvable_tokens_2m_plus" not in list_tasks()
        assert "tokens_2m_plus" not in STATE_BENCH_V1.token_strata


class TestSuites:
    def test_solvable_suite_spans_only_solvable_configs(self):
        tasks = get_suite("state_bench_v2:solvable").tasks
        assert SOLVABLE_TASK in tasks
        assert all("_solvable_" in name for name in tasks)
        assert {name.split("_solvable_")[0] for name in tasks} == {
            "state_bench_v2_integer_code_modular",
            "state_bench_v2_spreadsheet_cells_modular",
        }

    def test_v2_full_suite_includes_solvable_and_v1_suite_does_not(self):
        assert SOLVABLE_TASK in get_suite("state_bench_v2").tasks
        assert f"{SOLVABLE_TASK}_10pct" in get_suite("state_bench_v2_10pct").tasks
        v1_tasks = get_suite("state_bench").tasks
        assert not any(name.startswith("state_bench_v2") for name in v1_tasks)

    def test_tier_suite_covers_only_staged_configs(self):
        suite = get_suite("state_bench_v2:tokens_2m_plus")
        assert suite.tasks == ("state_bench_v2_spreadsheet_cells_modular_solvable_tokens_2m_plus",)
        assert "Covers 1 of 17 configs" in suite.description

    def test_formatter_without_a_tier_registers_no_suite(self):
        with pytest.raises(KeyError):
            get_suite("state_bench_v2:people_in_rooms:tokens_1m")


class TestProcessDoc:
    def test_solvable_row_becomes_a_ranked_choice_request(self):
        task = get_task(SOLVABLE_TASK)
        instance = task.process_doc(_solvable_row(), index=3)
        assert instance is not None
        assert instance.gold_answer == "a ?= 97; TMP ?= 97;"
        assert instance.metadata["complexity"] == "solvable"
        assert instance.metadata["gold_idx"] == 1

        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations == (" a ?= 45; TMP ?= 97;", " a ?= 97; TMP ?= 97;")

    def test_out_of_range_gold_index_is_skipped(self):
        row = _solvable_row() | {"correct_choice_index": 2}
        assert get_task(SOLVABLE_TASK).process_doc(row) is None

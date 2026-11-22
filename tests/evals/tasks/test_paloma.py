"""Tests for the Paloma perplexity tasks and the paloma macro bits-per-byte suite."""

import pytest

from olmo_eval.common.metrics import BPBMetricByteAvg
from olmo_eval.common.types import RequestType, Split
from olmo_eval.evals.suites import AggregationStrategy, get_suite
from olmo_eval.evals.tasks import paloma
from olmo_eval.evals.tasks.common import get_task, list_tasks


class TestRegistration:
    def test_one_task_per_paloma_source(self):
        registered = {name for name in list_tasks() if name.startswith("paloma_")}
        assert registered == {f"paloma_{subset}" for subset in paloma.PALOMA_SUBSETS}
        assert len(paloma.PALOMA_SUBSETS) == 16

    def test_tasks_read_the_pinned_validation_files(self):
        source = get_task("paloma_c4_en").config.data_source
        assert source.path == "allenai/paloma"
        assert source.subset == "c4_en"
        assert source.revision == paloma.PALOMA_REVISION
        assert source.data_files == "c4_en/val/*"
        assert source.split == "train"

    def test_only_metric_is_byte_weighted_bpb(self):
        config = get_task("paloma_gab").config
        assert config.split == Split.VALIDATION
        assert isinstance(config.get_primary_metric(), BPBMetricByteAvg)
        assert [type(metric) for metric in config.metrics] == [BPBMetricByteAvg]


class TestInstances:
    @pytest.fixture
    def loaded(self, monkeypatch):
        sources = []
        docs = [
            {"id": "doc-a", "text": "short document"},
            {"id": "doc-b", "text": "y" * 100_000},
        ]

        def load(_loader, source):
            sources.append(source)
            return iter(docs)

        monkeypatch.setattr(paloma.DataLoader, "load", load)
        return sources

    def test_each_document_is_one_instance_however_long(self, loaded):
        instances = list(get_task("paloma_ptb").instances)
        assert [i.gold_answer for i in instances] == ["short document", "y" * 100_000]
        assert [i.metadata["id"] for i in instances] == [0, 1]
        assert [i.metadata["doc_id"] for i in instances] == ["doc-a", "doc-b"]
        assert instances[1].metadata["num_bytes"] == 100_000

    def test_requests_roll_over_the_whole_document_from_an_empty_prompt(self, loaded):
        task = get_task("paloma_ptb")
        requests = [task.format_request(instance) for instance in task.instances]
        assert all(r.request_type == RequestType.LOGLIKELIHOOD for r in requests)
        assert all(r.rolling and r.prompt == "" for r in requests)
        assert requests[1].continuations == ("y" * 100_000,)

    def test_test_split_reads_the_test_files(self, loaded):
        list(get_task("paloma_ptb", {"split": Split.TEST}).instances)
        assert loaded[0].data_files == "ptb/test/*"
        assert loaded[0].revision == paloma.PALOMA_REVISION

    def test_dolma_test_split_skips_its_empty_file(self, loaded):
        list(get_task("paloma_dolma-v1_5", {"split": Split.TEST}).instances)
        files = loaded[0].data_files
        assert isinstance(files, tuple) and len(files) == 5
        assert "dolma-v1_5/test/test_books.jsonl.gz" not in files

    def test_split_paloma_lacks_is_rejected(self, loaded):
        with pytest.raises(ValueError, match="no train split"):
            list(get_task("paloma_ptb", {"split": Split.TRAIN}).instances)


def test_suite_averages_the_sixteen_sources_equally():
    suite = get_suite("paloma")
    assert suite.aggregation == AggregationStrategy.AVERAGE
    assert suite.expand() == tuple(f"paloma_{subset}" for subset in paloma.PALOMA_SUBSETS)

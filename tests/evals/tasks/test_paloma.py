"""Tests for the Paloma perplexity tasks and the paloma macro bits-per-byte suite."""

import pytest

from olmo_eval.common.metrics import BPBMetricByteAvg
from olmo_eval.common.types import RequestType, Split
from olmo_eval.evals.suites import AggregationStrategy, get_suite
from olmo_eval.evals.tasks import paloma
from olmo_eval.evals.tasks.common import get_task, list_tasks

#: The window every test runs with; the tasks refuse to run without one.
WINDOW = {"max_length": 2048}


class TestRegistration:
    def test_one_task_per_paloma_source(self):
        registered = {name for name in list_tasks() if name.startswith("paloma_")}
        assert registered == {f"paloma_{subset}" for subset in paloma.PALOMA_SUBSETS}
        assert len(paloma.PALOMA_SUBSETS) == 16

    def test_tasks_pin_the_dataset_and_choose_files_by_split(self):
        source = get_task("paloma_c4_en").config.data_source
        assert source.path == "allenai/paloma"
        assert source.subset == "c4_en"
        assert source.revision == paloma.PALOMA_REVISION
        # The split's files are chosen at load time, not stored in the config.
        assert source.data_files is None

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
            {"id": "doc-a", "text": "short document", "subdomain": "nyt"},
            {"id": "doc-b", "text": "y" * 100_000},
        ]

        def load(_loader, source):
            sources.append(source)
            return iter(docs)

        monkeypatch.setattr(paloma.DataLoader, "load", load)
        return sources

    def test_each_document_is_one_instance_however_long(self, loaded):
        instances = list(get_task("paloma_ptb", WINDOW).instances)
        assert [i.gold_answer for i in instances] == ["short document", "y" * 100_000]
        assert [i.metadata["id"] for i in instances] == [0, 1]
        assert [i.metadata["doc_id"] for i in instances] == ["doc-a", "doc-b"]
        assert [i.metadata["subdomain"] for i in instances] == ["nyt", None]

    @pytest.mark.parametrize(
        ("overrides", "files"), [({}, "ptb/val/*"), ({"split": Split.TEST}, "ptb/test/*")]
    )
    def test_each_split_reads_its_own_files(self, loaded, overrides, files):
        list(get_task("paloma_ptb", {**WINDOW, **overrides}).instances)
        assert loaded[0].data_files == files
        assert loaded[0].split == "train"
        assert loaded[0].revision == paloma.PALOMA_REVISION

    def test_requests_roll_over_the_whole_document_from_an_empty_prompt(self, loaded):
        task = get_task("paloma_ptb", WINDOW)
        requests = [task.format_request(instance) for instance in task.instances]
        assert all(r.request_type == RequestType.LOGLIKELIHOOD for r in requests)
        assert all(r.rolling and r.prompt == "" for r in requests)
        assert requests[1].continuations == ("y" * 100_000,)
        assert all(r.max_length == 2048 for r in requests)

    def test_a_run_without_max_length_is_refused_before_loading(self, loaded):
        with pytest.raises(ValueError, match="need max_length"):
            list(get_task("paloma_ptb").instances)
        assert loaded == []

    def test_max_length_is_in_the_task_hash(self):
        from olmo_eval.common.types.base import compute_task_hash

        hashes = {
            compute_task_hash(get_task("paloma_ptb", {"max_length": n}).config.to_dict())
            for n in (8192, 65536)
        }
        assert len(hashes) == 2

    @pytest.mark.parametrize("override", [{"num_fewshot": 2}])
    def test_settings_that_do_not_apply_are_refused(self, loaded, override):
        task = get_task("paloma_ptb", {**WINDOW, **override})
        with pytest.raises(ValueError, match="raw text"):
            task.format_request(next(iter(task.instances)))

    def test_dolma_test_split_skips_its_empty_file(self, loaded):
        list(get_task("paloma_dolma-v1_5", {**WINDOW, "split": Split.TEST}).instances)
        files = loaded[0].data_files
        assert isinstance(files, tuple) and len(files) == 5
        assert "dolma-v1_5/test/test_books.jsonl.gz" not in files

    def test_split_paloma_lacks_is_rejected(self, loaded):
        with pytest.raises(ValueError, match="no train split"):
            list(get_task("paloma_ptb", {**WINDOW, "split": Split.TRAIN}).instances)


def test_bits_per_byte_divides_the_rolled_logprob_by_every_byte(monkeypatch):
    import math

    from olmo_eval.common.types import LMOutput, Response

    docs = [{"id": "a", "text": "héllo"}, {"id": "b", "text": "y" * 40}]
    monkeypatch.setattr(paloma.DataLoader, "load", lambda _loader, _source: iter(docs))
    task = get_task("paloma_ptb", WINDOW)
    responses = []
    for instance, logprob in zip(task.instances, (-3.0, -11.0), strict=True):
        request = task.format_request(instance)
        output = LMOutput(
            text=request.continuations[0],
            logprobs=[{"token": "t", "logprob": logprob}],
        )
        responses.append(Response(instance=instance, request=request, outputs=[output]))

    bpb = task.config.get_primary_metric().compute(responses)
    assert bpb == pytest.approx(14.0 / ((6 + 40) * math.log(2)))


def test_bits_per_byte_leaves_out_the_bytes_of_an_unscored_first_token(monkeypatch):
    import math

    from olmo_eval.common.types import LMOutput, Response

    monkeypatch.setattr(
        paloma.DataLoader, "load", lambda _loader, _source: iter([{"id": "a", "text": "hello"}])
    )
    task = get_task("paloma_ptb", WINDOW)
    [instance] = task.instances
    request = task.format_request(instance)
    # A tokenizer without BOS: "h" was context, the other four bytes were scored.
    output = LMOutput(
        text="hello", logprobs=[{"token": "t", "logprob": -2.0}], metadata={"scored_bytes": 4}
    )
    response = Response(instance=instance, request=request, outputs=[output])

    assert task.config.get_primary_metric().compute([response]) == pytest.approx(
        2.0 / (4 * math.log(2))
    )


def test_suite_averages_the_sources_equally():
    suite = get_suite("paloma")
    assert suite.aggregation == AggregationStrategy.AVERAGE
    assert suite.expand() == tuple(f"paloma_{subset}" for subset in paloma.PALOMA_SUBSETS)

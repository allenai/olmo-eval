"""Tests for the Paloma perplexity tasks and the paloma macro bits-per-byte suite."""

import pytest

from olmo_eval.common.metrics import BPBMetricByteAvg
from olmo_eval.common.types import RequestType, Split
from olmo_eval.evals.suites import AggregationStrategy, get_suite
from olmo_eval.evals.tasks import paloma
from olmo_eval.evals.tasks.common import get_task, list_tasks


class TestSplitText:
    def test_short_text_is_one_chunk(self):
        assert paloma.split_text("hello world", max_bytes=100) == ["hello world"]

    def test_empty_text_has_no_chunks(self):
        assert paloma.split_text("", max_bytes=100) == []

    def test_cuts_after_the_last_newline_in_the_window(self):
        text = "a" * 6 + "\n" + "b" * 6 + " " + "c" * 6
        assert paloma.split_text(text, max_bytes=10) == ["aaaaaa\n", "bbbbbb ", "cccccc"]

    def test_falls_back_to_a_space_when_the_newline_is_too_early(self):
        text = "a\n" + "b" * 5 + " " + "c" * 6
        assert paloma.split_text(text, max_bytes=10)[0] == "a\nbbbbb "

    def test_hard_cuts_text_without_whitespace(self):
        assert paloma.split_text("x" * 25, max_bytes=10) == ["x" * 10, "x" * 10, "x" * 5]

    def test_hard_cuts_never_split_a_character(self):
        # "é" is 2 bytes and "🦊" is 4, so a 5-byte window holds two "é" or one "🦊".
        assert paloma.split_text("é" * 5, max_bytes=5) == ["éé", "éé", "é"]
        assert paloma.split_text("🦊🦊", max_bytes=5) == ["🦊", "🦊"]

    def test_limit_counts_bytes_not_characters(self):
        text = "日本語のテキスト" * 100  # 3 bytes per character
        chunks = paloma.split_text(text, max_bytes=64)
        assert all(len(chunk.encode("utf-8")) <= 64 for chunk in chunks)
        assert "".join(chunks) == text
        assert len(chunks) >= len(text.encode("utf-8")) // 64

    @pytest.mark.parametrize("max_bytes", [4, 7, 64])
    def test_chunks_rejoin_to_the_text_and_respect_the_limit(self, max_bytes):
        text = "The quick brown fox\njumps over\n\nthe lazy dog. " * 40 + "naïve café 🦊"
        chunks = paloma.split_text(text, max_bytes=max_bytes)
        assert "".join(chunks) == text
        assert all(0 < len(chunk.encode("utf-8")) <= max_bytes for chunk in chunks)

    def test_rejects_a_limit_too_small_for_a_character(self):
        with pytest.raises(ValueError, match="at least 4"):
            paloma.split_text("abc", max_bytes=3)


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
            {"id": "doc-b", "text": "y" * (paloma.MAX_CHUNK_BYTES + 5)},
        ]

        def load(_loader, source):
            sources.append(source)
            return iter(docs)

        monkeypatch.setattr(paloma.DataLoader, "load", load)
        return sources

    def test_long_documents_become_several_chunks(self, loaded):
        instances = list(get_task("paloma_ptb").instances)
        assert [i.gold_answer for i in instances] == [
            "short document",
            "y" * paloma.MAX_CHUNK_BYTES,
            "y" * 5,
        ]
        assert [i.metadata["id"] for i in instances] == [0, 1, 2]
        assert [(i.metadata["doc_id"], i.metadata["chunk_index"]) for i in instances] == [
            ("doc-a", 0),
            ("doc-b", 0),
            ("doc-b", 1),
        ]
        assert instances[1].metadata["num_chunks"] == 2

    def test_requests_score_each_chunk_from_an_empty_prompt(self, loaded):
        task = get_task("paloma_ptb")
        request = task.format_request(next(iter(task.instances)))
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.prompt == ""
        assert request.continuations == ("short document",)

    def test_test_split_reads_the_test_files(self, loaded):
        list(get_task("paloma_ptb", {"split": Split.TEST}).instances)
        assert loaded[0].data_files == "ptb/test/*"
        assert loaded[0].revision == paloma.PALOMA_REVISION

    def test_split_paloma_lacks_is_rejected(self, loaded):
        with pytest.raises(ValueError, match="no train split"):
            list(get_task("paloma_ptb", {"split": Split.TRAIN}).instances)


def test_suite_averages_the_sixteen_sources_equally():
    suite = get_suite("paloma")
    assert suite.aggregation == AggregationStrategy.AVERAGE
    assert suite.expand() == tuple(f"paloma_{subset}" for subset in paloma.PALOMA_SUBSETS)

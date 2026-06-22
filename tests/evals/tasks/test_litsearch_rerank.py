"""Tests for the fixed-corpus LitSearch reranking task."""

import json

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.litsearch_rerank import (
    POOLS_PATH,
    parse_ranked_numbers,
    recall_at_k,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("litsearch_rerank")


def _doc(query="kd papers?", gold=(10, 20), candidate_ids=(10, 11, 20, 12)):
    return {
        "query_id": 7,
        "query": query,
        "gold_corpusids": list(gold),
        "candidates": [
            {"corpusid": cid, "title": f"Title {cid}", "abstract": f"Abstract {cid}"}
            for cid in candidate_ids
        ],
        "query_set": "inline_acl",
    }


def _response(instance: Instance, text: str) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
    )


class TestRegistration:
    def test_registered(self):
        assert "litsearch_rerank" in list_tasks()

    def test_metrics(self, task):
        assert {m.name for m in task.config.metrics} == {"recall@5", "recall@20"}
        assert task.config.get_primary_metric().name == "recall@5"

    def test_data_source_points_at_pools(self, task):
        assert task.config.data_source.path == "allenai/litsearch-rerank-pools"
        assert task.config.data_source.split == "train"


class TestParseRankedNumbers:
    def test_json_object(self):
        assert parse_ranked_numbers('{"ranked_papers": [3, 1, 12]}') == [3, 1, 12]

    def test_bare_list_in_text(self):
        assert parse_ranked_numbers("My ranking: [5, 2]") == [5, 2]

    def test_floats_coerced(self):
        assert parse_ranked_numbers("[1.0, 2.0]") == [1, 2]

    def test_no_list(self):
        assert parse_ranked_numbers("no list here") == []

    def test_empty(self):
        assert parse_ranked_numbers("") == []


class TestRecallAtK:
    def test_partial(self):
        assert recall_at_k({10, 20, 30}, [10, 99, 20], 5) == pytest.approx(2 / 3)

    def test_respects_cutoff(self):
        # gold 20 sits at rank 6, outside top-5.
        ranked = [1, 2, 3, 4, 5, 20]
        assert recall_at_k({20}, ranked, 5) == 0.0
        assert recall_at_k({20}, ranked, 20) == 1.0

    def test_no_gold(self):
        assert recall_at_k(set(), [1, 2], 5) == 0.0


class TestProcessDoc:
    def test_builds_instance(self, task):
        instance = task.process_doc(_doc(), index=3)
        assert instance is not None
        assert instance.question == "kd papers?"
        assert instance.metadata["gold_corpusids"] == [10, 20]
        assert instance.metadata["candidate_corpusids"] == [10, 11, 20, 12]
        assert instance.metadata["case_id"] == 7

    def test_skips_empty_query(self, task):
        assert task.process_doc(_doc(query=""), 0) is None

    def test_skips_no_gold(self, task):
        assert task.process_doc(_doc(gold=()), 0) is None


class TestFormatRequest:
    def test_numbered_candidate_list(self, task):
        instance = task.process_doc(_doc(), index=0)
        request = task.format_request(instance)
        content = request.messages[0]["content"]
        assert request.request_type == RequestType.CHAT
        assert "kd papers?" in content
        assert "[1] Title 10" in content
        assert "[3] Title 20" in content


@pytest.mark.skipif(not POOLS_PATH.exists(), reason="rerank pools artifact not built")
class TestBuiltArtifact:
    """Regression checks pinning the shipped BM25 pool artifact.

    Candidates are stored in BM25-ranked order, so the retriever baseline is
    recomputable here without re-downloading the corpus.
    """

    @pytest.fixture(scope="class")
    def rows(self):
        with POOLS_PATH.open() as handle:
            return [json.loads(line) for line in handle]

    def test_shape(self, rows):
        assert len(rows) == 597
        assert all(len(r["candidates"]) == 50 for r in rows)
        assert all(r["gold_corpusids"] for r in rows)

    def test_bm25_baseline(self, rows):
        def mean_recall(k):
            vals = []
            for r in rows:
                gold = set(r["gold_corpusids"])
                ranked = [c["corpusid"] for c in r["candidates"]]
                vals.append(recall_at_k(gold, ranked, k))
            return sum(vals) / len(vals)

        assert mean_recall(5) == pytest.approx(0.446, abs=1e-3)
        assert mean_recall(20) == pytest.approx(0.6018, abs=1e-3)

    def test_reranker_ceiling(self, rows):
        # Perfect reranker promotes every in-pool gold into top-k; gold <= 5/query.
        def mean_ceiling(k):
            vals = []
            for r in rows:
                gold = set(r["gold_corpusids"])
                pool = {c["corpusid"] for c in r["candidates"]}
                vals.append(min(len(gold & pool), k) / len(gold))
            return sum(vals) / len(vals)

        assert mean_ceiling(5) == pytest.approx(0.6671, abs=1e-3)
        assert mean_ceiling(20) == pytest.approx(0.6671, abs=1e-3)


class TestScoreResponses:
    @pytest.mark.anyio
    async def test_maps_numbers_and_scores(self, task):
        instance = task.process_doc(_doc(), index=0)
        # Candidate numbers 1->10, 3->20 are both gold (gold={10,20}).
        response = _response(instance, '{"ranked_papers": [1, 3, 2]}')

        await task.score_responses([response])

        assert response.scores["recall@5"] == 1.0
        assert response.scores["recall@20"] == 1.0
        assert response.outputs[0].metadata["litsearch_rerank_selected"] == [10, 20, 11]

    @pytest.mark.anyio
    async def test_topk_cutoff(self, task):
        # 7 candidates; gold is candidate #7, ranked 7th -> outside top-5, inside top-20.
        doc = _doc(gold=(70,), candidate_ids=(10, 20, 30, 40, 50, 60, 70))
        instance = task.process_doc(doc, index=0)
        response = _response(instance, "[1, 2, 3, 4, 5, 6, 7]")

        await task.score_responses([response])

        assert response.scores["recall@5"] == 0.0
        assert response.scores["recall@20"] == 1.0

    @pytest.mark.anyio
    async def test_out_of_range_and_empty(self, task):
        instance = task.process_doc(_doc(), index=0)
        response = _response(instance, "[99, 0, -1]")  # all out of range

        await task.score_responses([response])

        assert response.scores["recall@5"] == 0.0
        assert response.outputs[0].metadata["litsearch_rerank_selected"] == []

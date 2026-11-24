"""Unit tests for ResponseBatchScorer contract, Task integration, and runner batching."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.metrics import AccuracyMetric, PassAtKMetric
from olmo_eval.common.scorers import ExactMatchScorer, ResponseBatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.runners.asynq.results import process_results
from olmo_eval.runners.asynq.types import ResultItem, TaskTracker


class _DummyTask(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="A")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)


@dataclass(frozen=True)
class _RecordingBatchScorer(ResponseBatchScorer):
    name: str = "recording_batch"
    batch_size: int | None = 3
    group_by_category: bool = False
    recorded_batches: list[list[str]] = field(default_factory=list, compare=False, hash=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "batch_size": self.batch_size,
            "group_by_category": self.group_by_category,
        }

    def batch_key(self, response: Response) -> Hashable:
        if self.group_by_category:
            return response.instance.metadata.get("category")
        return None

    async def ascore_batch(
        self,
        responses: Sequence[Response],
        context: ScoringContext,
    ) -> Sequence[float | list[float | None] | tuple[float | None, ...] | None]:
        self.recorded_batches.append([r.instance.question for r in responses])
        results: list[float | list[float | None] | tuple[float | None, ...] | None] = []
        for response in responses:
            if any(o.text == "boom" for o in response.outputs):
                raise RuntimeError("batch judge failed")
            if any(o.text == "skip" for o in response.outputs):
                results.append(None)
                continue
            if len(response.outputs) > 1:
                results.append(
                    [
                        1.0 if o.text == response.instance.gold_answer else 0.0
                        for o in response.outputs
                    ]
                )
            else:
                text = response.outputs[0].text if response.outputs else ""
                results.append(1.0 if text == response.instance.gold_answer else 0.0)
        return results


def _make_response(question: str, gold: str, *outputs: str, category: str = "default") -> Response:
    instance = Instance(question=question, gold_answer=gold, metadata={"category": category})
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=question),
        outputs=[LMOutput(text=text) for text in outputs],
    )


class TestResponseBatchScorer:
    def test_sync_score_raises_runtime_error(self) -> None:
        scorer = _RecordingBatchScorer()
        with pytest.raises(RuntimeError, match="ascore_batch"):
            scorer.score(Instance(question="Q", gold_answer="A"), LMOutput(text="A"))

    def test_task_score_responses_chunks_by_batch_size_and_key(self) -> None:
        scorer = _RecordingBatchScorer(batch_size=2, group_by_category=True)
        metric = AccuracyMetric(scorer=scorer)
        task = _DummyTask(TaskConfig(name="t", data_source="test/d", metrics=(metric,)))

        responses = [
            _make_response("q1", "A", "A", category="cat1"),
            _make_response("q2", "A", "B", category="cat2"),
            _make_response("q3", "A", "A", category="cat1"),
            _make_response("q4", "A", "A", category="cat1"),
            _make_response("q5", "A", "A", category="cat2"),
        ]

        scored = asyncio.run(task.score_responses(responses))

        assert scorer.recorded_batches == [["q1", "q3"], ["q4"], ["q2", "q5"]]
        assert [r.scores["recording_batch"] for r in scored] == [1.0, 0.0, 1.0, 1.0, 1.0]
        assert scored[0].outputs[0].metadata["score:recording_batch"] == 1.0
        assert scored[1].outputs[0].metadata["score:recording_batch"] == 0.0

    def test_multi_output_scores_and_batch_error_isolation(self) -> None:
        batch_scorer = _RecordingBatchScorer(batch_size=2)
        sync_metric = AccuracyMetric(scorer=ExactMatchScorer)
        batch_metric = PassAtKMetric(k=1, scorer=batch_scorer)
        task = _DummyTask(
            TaskConfig(name="t", data_source="test/d", metrics=(sync_metric, batch_metric))
        )

        r1 = _make_response("q1", "4", "3", "4")
        r2 = _make_response("q2", "4", "4", "5")
        r3 = _make_response("q3", "4", "boom")

        scored = asyncio.run(task.score_responses([r1, r2, r3], context=ScoringContext()))

        # First batch [r1, r2] succeeds with per-output scores
        assert scored[0].scores["recording_batch"] == 1.0
        assert scored[0].outputs[0].metadata["score:recording_batch"] == 0.0
        assert scored[0].outputs[1].metadata["score:recording_batch"] == 1.0
        assert scored[1].scores["recording_batch"] == 1.0

        # Second batch [r3] fails in batch_scorer; error is recorded and sync scorer still succeeds
        assert "recording_batch" not in scored[2].scores
        assert scored[2].outputs[0].metadata["scoring_errors"]["recording_batch"] == {
            "phase": "context",
            "type": "RuntimeError",
            "message": "batch judge failed",
        }
        assert scored[2].scores["exact_match"] == 0.0

    def test_process_results_orders_batches_and_skips_hard_failures_and_restored(self) -> None:
        scorer = _RecordingBatchScorer(batch_size=3)
        metric = AccuracyMetric(scorer=scorer)
        task = _DummyTask(TaskConfig(name="t", data_source="test/d", metrics=(metric,)))

        total = 8
        # Instance 0 is already restored; instance 3 is a hard failure.
        # Remaining instances (1, 2, 4, 5, 6, 7) should form two batches of 3 in exact order:
        # [q1, q2, q4] and [q5, q6, q7].
        restored_resp = _make_response("q0", "A", "A")
        restored_resp.scores["recording_batch"] = 1.0
        restored_resp.outputs[0].metadata["score:recording_batch"] = 1.0

        result_queue: queue.Queue[ResultItem] = queue.Queue()
        items: dict[int, ResultItem] = {}
        for idx in range(1, total):
            if idx == 3:
                items[idx] = ResultItem(
                    model_name="m",
                    task_id="t",
                    instance_idx=idx,
                    instance=Instance(question=f"q{idx}", gold_answer="A"),
                    request=LMRequest(request_type=RequestType.COMPLETION, prompt=f"q{idx}"),
                    outputs=[],
                    error="generation failed",
                )
            else:
                resp = _make_response(f"q{idx}", "A", "A")
                items[idx] = ResultItem(
                    model_name="m",
                    task_id="t",
                    instance_idx=idx,
                    instance=resp.instance,
                    request=resp.request,
                    outputs=resp.outputs,
                )

        # Arrival order is scrambled
        for idx in [5, 2, 1, 4, 3, 7, 6]:
            result_queue.put(items[idx])

        results = asyncio.run(
            process_results(
                trackers={
                    "t": TaskTracker(model_name="m", spec="t", task=task, total_instances=total)
                },
                result_queue=result_queue,  # type: ignore[arg-type]
                workers=[],
                scoring_context=ScoringContext(scoring_concurrency=2),
                scoring_concurrency=2,
                total_tasks=1,
                total_instances=total - 1,  # 7 queued (1 restored)
                model_name="m",
                save_predictions=False,
                write_predictions_fn=None,
                save_requests=False,
                write_requests_fn=None,
                restored={"t": {0: restored_resp}},
            )
        )

        assert scorer.recorded_batches == [["q1", "q2", "q4"], ["q5", "q6", "q7"]]
        assert results["t"].num_instances == 7
        assert results["t"].instances_failed == 1
        assert results["t"].metrics["accuracy"]["recording_batch"] == pytest.approx(1.0)

    def test_unbounded_batch_size_scores_entire_bucket_in_one_call(self) -> None:
        scorer = _RecordingBatchScorer(batch_size=None)
        metric = AccuracyMetric(scorer=scorer)
        task = _DummyTask(TaskConfig(name="t", data_source="test/d", metrics=(metric,)))

        result_queue: queue.Queue[ResultItem] = queue.Queue()
        for idx in range(5):
            resp = _make_response(f"q{idx}", "A", "A")
            result_queue.put(
                ResultItem(
                    model_name="m",
                    task_id="t",
                    instance_idx=idx,
                    instance=resp.instance,
                    request=resp.request,
                    outputs=resp.outputs,
                )
            )

        results = asyncio.run(
            process_results(
                trackers={"t": TaskTracker(model_name="m", spec="t", task=task, total_instances=5)},
                result_queue=result_queue,  # type: ignore[arg-type]
                workers=[],
                scoring_context=ScoringContext(scoring_concurrency=2),
                scoring_concurrency=2,
                total_tasks=1,
                total_instances=5,
                model_name="m",
                save_predictions=False,
                write_predictions_fn=None,
                save_requests=False,
                write_requests_fn=None,
            )
        )

        assert scorer.recorded_batches == [["q0", "q1", "q2", "q3", "q4"]]
        assert results["t"].metrics["accuracy"]["recording_batch"] == pytest.approx(1.0)

    def test_none_score_in_batch_records_incomplete_error_for_that_item_only(self) -> None:
        scorer = _RecordingBatchScorer(batch_size=3)
        metric = AccuracyMetric(scorer=scorer)
        task = _DummyTask(TaskConfig(name="t", data_source="test/d", metrics=(metric,)))

        r1 = _make_response("q1", "A", "A")
        r2 = _make_response("q2", "A", "skip")
        r3 = _make_response("q3", "A", "A")

        scored = asyncio.run(task.score_responses([r1, r2, r3]))

        assert scored[0].scores["recording_batch"] == 1.0
        assert "recording_batch" not in scored[1].scores
        assert scored[1].outputs[0].metadata["scoring_errors"]["recording_batch"] == {
            "phase": "context",
            "type": "ScoringIncompleteError",
            "message": "_RecordingBatchScorer returned no score",
            "infrastructure": "true",
        }
        assert scored[2].scores["recording_batch"] == 1.0

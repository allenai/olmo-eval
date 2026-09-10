"""strip_thinking is applied by the runner, so tasks overriding score_responses are covered."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence

from olmo_eval.common.execution.environment import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.runners.asynq.preparation import finalize_task
from olmo_eval.runners.asynq.types import TaskTracker


class CustomScoringTask(Task):
    """Scores output.text directly, bypassing the base class pipeline."""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="4")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    async def score_responses(
        self, responses: Sequence[Response], context: ScoringContext | None = None
    ) -> Sequence[Response]:
        for response in responses:
            for output in response.outputs:
                output.extracted_answer = output.text
            response.scores["custom"] = float(
                all(o.text == response.instance.gold_answer for o in response.outputs)
            )
        return responses


def test_finalize_task_strips_before_custom_score_responses() -> None:
    config = TaskConfig(name="custom", data_source="test/dataset", strip_thinking=True)
    task = CustomScoringTask(config)
    instance = Instance(question="Q", gold_answer="4")
    response = Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text="<think>3? no, 4.</think>4")],
    )
    tracker = TaskTracker(
        model_name="m",
        spec="custom",
        task=task,
        total_instances=1,
        completed_count=1,
        responses={0: response},
    )

    result = asyncio.run(finalize_task(tracker))

    assert result.error is None
    assert response.outputs[0].text == "4"
    assert response.scores["custom"] == 1.0
    assert result.predictions[0]["model_output"][0]["original_text"] == "<think>3? no, 4.</think>4"

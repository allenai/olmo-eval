from __future__ import annotations

from datetime import UTC, datetime

from olmo_eval.common.types import EvalResult, StoredTaskResult
from olmo_eval.storage.backends.postgres.queries import QueryHelper


class _FakeExperimentRepository:
    def save(self, result: EvalResult) -> int:
        self.saved_result = result
        return 17


class _FakeInstancePredictionRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def save_instances(
        self,
        experiment_pk: int,
        task_hash: str,
        instances: list[dict[str, object]],
        experiment_group: str = "",
        chunk_size: int = 1000,
    ) -> None:
        self.calls.append(
            {
                "experiment_pk": experiment_pk,
                "task_hash": task_hash,
                "instances": instances,
                "experiment_group": experiment_group,
                "chunk_size": chunk_size,
            }
        )


def test_save_with_instances_normalizes_old_style_mc_predictions_before_insert() -> None:
    helper = QueryHelper(session=None)  # type: ignore[arg-type]
    helper.experiment_repo = _FakeExperimentRepository()
    helper.instance_repo = _FakeInstancePredictionRepository()

    result = EvalResult(
        experiment_id="exp-123",
        model_name="model-a",
        backend_name="vllm",
        timestamp=datetime(2026, 5, 3, tzinfo=UTC),
        experiment_name="exp-123",
        workspace="workspace",
        author="tester",
        git_ref="abc123",
        revision="main",
        experiment_group="exp-123",
        tasks=[
            StoredTaskResult(
                task_name="basic_skills_coding:rc:olmo3base",
                task_hash="task-hash-1",
                metrics={"accuracy": {"logprob": 1.0}},
            )
        ],
    )

    instances_by_task = {
        "basic_skills_coding:rc:olmo3base": [
            {
                "native_id": "doc-0",
                "label": 1,
                "instance_metrics": {
                    "accuracy": 1.0,
                    "logprob": {"logprob": -1.0},
                },
                "model_output": [
                    {
                        "text": "A",
                        "sum_logits": -4.0,
                        "num_tokens": 1,
                        "num_tokens_all": 1,
                        "num_chars": 1,
                        "is_greedy": False,
                    },
                    {
                        "text": "B",
                        "sum_logits": -1.0,
                        "num_tokens": 1,
                        "num_tokens_all": 1,
                        "num_chars": 1,
                        "is_greedy": False,
                    },
                ],
            }
        ]
    }

    experiment_pk = helper.save_with_instances(result, instances_by_task)

    assert experiment_pk == 17
    calls = helper.instance_repo.calls  # type: ignore[attr-defined]
    assert len(calls) == 1
    saved_instances = calls[0]["instances"]
    assert isinstance(saved_instances, list)
    saved_instance = saved_instances[0]
    assert saved_instance["instance_metrics"]["logprob"]["logprob"] == -1.0
    assert saved_instance["instance_metrics"]["accuracy"]["accuracy"] == 1.0
    assert saved_instance["instance_metrics"]["accuracy"]["logprob"] == 1.0


def test_save_with_instances_normalizes_flat_instance_metrics_when_task_metrics_unavailable() -> (
    None
):
    helper = QueryHelper(session=None)  # type: ignore[arg-type]
    helper.experiment_repo = _FakeExperimentRepository()
    helper.instance_repo = _FakeInstancePredictionRepository()

    result = EvalResult(
        experiment_id="exp-123",
        model_name="model-a",
        backend_name="vllm",
        timestamp=datetime(2026, 5, 3, tzinfo=UTC),
        experiment_name="exp-123",
        workspace="workspace",
        author="tester",
        git_ref="abc123",
        revision="main",
        experiment_group="exp-123",
        tasks=[
            StoredTaskResult(
                task_name="custom_task",
                task_hash="task-hash-1",
                metrics={"accuracy": {"exact_match": 1.0}},
            )
        ],
    )

    instances_by_task = {
        "custom_task": [
            {
                "native_id": "doc-0",
                "instance_metrics": {"acc": 1.0, "f1": 0.5},
            }
        ]
    }

    helper.save_with_instances(result, instances_by_task)

    calls = helper.instance_repo.calls  # type: ignore[attr-defined]
    assert len(calls) == 1
    saved_instances = calls[0]["instances"]
    assert isinstance(saved_instances, list)
    saved_instance = saved_instances[0]
    assert saved_instance["instance_metrics"] == {
        "acc": {"acc": 1.0},
        "f1": {"f1": 0.5},
    }

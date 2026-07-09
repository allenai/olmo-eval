"""Tests for async runner task preparation."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.runners.asynq import preparation


class _PreparationTask(Task):
    def __init__(self, config: TaskConfig, instances: list[Instance]):
        super().__init__(config)
        self._instances = instances

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._instances

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text


def test_prepare_task_items_filters_restrict_native_ids_after_limit(monkeypatch):
    instances = [Instance(question=f"q{i}") for i in range(5)]
    task = _PreparationTask(TaskConfig(name="prep"), instances)
    monkeypatch.setattr(preparation, "get_task", lambda spec: task)

    prepared_task, items = preparation.prepare_task_items(
        spec="prep",
        model_name="model",
        overrides={"restrict_native_ids": ["doc_1", "doc_3"]},
    )

    assert prepared_task.config.restrict_native_ids == frozenset({"doc_1", "doc_3"})
    assert [item.instance.question for item in items] == ["q1", "q3"]
    assert [item.request.prompt for item in items] == ["q1", "q3"]


def test_prepare_task_items_warns_when_restrict_native_ids_match_nothing(
    monkeypatch,
    caplog,
):
    instances = [Instance(question=f"q{i}") for i in range(2)]
    task = _PreparationTask(TaskConfig(name="prep"), instances)
    monkeypatch.setattr(preparation, "get_task", lambda spec: task)

    with caplog.at_level(logging.WARNING, logger=preparation.logger.name):
        _, items = preparation.prepare_task_items(
            spec="prep",
            model_name="model",
            overrides={"restrict_native_ids": ["missing"]},
        )

    assert items == []
    assert "restrict_native_ids matched 0 of 2 instances" in caplog.text
    assert "check the ids / limit / seed" in caplog.text
    assert "restrict_native_ids: matched 0/1 requested ids" in caplog.text
    assert "missing" in caplog.text


def test_prepare_task_items_warns_when_restrict_native_ids_partially_match(
    monkeypatch,
    caplog,
):
    instances = [
        Instance(question="q0", metadata={"id": "keep"}),
        Instance(question="q1"),
        Instance(question="q2"),
    ]
    task = _PreparationTask(TaskConfig(name="prep"), instances)
    monkeypatch.setattr(preparation, "get_task", lambda spec: task)

    with caplog.at_level(logging.WARNING, logger=preparation.logger.name):
        _, items = preparation.prepare_task_items(
            spec="prep",
            model_name="model",
            overrides={"restrict_native_ids": ["keep", "doc_1", "missing"]},
        )

    assert [item.instance.question for item in items] == ["q0", "q1"]
    assert "restrict_native_ids: matched 2/3 requested ids" in caplog.text
    assert "missing" in caplog.text
    assert "matched 0 of" not in caplog.text

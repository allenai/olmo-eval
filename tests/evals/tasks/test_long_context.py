"""Tests for the shared long-context task base."""

import pickle
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.types import LMOutput, RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.long_context import (
    LongContextTask,
    long_context_sampling_params,
)

_TABLE = {
    "kv__4096": {"tag": "recall"},
    "kv_chat__4096": {"tag": "recall", "use_chat_template": True},
}
_ROWS = [
    {"context": "a=1 b=2", "key": "a", "answer": ["1"], "extra": "x"},
    {"context": "c=3", "key": "c", "answer": "3"},
]


class _KvTask(LongContextTask):
    name_prefix = "test_"
    task_table = _TABLE

    def _load_dataset(self) -> dict[str, Any]:
        return {
            "data": _ROWS,
            "user_template": "{context}\nWhat is {key}?",
            "system_template": "The value of {key} is",
        }

    def scoring_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {"extra": doc["extra"]} if "extra" in doc else {}


def _task(name: str = "test_kv__4096", **config: Any) -> _KvTask:
    return _KvTask(TaskConfig(name=name, **config))


def test_name_is_parsed_into_type_and_context_size():
    task = _task()
    assert (task.task_name, task.task_type, task.context_size) == ("kv__4096", "kv", 4096)
    assert task.task_config is _TABLE["kv__4096"]


def test_prompt_is_the_user_template_then_the_filled_answer_prefix():
    task = _task()
    first, second = list(task.instances)
    request = task.format_request(first)
    assert request.request_type == RequestType.COMPLETION
    assert request.prompt == "a=1 b=2\nWhat is a?\nThe value of a is"
    assert first.metadata["all_gold_answers"] == ["1"]
    assert first.metadata["extra"] == "x"
    assert "all_gold_answers" not in second.metadata
    assert [i.metadata["id"] for i in (first, second)] == [0, 1]


def test_chat_template_tasks_leave_off_the_answer_prefix():
    task = _task("test_kv_chat__4096")
    first = next(iter(task.instances))
    assert task.format_request(first).prompt == "a=1 b=2\nWhat is a?"


def test_instances_are_cached():
    task = _task()
    assert list(task.instances) == list(task.instances)
    assert task._instances_cache is not None


def test_ppl_formatter_gets_list_answers_joined():
    task = _task(formatter=PPLFormatter())
    first = next(iter(task.instances))
    assert "1" in task.format_request(first).continuations[0]


def test_extract_answer_is_the_raw_generation():
    assert _task().extract_answer(LMOutput(text=" The value is 1.")) == " The value is 1."


def test_sampling_params_are_greedy():
    params = long_context_sampling_params(50, ("\n",))
    assert (params.temperature, params.top_p, params.max_tokens) == (0.0, 1.0, 50)
    assert params.stop_sequences == ("\n",)


def test_registered_long_context_tasks_pickle():
    from olmo_eval.evals.tasks.common.registry import get_task

    for name in ("ruler_niah_s_1__4096", "helmet_kilt_nq__4096", "helmet_ruler_niah_mv__8192"):
        task = get_task(name)
        restored = pickle.loads(pickle.dumps(task))
        assert type(restored) is type(task)
        assert restored.config.to_dict() == task.config.to_dict()

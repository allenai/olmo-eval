"""Tests for MMLU task variants."""

from olmo_eval.common.types import LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task


def test_chat_variant_uses_fewshot_chat_and_extracts_answer(monkeypatch):
    import olmo_eval.evals.tasks  # noqa: F401

    task = get_task("mmlu_abstract_algebra:chat")
    instance = task.process_doc(
        {
            "question": "Which option is correct?",
            "choices": ["one", "two", "three", "four"],
            "answer": 2,
        }
    )
    assert instance is not None
    monkeypatch.setattr(task, "get_fewshot", lambda: [instance])

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.messages is not None
    assert [message["role"] for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert request.messages[2]["content"] == "ANSWER: C"
    assert task.extract_answer(LMOutput(text="Reasoning...\nANSWER: C")) == "C"


def test_chat_suite_covers_all_mmlu_subjects():
    import olmo_eval.evals  # noqa: F401
    from olmo_eval.evals.suites import get_suite
    from olmo_eval.evals.tasks.mmlu import MMLU_SUBJECTS

    tasks = get_suite("mmlu:chat").expand()

    assert len(tasks) == len(MMLU_SUBJECTS)
    assert all(task.endswith(":chat") for task in tasks)

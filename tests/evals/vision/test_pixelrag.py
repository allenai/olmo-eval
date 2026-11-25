"""PixelRAG reader benchmark: the official protocol, the judge call, exact match, and task
identity. Offline: the dataset is never loaded and the judge is faked."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.scorers import llm_judge
from olmo_eval.common.scoring_failures import count_scoring_failures
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common import judge as judge_module
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks import pixelrag
from olmo_eval.evals.vision.benchmarks.pixelrag import (
    QUERY_IMAGE_NOTE,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_QUERY_IMAGE,
    reader_messages,
    verify_local_copy,
)
from olmo_eval.evals.vision.scoring.pixelrag import (
    JUDGE_SYSTEM_MESSAGE,
    JUDGE_WORLDVQA_PROMPT,
    SIMPLEQA_GRADER_TEMPLATE,
    is_exact_match,
    judge_prompt,
    parse_simpleqa_grade,
    parse_worldvqa_label,
    strip_think,
)
from olmo_eval.runners.processing.utils import compute_task_hash

IMG = {"type": "image"}


@pytest.fixture(autouse=True)
def judge_env(monkeypatch):
    """Each task's own judge, retried without waiting."""
    monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)

    async def sleep(_delay):
        return None

    monkeypatch.setattr(judge_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(llm_judge, "retry_with_backoff", lambda call, **_: call())


def _instance(question: str, answer: str, gold_answers: list[str]) -> Instance:
    return Instance(
        question=question,
        gold_answer=answer,
        metadata={
            "id": question,
            "problem": question,
            "answer": answer,
            "gold_answers": gold_answers,
            "instructions": "",
            "has_query_image": False,
            "tile_ranks": (0, 1, 2),
            "images": None,
        },
    )


def _response(task, instance: Instance, text: str) -> Response:
    return Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(text=text)]
    )


class TestOfficialProtocol:
    def test_judge_prompts_are_the_official_files(self):
        sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
        assert sha(JUDGE_WORLDVQA_PROMPT) == (
            "d0ded5cf964faf1acda862a0a55eeae2151695d59fb4d861f8919a8faf1f1af4"
        )
        assert sha(SIMPLEQA_GRADER_TEMPLATE) == (
            "063c04ea798f91c4706c050b66a721afc77a458b12d8f0d72286d4b0eee8a3ee"
        )

    def test_reader_turn_is_question_tiles_then_instruction(self):
        system, user = reader_messages("Who won?", "Answer briefly.", 3, query_image=False)
        assert system == {"role": "system", "content": SYSTEM_PROMPT}
        assert user["content"] == [
            {"type": "text", "text": "Who won?"},
            IMG,
            IMG,
            IMG,
            {"type": "text", "text": "Answer briefly."},
        ]

    def test_query_image_turn_is_one_text_block_then_the_images(self):
        system, user = reader_messages("What is this?", "Exact Answer: ...", 2, query_image=True)
        assert system["content"] == SYSTEM_PROMPT_QUERY_IMAGE
        text = f"Question: What is this?\n\n{QUERY_IMAGE_NOTE}\n\nExact Answer: ..."
        assert user["content"] == [{"type": "text", "text": text}, IMG, IMG, IMG]

    def test_verdict_parsing_follows_the_grader(self):
        assert parse_simpleqa_grade("B") == "incorrect"
        assert parse_simpleqa_grade("no letter") == "unattempted"
        assert parse_worldvqa_label("Evaluation: ...\nLabel: Correct") == "correct"
        assert parse_worldvqa_label("this is incorrect") == "incorrect"
        assert parse_worldvqa_label("garbled") == "incorrect"

    def test_exact_match_and_think_stripping_follow_the_grader(self):
        assert strip_think("<think>hmm</think> Paris") == "Paris"
        assert is_exact_match("Exact Answer: The Beatles", ["beatles"])
        assert not is_exact_match("Beatles band", ["The Beatles"])
        prompt = judge_prompt(
            "simpleqa", question="Q?", ground_truth="A", response="<think>x</think>A"
        )
        assert "Question: Q?\nGold target: A\nPredicted answer: A\n" in prompt


class _FakeClient:
    def __init__(self, reply="Label: Correct", fail_on=None):
        self.calls: list[dict] = []
        self.reply = reply
        self.fail_on = fail_on
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on and self.fail_on in kwargs["messages"][-1]["content"]:
            raise RuntimeError("judge unavailable")
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    clients: list[_FakeClient] = []

    def install(**kwargs) -> _FakeClient:
        client = _FakeClient(**kwargs)
        clients.append(client)
        monkeypatch.setattr(llm_judge, "create_async_openai_client", lambda _creds: client)
        return client

    return install


class TestJudgeCall:
    def test_default_call_is_the_official_judge(self, fake_client):
        client = fake_client()
        task = get_task("pixelrag_nq")
        response = _response(task, _instance("Who won?", "Any of: X", ["X"]), "X")
        asyncio.run(task.score_responses([response]))
        (call,) = client.calls
        assert call["model"] == "gpt-4.1-2025-04-14"
        assert call["max_tokens"] == 1000
        assert call["temperature"] == 0.0
        assert call["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM_MESSAGE}
        assert "Question: Who won?" in call["messages"][1]["content"]
        assert response.scores["accuracy"] == 1.0

    def test_call_uses_the_judge_the_config_records(self, fake_client):
        client = fake_client()
        overrides = {"judge_model": "gpt-4o-2024-08-06", "judge_max_tokens": 77}
        task = get_task("pixelrag_nq", config_overrides={**overrides, "judge_temperature": 0.5})
        assert (task.config.judge_model, task.config.judge_max_tokens) == (
            "gpt-4o-2024-08-06",
            77,
        )
        asyncio.run(task.score_responses([_response(task, _instance("Q", "A", ["A"]), "A")]))
        (call,) = client.calls
        assert (call["model"], call["max_tokens"], call["temperature"]) == (
            "gpt-4o-2024-08-06",
            77,
            0.5,
        )


class TestExactMatchIsIndependentOfTheJudge:
    def test_a_failed_judge_call_does_not_drop_the_response_from_exact_match(self, fake_client):
        fake_client(fail_on="Model Answer: wrong")
        task = get_task("pixelrag_nq")
        right = _response(task, _instance("Q1", "Any of: right", ["right"]), "right")
        wrong = _response(task, _instance("Q2", "Any of: other", ["other"]), "wrong")
        responses = asyncio.run(task.score_responses([right, wrong]))
        metrics = task.compute_metrics(responses)
        # Exact match sees both answers; the judge only the one it graded.
        assert metrics["exact_match"]["pixelrag_exact_match"] == 0.5
        assert metrics["accuracy"]["pixelrag"] == 1.0
        failures = count_scoring_failures(responses, tuple(task._get_scorers()))
        assert failures.instances_unscored == 1
        assert "pixelrag" in wrong.outputs[0].metadata["scoring_errors"]


class TestTaskIdentity:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("revision", "0" * 40),
            ("reader_top_k", 2),
            ("system_prompt", "Be brief."),
            ("query_image_note", "Look."),
        ],
    )
    def test_data_and_reader_settings_are_in_the_task_hash(self, key, value):
        config = get_task("pixelrag_nq").config
        changed = replace(config, task_settings={**config.task_settings, key: value})
        assert compute_task_hash(changed.to_dict()) != compute_task_hash(config.to_dict())

    def test_the_judge_is_in_the_task_hash(self):
        config = get_task("pixelrag_nq").config
        changed = replace(config, judge_model="gpt-4o-2024-08-06")
        assert compute_task_hash(changed.to_dict()) != compute_task_hash(config.to_dict())

    def test_requests_use_the_configured_prompts(self):
        task = get_task("pixelrag_nq")
        task.config = replace(
            task.config, task_settings={**task.config.task_settings, "system_prompt": "Be brief."}
        )
        request = task.format_request(_instance("Q", "A", ["A"]))
        assert request.messages[0] == {"role": "system", "content": "Be brief."}


class TestLocalCopy:
    @pytest.fixture
    def copy(self, tmp_path, monkeypatch):
        folder = tmp_path / "nq"
        folder.mkdir()
        (folder / "test-00000-of-00001.parquet").write_bytes(b"pinned")
        pinned = hashlib.sha256(b"pinned").hexdigest()
        monkeypatch.setattr(pixelrag, "DATA_MANIFEST", {"nq": {"test-00000-of-00001": pinned}})
        verify_local_copy.cache_clear()
        yield tmp_path
        verify_local_copy.cache_clear()

    def test_a_matching_copy_is_accepted(self, copy):
        verify_local_copy(str(copy), "nq", pixelrag.HF_REVISION)

    def test_a_changed_file_is_refused(self, copy):
        (copy / "nq" / "test-00000-of-00001.parquet").write_bytes(b"edited")
        with pytest.raises(ValueError, match="sha256"):
            verify_local_copy(str(copy), "nq", pixelrag.HF_REVISION)

    def test_a_missing_or_extra_file_is_refused(self, copy):
        (copy / "nq" / "test-00001-of-00001.parquet").write_bytes(b"extra")
        with pytest.raises(ValueError, match="not in the pinned revision"):
            verify_local_copy(str(copy), "nq", pixelrag.HF_REVISION)
        (copy / "nq" / "test-00000-of-00001.parquet").unlink()
        with pytest.raises(ValueError, match="missing"):
            verify_local_copy(str(copy), "nq", pixelrag.HF_REVISION)

    def test_a_copy_cannot_stand_in_for_another_revision(self, copy):
        with pytest.raises(ValueError, match="revision"):
            verify_local_copy(str(copy), "nq", "0" * 40)

"""Tests for the PopQA task."""

from __future__ import annotations

import unittest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.popqa import PopQAContainsScorer


def _make_instance(aliases: list[str]) -> Instance:
    return Instance(
        question="Q: What is X? A:",
        gold_answer=aliases[0],
        metadata={"id": "test", "aliases": aliases},
    )


class TestPopQATask(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("popqa")
        self.assertEqual(task.request_type, RequestType.COMPLETION)
        self.assertEqual(task.config.num_fewshot, 15)
        self.assertEqual(task.config.sampling_params.max_tokens, 15)
        self.assertEqual(task.config.sampling_params.stop_sequences, ("\n\n",))

    def test_chat_variant(self) -> None:
        task = get_task("popqa:chat")
        self.assertEqual(task.request_type, RequestType.CHAT)
        self.assertIsNone(task.config.sampling_params.stop_sequences)
        self.assertEqual(task.config.sampling_params.temperature, 0.6)

    def test_process_doc(self) -> None:
        task = get_task("popqa")
        doc = {
            "id": 1,
            "prop_id": 22,
            "question": "What is George Rankin's occupation?",
            "obj": "politician",
            "possible_answers": '["politician", "political leader"]',
            "s_pop": 100,
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        self.assertEqual(instance.question, "Q: What is George Rankin's occupation? A:")
        self.assertEqual(instance.gold_answer, "politician")
        self.assertEqual(instance.metadata["aliases"], ["politician", "political leader"])
        self.assertEqual(instance.metadata["subject_popularity"], 100)

    def test_fewshot_prompt_format(self) -> None:
        task = get_task("popqa")
        instance = _make_instance(["answer"])
        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.COMPLETION)
        examples = request.prompt.split("\n\n")
        self.assertEqual(len(examples), 16)  # 15 shots + the question
        self.assertTrue(examples[0].startswith("Q: "))
        self.assertIn(" A: ", examples[0])
        self.assertTrue(request.prompt.endswith("Q: What is X? A:"))

    def test_chat_fewshot_in_single_message(self) -> None:
        task = get_task("popqa:chat")
        request = task.format_request(_make_instance(["answer"]))
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(len(request.messages[0]["content"].split("\n\n")), 16)


class TestPopQAContainsScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = PopQAContainsScorer()
        self.instance = _make_instance(["United States of America", "USA"])

    def _score(self, text: str) -> float:
        return self.scorer.score(self.instance, LMOutput(text=text))

    def test_verbatim_match(self) -> None:
        self.assertEqual(self._score(" United States of America"), 1.0)

    def test_any_alias_matches(self) -> None:
        self.assertEqual(self._score("I believe it is the USA."), 1.0)

    def test_lowercase_alias_in_response(self) -> None:
        # Response is checked against the lowered alias too.
        self.assertEqual(self._score("the united states of america"), 1.0)

    def test_miss(self) -> None:
        self.assertEqual(self._score("Canada"), 0.0)

    def test_empty_response(self) -> None:
        self.assertEqual(self._score(""), 0.0)


if __name__ == "__main__":
    unittest.main()

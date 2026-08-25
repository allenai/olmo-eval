"""Tests for the MMLU chain-of-thought tasks."""

from __future__ import annotations

import unittest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.mmlu import MMLU_SUBJECTS
from olmo_eval.evals.tasks.mmlu_cot import MMLUCoTExactMatchScorer

_DOC = {
    "question": "What is 2 + 2?",
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
}


class TestMMLUCoTRegistration(unittest.TestCase):
    def test_all_subjects_registered(self) -> None:
        self.assertEqual(len(MMLU_SUBJECTS), 57)
        for subject in MMLU_SUBJECTS:
            task = get_task(f"mmlu_{subject}:cot")
            self.assertEqual(task.request_type, RequestType.CHAT)
            self.assertEqual(task.config.num_fewshot, 0)

    def test_sampling_matches_reasoning_regime(self) -> None:
        params = get_task("mmlu_anatomy:cot").config.sampling_params
        self.assertEqual(params.temperature, 0.6)
        self.assertEqual(params.top_p, 0.95)
        self.assertIsNone(params.max_tokens)

    def test_suite_covers_every_subject(self) -> None:
        self.assertEqual(len(get_suite("mmlu:cot").tasks), 57)


class TestMMLUCoTPrompt(unittest.TestCase):
    def test_prompt_shape(self) -> None:
        task = get_task("mmlu_abstract_algebra:cot")
        instance = task.process_doc(dict(_DOC), index=0)
        assert instance is not None
        self.assertTrue(
            instance.question.startswith(
                "The following are multiple choice questions about abstract algebra."
            )
        )
        self.assertIn("Question: What is 2 + 2?\n A. 3\n B. 4\n C. 5\n D. 6\n", instance.question)
        self.assertEqual(instance.gold_answer, "B")

    def test_single_user_message(self) -> None:
        task = get_task("mmlu_anatomy:cot")
        instance = task.process_doc(dict(_DOC), index=0)
        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")

    def test_skips_malformed_docs(self) -> None:
        task = get_task("mmlu_anatomy:cot")
        self.assertIsNone(task.process_doc({"question": "", "choices": ["a"], "answer": 0}))
        self.assertIsNone(task.process_doc({"question": "q", "choices": [], "answer": 0}))
        self.assertIsNone(task.process_doc({"question": "q", "choices": ["a"], "answer": "B"}))


class TestMMLUCoTScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = MMLUCoTExactMatchScorer()
        self.instance = Instance(question="q", gold_answer="B", metadata={})

    def _score(self, text: str) -> float:
        return self.scorer.score(self.instance, LMOutput(text=text))

    def test_requested_phrasing(self) -> None:
        self.assertEqual(self._score("Reasoning.\n\nTherefore, the answer is: B"), 1.0)

    def test_parenthesized_and_boxed(self) -> None:
        self.assertEqual(self._score("Therefore, the answer is: (B)"), 1.0)
        self.assertEqual(self._score("\\boxed{B}"), 1.0)

    def test_alternate_phrasings(self) -> None:
        self.assertEqual(self._score("So the answer is B."), 1.0)
        self.assertEqual(self._score("The correct answer is: (B)"), 1.0)

    def test_case_insensitive(self) -> None:
        self.assertEqual(self._score("the ANSWER is (b)"), 1.0)

    def test_wrong_and_empty(self) -> None:
        self.assertEqual(self._score("Therefore, the answer is: C"), 0.0)
        self.assertEqual(self._score(""), 0.0)

    def test_template_priority_beats_position(self) -> None:
        """The requested phrasing outranks a later, weaker one.

        The cascade tries templates in priority order, so "Therefore, the
        answer is: A" wins over a subsequent "the answer is B" — verified
        to match the reference implementation's extraction.
        """
        self.assertEqual(
            self._score("Therefore, the answer is: A. Wait, no — the answer is B"), 0.0
        )
        self.assertEqual(
            self._score("Some reasoning. The answer is A. Therefore, the answer is: B"), 1.0
        )


if __name__ == "__main__":
    unittest.main()

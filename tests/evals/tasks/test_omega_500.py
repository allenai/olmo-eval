"""Tests for the OMEGA-500 task."""

from __future__ import annotations

import unittest

from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.omega_500 import _FLEX, _STRICT


def _make_instance(gold: str) -> Instance:
    return Instance(question="q", gold_answer=gold, metadata={"id": "test"})


class TestOmega500Task(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("omega_500")
        self.assertEqual(task.request_type, RequestType.CHAT)
        self.assertEqual(task.config.num_fewshot, 0)
        self.assertEqual(task.config.sampling_params.temperature, 0.6)
        self.assertEqual(task.config.sampling_params.top_p, 0.95)
        metric_names = {m.name for m in task.config.metrics}
        self.assertEqual(metric_names, {"exact_match", "exact_match_flex"})
        primary = task.config.get_primary_metric()
        assert primary is not None
        self.assertEqual(primary.name, "exact_match_flex")

    def test_process_doc(self) -> None:
        task = get_task("omega_500")
        doc = {
            "index": 3,
            "dataset": "algebra_func_intersection",
            "ground_truth": "42",
            "messages": [
                {
                    "role": "user",
                    "content": "Solve for x."
                    "\n\nPresent the answer in LaTex format: \\boxed{Your answer}",
                }
            ],
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        self.assertEqual(instance.gold_answer, "42")
        self.assertTrue(instance.question.startswith("Solve for x.\n\nShow your work"))
        self.assertNotIn("Present the answer in LaTex format", instance.question)
        self.assertIn(
            'conclude with "Therefore, the final answer is \\boxed{answer}."',
            instance.question,
        )

        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")


class TestOmegaScorers(unittest.TestCase):
    def _scores(self, gold: str, text: str) -> tuple[float, float]:
        instance = _make_instance(gold)
        return (
            _STRICT.score(instance, LMOutput(text=text)),
            _FLEX.score(instance, LMOutput(text=text)),
        )

    def test_requested_format(self) -> None:
        text = "Work...\n\nTherefore, the final answer is \\boxed{42}."
        self.assertEqual(self._scores("42", text), (1.0, 1.0))

    def test_prefix_answer_counts_for_both(self) -> None:
        # A recognized answer prefix scores above the format cutoff.
        self.assertEqual(self._scores("42", "The answer is 42."), (1.0, 1.0))

    def test_bare_answer_demoted_to_flex(self) -> None:
        # A boxed answer without the requested concluding phrase falls
        # below the format cutoff: flex only.
        self.assertEqual(self._scores("42", "So we get \\boxed{42} somewhere"), (0.0, 1.0))

    def test_case_insensitive(self) -> None:
        text = "Therefore, the final answer is \\boxed{TRUE}."
        self.assertEqual(self._scores("true", text), (1.0, 1.0))

    def test_comma_normalization(self) -> None:
        text = "Therefore, the final answer is \\boxed{1,234}."
        self.assertEqual(self._scores("1234", text), (1.0, 1.0))

    def test_last_format_match_wins(self) -> None:
        text = (
            "Therefore, the final answer is \\boxed{42}. "
            "Wait: Therefore, the final answer is \\boxed{41}."
        )
        self.assertEqual(self._scores("42", text), (0.0, 0.0))

    def test_wrong_answer(self) -> None:
        text = "Therefore, the final answer is \\boxed{41}."
        self.assertEqual(self._scores("42", text), (0.0, 0.0))

    def test_empty_response(self) -> None:
        self.assertEqual(self._scores("42", ""), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

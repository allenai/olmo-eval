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
        # The dataset's own boxed suffix is stripped; the CoT suffix is
        # applied by the formatter, not stored on the instance.
        self.assertEqual(instance.question, "Solve for x.")

        request = task.format_request(instance)
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")
        content = request.messages[0]["content"]
        self.assertTrue(content.startswith("Solve for x.\n\nShow your work"))
        self.assertNotIn("Present the answer in LaTex format", content)
        self.assertIn(
            'conclude with "Therefore, the final answer is \\boxed{answer}."',
            content,
        )


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

    def test_format_correct_recorded_in_metadata(self) -> None:
        output = LMOutput(text="Therefore, the final answer is \\boxed{42}.")
        _STRICT.score(_make_instance("42"), output)
        self.assertEqual(output.metadata["answer_format_correct"], 1.0)
        output2 = LMOutput(text="So we get \\boxed{42} somewhere")
        _STRICT.score(_make_instance("42"), output2)
        self.assertEqual(output2.metadata["answer_format_correct"], 0.2)

    def test_flex_scorer_name_derived(self) -> None:
        """flex=True with no explicit name must not collide with strict."""
        from olmo_eval.evals.tasks.omega_500 import OmegaExactMatchScorer

        self.assertEqual(OmegaExactMatchScorer(flex=True).name, "exact_match_flex")
        self.assertEqual(OmegaExactMatchScorer().name, "exact_match")


if __name__ == "__main__":
    unittest.main()

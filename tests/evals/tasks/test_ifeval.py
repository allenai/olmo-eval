"""Tests for the single-turn IFEval task."""

from __future__ import annotations

import unittest
from typing import Any

from olmo_eval.common.scorers import IFEvalScorer
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task

# A real google/IFEval row shape: kwargs arrive null-padded to the dataset's
# full schema, with one entry per instruction id.
_DOC: dict[str, Any] = {
    "key": 1000,
    "prompt": (
        "Write a resume for a software engineer. Your answer must contain "
        "exactly 3 bullet points, such as: * This is a point."
    ),
    "instruction_id_list": ["detectable_format:number_bullet_lists"],
    "kwargs": [{"num_bullets": 3, "num_placeholders": None, "language": None}],
}


def _instance(task: Any) -> Instance:
    instance = task.process_doc(dict(_DOC), index=0)
    assert instance is not None
    return instance


class TestIFEvalTask(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("ifeval")
        self.assertEqual(task.request_type, RequestType.CHAT)
        self.assertEqual(task.config.sampling_params.max_tokens, 2048)
        self.assertEqual(task.config.sampling_params.temperature, 0.0)
        self.assertFalse(task.config.sampling_params.do_sample)

    def test_metrics(self) -> None:
        """All four IFEval aggregations, with loose prompt-level primary."""
        task = get_task("ifeval")
        self.assertEqual(
            {metric.name for metric in task.config.metrics},
            {
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            },
        )
        primary = task.config.get_primary_metric()
        assert primary is not None
        self.assertEqual(primary.name, "prompt_level_loose_acc")

    def test_process_doc_drops_null_padded_kwargs(self) -> None:
        """Absent kwargs arrive as None and must not reach the verifier."""
        instance = _instance(get_task("ifeval"))
        self.assertEqual(instance.metadata["kwargs"], [{"num_bullets": 3}])
        self.assertEqual(
            instance.metadata["instruction_id_list"],
            ["detectable_format:number_bullet_lists"],
        )
        self.assertEqual(instance.metadata["key"], 1000)
        self.assertEqual(instance.metadata["prompt"], _DOC["prompt"])
        self.assertIsNone(instance.gold_answer)

    def test_process_doc_without_instructions(self) -> None:
        task = get_task("ifeval")
        instance = task.process_doc({"key": 7, "prompt": "Hello."}, index=0)
        assert instance is not None
        self.assertEqual(instance.metadata["instruction_id_list"], [])
        self.assertEqual(instance.metadata["kwargs"], [])

    def test_format_request_is_single_user_message(self) -> None:
        task = get_task("ifeval")
        request = task.format_request(_instance(task))
        self.assertEqual(request.request_type, RequestType.CHAT)
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0]["role"], "user")
        self.assertEqual(request.messages[0]["content"], _DOC["prompt"])

    def test_extract_answer_is_verbatim(self) -> None:
        task = get_task("ifeval")
        self.assertEqual(task.extract_answer(LMOutput(text="  spaced  ")), "  spaced  ")


class TestIFEvalScoring(unittest.TestCase):
    """The task's metadata must drive the scorer end to end."""

    def setUp(self) -> None:
        self.instance = _instance(get_task("ifeval"))
        self.scorer = IFEvalScorer()

    def _score(self, text: str) -> tuple[float, dict[str, Any]]:
        output = LMOutput(text=text)
        score = self.scorer.score(self.instance, output)
        return score, output.metadata["ifeval"]

    def test_compliant_response(self) -> None:
        score, result = self._score("* Led backend\n* Shipped v2\n* Cut latency")
        self.assertEqual(score, 1.0)
        self.assertEqual(result["strict"], [True])
        self.assertEqual(result["loose"], [True])

    def test_noncompliant_response(self) -> None:
        score, result = self._score("I led the backend team and shipped v2.")
        self.assertEqual(score, 0.0)
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])

    def test_no_instructions_scores_zero(self) -> None:
        task = get_task("ifeval")
        instance = task.process_doc({"key": 7, "prompt": "Hello."}, index=0)
        assert instance is not None
        self.assertEqual(self.scorer.score(instance, LMOutput(text="hi")), 0.0)


if __name__ == "__main__":
    unittest.main()

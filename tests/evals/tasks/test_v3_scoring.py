"""Tests for the ``v3`` variants of MMLU-Pro and HLE.

``v3`` is ``v2`` (an unfinished reasoning trace is no answer) with one scoring rule
added per task: MMLU-Pro rejects an answer naming several letters, and HLE reads
calibration only from the outputs that state a confidence.
"""

from __future__ import annotations

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common import get_task, register_sampling_variant, registry
from olmo_eval.evals.tasks.hle import states_confidence
from olmo_eval.evals.tasks.mmlu_pro import names_several_choices
from tests.evals.tasks.judge_stubs import scripted_judge, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

OPEN = {"prompt_opens_think": True, "think_closed": False}
V3_SPECS = ("mmlu_pro_law:cot:v3", "hle:text:v3", "hle:text:verified:v3")


def _response(instance: Instance, *outputs: tuple[str, dict]) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text, metadata=dict(metadata)) for text, metadata in outputs],
    )


async def _score(task, responses: list[Response]) -> list[Response]:
    """Strip traces and score, as the runner does."""
    task.strip_thinking_traces(responses)
    return list(await task.score_responses(responses, ScoringContext()))


def _metrics(task, responses: list[Response]) -> dict[str, float]:
    return {
        name: next(iter(values.values()))
        for name, values in task.compute_metrics(responses).items()
    }


class TestRegistration:
    @pytest.mark.parametrize("spec", V3_SPECS)
    def test_v3_keeps_the_v2_trace_handling(self, spec):
        v3, v2 = get_task(spec).config, get_task(spec.removesuffix(":v3") + ":v2").config
        assert v3.strip_unclosed_thinking is True
        assert v3.thinking_prefilled == "auto"
        assert v3.strip_thinking == v2.strip_thinking
        assert v3.metrics != v2.metrics

    @pytest.mark.parametrize("spec", V3_SPECS)
    def test_a_sampling_variant_can_be_formed_from_v3(self, spec):
        sampled = register_sampling_variant(spec, "test_t07", temperature=0.7, max_tokens=64)
        try:
            config = get_task(sampled).config
            assert config.sampling_params.max_tokens == 64
            assert config.metrics == get_task(spec).config.metrics
        finally:
            # Leave the registry as it was: other tests enumerate every registered spec.
            for variants in registry._variants.values():
                for name in [name for name in variants if name.endswith("test_t07")]:
                    del variants[name]

    @pytest.mark.parametrize("spec", V3_SPECS)
    def test_an_output_without_evidence_is_not_scored(self, spec):
        from olmo_eval.common.scorers.execution import ScoringIncompleteError

        response = _response(
            Instance(question="q", gold_answer="A"),
            ("The answer is (A).", {"prompt_opens_think": True}),
        )

        with pytest.raises(ScoringIncompleteError, match="no evidence"):
            get_task(spec).strip_thinking_traces([response])

    def test_the_suite_holds_the_v3_specs(self):
        specs = get_suite("mmlu_pro:cot:v3").expand()
        assert len(specs) == 14
        assert all(spec.endswith(":cot:v3") for spec in specs)


class TestMMLUProSeveralLetters:
    @pytest.mark.parametrize(
        ("text", "several"),
        [
            ("Therefore, the answer is (C, E).", True),
            ("Therefore, the answer is (I) and (J)", True),
            ("**Therefore, the answer is (A), (C), and (J).**", True),
            ("The answer is (A or B)", True),
            ("The answer is C and E", True),
            ("The answer is C, E", True),
            ("ANSWER: C, E", True),
            ("Answer: (C) and (E)", True),
            ("The answer is: C or E.", True),
            ("Answer: C OR E.", True),
            ("The answer is A and B.", True),
            # The letters of an uppercase join are not choices.
            ("The answer is A AND A.", False),
            ("The answer is (A) AND (A).", False),
            ("Answer: D OR D", False),
            ("Answer: C AND E", True),
            ("The answer is (C) Or (E).", True),
            ("The answer is C, And E.", True),
            ("The answer is **C** and **E**.", True),
            ("So \\boxed{C, E}", True),
            ("So \\boxed{\\text{C, E}}", True),
            ("The answer is (I) and (J), which are both right.", True),
            ("Therefore, the answer is (A).", False),
            ("Therefore, the answer is (A) and I am sure.", False),
            ("The answer is A and I am sure of it.", False),
            ("The answer is A AND I AM SURE.", False),
            ("The answer is C, I believe.", False),
            ("The answer is C, a good choice.", False),
            ("The answer is C and Earth is round.", False),
            ("Answer: A. Both B and C fail.", False),
            ("\\boxed{C}", False),
            ("The answer is B or, failing that, nothing.", False),
            ("Therefore, the answer is (B), not (C).", False),
            ("Therefore, the answer is (B), because (A, C) are wrong.", False),
            ("Therefore, the answer is (A), (A).", False),
            ("Options (A, B) are wrong.\nTherefore, the answer is (D).", False),
            ("Therefore, the answer is (C, E).\nOn reflection, the answer is (C).", False),
            ("No statement, just (A, B).", False),
        ],
    )
    def test_names_several_choices(self, text, several):
        assert names_several_choices(text) is several

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "answer",
        ["Therefore, the answer is (C, E).", "Answer: C OR E.", "The answer is: E or C."],
    )
    async def test_several_letters_are_no_answer_in_v3_only(self, answer):
        def response() -> Response:
            return _response(Instance(question="q", gold_answer="C"), ("hm</think>" + answer, OPEN))

        v3, v2 = get_task("mmlu_pro_law:cot:v3"), get_task("mmlu_pro_law:cot:v2")

        new = await _score(v3, [response()])
        old = await _score(v2, [response()])

        assert new[0].scores["exact_match"] == 0.0
        assert new[0].outputs[0].extracted_answer is None
        assert new[0].outputs[0].metadata["answer_format_correct"] == 0.0
        # v2 reads one of the listed letters, as cot does.
        assert old[0].outputs[0].extracted_answer in {"C", "E"}

    @pytest.mark.anyio
    async def test_single_answers_and_unfinished_traces_score_as_in_v2(self):
        texts = [
            ("is it B? no, (C) fits the facts", OPEN),
            ("hm</think>Therefore, the answer is (C).", OPEN),
            ("hm</think>Therefore, the answer is (B), not (C).", OPEN),
        ]

        def responses() -> list[Response]:
            return [_response(Instance(question="q", gold_answer="C"), text) for text in texts]

        new = await _score(get_task("mmlu_pro_law:cot:v3"), responses())
        old = await _score(get_task("mmlu_pro_law:cot:v2"), responses())

        assert [r.scores["exact_match"] for r in new] == [0.0, 1.0, 0.0]
        assert [r.scores["exact_match"] for r in new] == [r.scores["exact_match"] for r in old]


class TestHLEStatedConfidence:
    @pytest.mark.parametrize(
        ("answer", "stated"),
        [
            ("Explanation: x\nAnswer: 42\nConfidence: 85%", True),
            ("Answer: 42\n**Confidence:** 0.9", True),
            ("Answer: 42\nconfidence level - 70", True),
            ("Answer: 42. My confidence is about 60%.", True),
            ("Answer: 42. I am 85% confident.", True),
            ("Answer: 42. I'm 85 percent sure.", True),
            ("I have 70 percent confidence in this.", True),
            ("Answer: 42 (confidence of 0.8)", True),
            ("Answer: {42}\nConfidence: {90%}", True),
            ("### Answer:\n42\n\n### Confidence:\n100%", True),
            ("**Answer:**  \n42\n\n**Confidence:**  \n85%", True),
            ("Confidence is high (90%) based on the task requirements.", True),
            ("Answer: 42\nConfidence: high (80%)", True),
            ("My answer comes with 90% confidence.", True),
            # A statistical statement, not the model's own confidence.
            ("With 95% confidence the bound holds.\nAnswer: 4", False),
            ("The bound holds with 95% confidence.\nAnswer: 4", False),
            ("The confidence in this result is strong.\nAnswer: 42", False),
            ("The final answer is \\boxed{2}", False),
            ("Answer: 42\nConfidence: high", False),
            # A statistic, not how sure the model is.
            ("The 95% confidence interval is 3 to 5.\nAnswer: 4", False),
            ("A confidence interval of 95% covers 3 to 5.\nAnswer: 4", False),
            ("Significant at the 95% confidence level.\nAnswer: 4", False),
            ("The confidence band (2, 5) excludes zero.\nAnswer: 4", False),
            # Both: the stated line still counts.
            ("The 95% confidence interval is 3 to 5.\nAnswer: 4\nConfidence: 70%", True),
        ],
    )
    def test_states_confidence(self, answer, stated):
        assert states_confidence(answer) is stated

    def _instance(self) -> Instance:
        return Instance(question="q", gold_answer="42", metadata={"category": "Math"})

    @pytest.mark.anyio
    async def test_calibration_is_read_from_stated_confidences(self, monkeypatch):
        task = get_task("hle:text:verified:v3")

        def reply(correct: str, confidence: str) -> str:
            return f"extracted_final_answer: 42\ncorrect: {correct}\nconfidence: {confidence}"

        # The judge reads a missing confidence as 100, as its prompt directs.
        judge, calls = scripted_judge(reply("yes", "60"), reply("no", "100"), reply("no", "100"))
        use_judge(monkeypatch, judge)
        responses = [
            _response(self._instance(), ("x</think>Answer: 42\nConfidence: 60%", OPEN)),
            _response(self._instance(), ("x</think>Answer: 41", OPEN)),
            _response(
                self._instance(),
                ("x</think>The 95% confidence interval is 3 to 5.\nAnswer: 4", OPEN),
            ),
            # Cut off inside the trace: no judge call, wrong at an imputed 100%.
            _response(self._instance(), ("still working, confidence: 90", OPEN)),
        ]

        scored = await _score(task, responses)

        assert len(calls) == 3
        stated = [r.outputs[0].metadata["judge_result"]["confidence_stated"] for r in scored]
        assert stated == [True, False, False, False]
        metrics = _metrics(task, scored)
        assert metrics["accuracy"] == pytest.approx(1 / 4)
        # One stated confidence: 60% on a correct answer.
        assert metrics["calibration_error_stated"] == pytest.approx(0.4)
        assert metrics["confidence_stated_rate"] == pytest.approx(1 / 4)
        assert "calibration_error_all_bins" not in metrics

    @pytest.mark.anyio
    async def test_calibration_is_left_out_when_no_output_states_a_confidence(self, monkeypatch):
        task = get_task("hle:text:verified:v3")
        judge, _ = scripted_judge("extracted_final_answer: 41\ncorrect: no\nconfidence: 100")
        use_judge(monkeypatch, judge)

        scored = await _score(task, [_response(self._instance(), ("x</think>Answer: 41", OPEN))])

        metrics = _metrics(task, scored)
        # 0.0 would read as perfect calibration.
        assert "calibration_error_stated" not in metrics
        assert metrics["confidence_stated_rate"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_v2_and_the_original_keep_their_metrics_and_judge_result(self):
        for spec in ("hle:text:verified", "hle:text:verified:v2"):
            names = [metric.name for metric in get_task(spec).config.metrics]
            assert "calibration_error_all_bins" in names
            assert "calibration_error_stated" not in names

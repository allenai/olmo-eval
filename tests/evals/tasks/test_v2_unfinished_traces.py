"""Tests for the ``v2`` variants, which treat an unfinished reasoning trace as no answer.

A chat template that opens ``<think>`` in the prompt leaves no opening tag in the
generation, so a trace that ran to the token limit looks like an answer. Providers record
such a prompt on each output; ``PREFILLED`` is that record.
"""

from __future__ import annotations

import json

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks import arena_hard
from olmo_eval.evals.tasks.common import get_task, register_sampling_variant, registry
from olmo_eval.evals.tasks.hle import states_confidence
from olmo_eval.evals.tasks.mmlu_pro import names_several_choices
from tests.evals.tasks.judge_stubs import scripted_judge, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

PREFILLED = {"prompt_opens_think": True}
V2_SPECS = (
    "wmdp:mcq:v2",
    "mmlu_pro_law:cot:v2",
    "zebralogic:chat:v2",
    "frontierscience_olympiad:v2",
    "frontierscience_olympiad:verified:v2",
    "frontierscience_research:v2",
    "arena_hard_v2:hard_prompt:v2",
    "arena_hard_v2:creative_writing:v2",
    "omniscience:judge:v2",
    "hle:text:v2",
    "hle:text:verified:v2",
)


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
    @pytest.mark.parametrize("spec", V2_SPECS)
    def test_v2_reads_the_provider_record_and_the_original_does_not(self, spec):
        config = get_task(spec).config
        assert config.strip_unclosed_thinking is True
        assert config.thinking_prefilled == "auto"

        original = get_task(spec.removesuffix(":v2")).config
        assert original.strip_unclosed_thinking is False
        assert original.thinking_prefilled is False

    @pytest.mark.parametrize("spec", V2_SPECS)
    def test_a_sampling_variant_can_be_formed_from_v2(self, spec):
        sampled = register_sampling_variant(spec, "test_t07", temperature=0.7, max_tokens=64)
        try:
            config = get_task(sampled).config
            params = config.sampling_params
            assert (params.temperature, params.max_tokens) == (0.7, 64)
            assert config.thinking_prefilled == "auto"
            assert config.metrics == get_task(spec).config.metrics
        finally:
            # Leave the registry as it was: other tests enumerate every registered spec.
            for variants in registry._variants.values():
                for name in [name for name in variants if name.endswith("test_t07")]:
                    del variants[name]

    def test_v2_alone_is_the_same_measurement_as_the_stacked_spec(self):
        # A variant is also a spec of its base task; it must not be a meaningless one.
        for stacked, alone in (
            ("zebralogic:chat:v2", "zebralogic:v2"),
            ("omniscience:judge:v2", "omniscience:v2"),
        ):
            assert get_task(alone).config == get_task(stacked).config

    def test_suites_hold_the_v2_specs(self):
        assert len(get_suite("mmlu_pro:cot:v2").expand()) == 14
        assert all(spec.endswith(":cot:v2") for spec in get_suite("mmlu_pro:cot:v2").expand())
        assert list(get_suite("arena_hard_v2:v2").expand()) == [
            "arena_hard_v2:hard_prompt:v2",
            "arena_hard_v2:creative_writing:v2",
        ]
        assert list(get_suite("frontierscience:v2").expand()) == [
            "frontierscience_olympiad:v2",
            "frontierscience_research:v2",
        ]


class TestWMDP:
    def _responses(self) -> list[Response]:
        def instance(category: str = "wmdp-bio") -> Instance:
            return Instance(
                question="q",
                choices=("w", "x", "y", "z"),
                gold_answer="A",
                metadata={"subcategory": category},
            )

        return [
            # Cut off while weighing the options: the trace mentions the gold letter.
            _response(instance("wmdp-cyber"), ("maybe (A), or (B)", PREFILLED)),
            _response(instance(), ("weighing (B)</think>ANSWER: A", PREFILLED)),
            _response(instance(), ("weighing (A)</think>ANSWER: B", PREFILLED)),
            _response(instance(), ("thinking</think>I cannot say.", PREFILLED)),
        ]

    @pytest.mark.anyio
    async def test_unanswered_questions_are_left_out_and_counted(self):
        task = get_task("wmdp:mcq:v2")
        scored = await _score(task, self._responses())

        metrics = _metrics(task, scored)
        # One answered question gives the gold answer, one does not.
        assert metrics["accuracy"] == 0.5
        assert metrics["no_answer_rate"] == 0.5
        assert metrics["parsing_error"] == 2
        assert metrics["subcategory__wmdp-bio"] == 0.5
        # A subset with no answered question has no accuracy; 0.0 would read as unsafe.
        assert "subcategory__wmdp-cyber" not in metrics
        accuracy = task.config.primary_metric
        assert [accuracy.compute_instance(r) for r in scored] == [None, 0.0, 1.0, None]
        assert scored[0].outputs[0].metadata["unclosed_thinking"] is True

    @pytest.mark.anyio
    async def test_accuracy_is_left_out_when_no_question_was_answered(self):
        task = get_task("wmdp:mcq:v2")
        scored = await _score(task, self._responses()[:1])

        metrics = _metrics(task, scored)
        assert "accuracy" not in metrics
        assert metrics["no_answer_rate"] == 1.0

    @pytest.mark.anyio
    async def test_the_original_counts_the_same_questions_as_safe(self):
        task = get_task("wmdp:mcq")
        scored = await _score(task, self._responses())

        # The unfinished trace is read for a letter and the refusal counts as safe.
        assert _metrics(task, scored)["accuracy"] == 0.75
        assert [r.scores["wmdp_multiple_choice"] for r in scored] == [1.0, 0.0, 1.0, 1.0]


class TestMMLUPro:
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
            ("The answer is **C** and **E**.", True),
            ("So \\boxed{C, E}", True),
            ("So \\boxed{\\text{C, E}}", True),
            ("The answer is (I) and (J), which are both right.", True),
            ("Therefore, the answer is (A).", False),
            ("The answer is A and I am sure of it.", False),
            ("The answer is C, I believe.", False),
            ("The answer is C, a good choice.", False),
            ("The answer is C and Earth is round.", False),
            ("Answer: A. Both B and C fail.", False),
            ("\\boxed{C}", False),
            ("The answer is B or, failing that, nothing.", False),
            ("Therefore, the answer is (A) and I am sure.", False),
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
    async def test_unfinished_traces_and_several_letters_are_not_answers(self):
        def instance() -> Instance:
            return Instance(question="q", gold_answer="C")

        texts = [
            ("is it B? no, (C) fits the facts", PREFILLED),
            ("hm</think>Therefore, the answer is (C, E).", PREFILLED),
            ("hm</think>Therefore, the answer is (C).", PREFILLED),
        ]
        v2, original = get_task("mmlu_pro_law:cot:v2"), get_task("mmlu_pro_law:cot")

        new = await _score(v2, [_response(instance(), text) for text in texts])
        old = await _score(original, [_response(instance(), text) for text in texts])

        assert [r.scores["exact_match"] for r in new] == [0.0, 0.0, 1.0]
        assert [o.extracted_answer for r in new for o in r.outputs] == [None, None, "C"]
        # The original reads "C" out of the unfinished trace and out of the list.
        assert [r.scores["exact_match"] for r in old] == [1.0, 1.0, 1.0]


class TestZebraLogic:
    SOLUTION = {"House 1": {"Name": "Eric"}, "House 2": {"Name": "Peter"}}

    def _response(self, text: str) -> Response:
        instance = Instance(
            question="q", metadata={"solution_table": self.SOLUTION, "total_cells": 2}
        )
        return _response(instance, (text, PREFILLED))

    @pytest.mark.anyio
    async def test_the_solution_is_parsed_from_the_answer_alone(self):
        answer = json.dumps({"reasoning": "r", "solution": self.SOLUTION})
        # One unmatched brace in the trace keeps the upstream parser from closing the JSON.
        finished = "the set {Eric, Peter ... \n</think>\n" + answer
        # A draft inside a trace that never closed is not a solution.
        unfinished = "draft: " + answer + " but clue 3 says"
        v2, original = get_task("zebralogic:chat:v2"), get_task("zebralogic:chat")

        new = await _score(v2, [self._response(finished), self._response(unfinished)])
        old = await _score(original, [self._response(finished), self._response(unfinished)])

        assert [(r.scores["parsed"], r.scores["puzzle_accuracy"]) for r in new] == [
            (1.0, 1.0),
            (0.0, 0.0),
        ]
        assert [(r.scores["parsed"], r.scores["puzzle_accuracy"]) for r in old] == [
            (0.0, 0.0),
            (1.0, 1.0),
        ]


class TestFrontierScience:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("spec", "scores"),
        [
            ("frontierscience_olympiad:v2", {"accuracy": 0.0}),
            ("frontierscience_research:v2", {"success_rate": 0.0, "rubric_score": 0.0}),
        ],
    )
    async def test_no_visible_answer_scores_zero_without_the_judge(self, spec, scores, monkeypatch):
        task = get_task(spec)
        judge, calls = scripted_judge()
        use_judge(monkeypatch, judge)
        instance = Instance(question="Find T.", gold_answer="42", metadata={"subject": "physics"})
        response = _response(instance, ("so T is 42? FINAL ANSWER\n42 seems", PREFILLED))

        scored = await _score(task, [response])

        assert calls == []
        assert {key: scored[0].scores[key] for key in scores} == scores
        assert scored[0].outputs[0].metadata["judge_result"] == {
            "scorer": "frontierscience_judge",
            "no_visible_answer": True,
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_a_finished_answer_still_goes_to_the_judge(self, monkeypatch):
        task = get_task("frontierscience_olympiad:v2")
        judge, calls = scripted_judge("VERDICT: CORRECT")
        use_judge(monkeypatch, judge)
        instance = Instance(question="Find T.", gold_answer="42", metadata={"subject": "physics"})
        response = _response(instance, ("work</think>FINAL ANSWER\n42", PREFILLED))

        scored = await _score(task, [response])

        assert len(calls) == 1 and "The attempted answer: 42" in calls[0]
        assert scored[0].scores["accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_the_original_sends_the_unfinished_trace_to_the_judge(self, monkeypatch):
        task = get_task("frontierscience_olympiad")
        judge, calls = scripted_judge("VERDICT: CORRECT")
        use_judge(monkeypatch, judge)
        instance = Instance(question="Find T.", gold_answer="42", metadata={"subject": "physics"})

        scored = await _score(task, [_response(instance, ("so T is 42? maybe", PREFILLED))])

        assert "so T is 42? maybe" in calls[0]
        assert scored[0].scores["accuracy"] == 1.0


class TestArenaHard:
    @pytest.fixture(autouse=True)
    def _word_tokens(self, monkeypatch):
        monkeypatch.setattr(arena_hard, "count_tokens", lambda text: len(text.split()))

    def _instance(self) -> Instance:
        return Instance(
            question="Write X.",
            metadata={
                "id": "q1",
                "subcategory": "coding",
                "baseline_model": "o3-mini-2025-01-31",
                "baseline_answer": "The baseline answer.",
            },
        )

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "metadata",
        [
            PREFILLED,
            # A server with a reasoning parser: the trace hit the limit before closing.
            {**PREFILLED, "think_close_hidden": True, "finish_reason": "length"},
        ],
    )
    async def test_no_visible_answer_loses_both_games_without_the_judge(
        self, metadata, monkeypatch
    ):
        task = get_task("arena_hard_v2:hard_prompt:v2")
        judge, calls = scripted_judge()
        use_judge(monkeypatch, judge)
        trace = "Okay, the user wants X. " * 50

        scored = await _score(task, [_response(self._instance(), (trace, metadata))])

        assert calls == []
        assert scored[0].scores["win_rate"] == 0.0
        assert scored[0].outputs[0].metadata["judge_result"] == {
            "scorer": "arena_hard_judge",
            "baseline_model": "o3-mini-2025-01-31",
            "no_visible_answer": True,
            "battle_outcomes": [0.0] * 6,
            "style_features": [0.0] * 4,
            "parse_error": False,
        }

    @pytest.mark.anyio
    async def test_an_answer_the_server_separated_from_its_trace_is_judged(self, monkeypatch):
        task = get_task("arena_hard_v2:hard_prompt:v2")
        judge, calls = scripted_judge("[[A=B]]", "[[A=B]]")
        use_judge(monkeypatch, judge)
        metadata = {
            **PREFILLED,
            "think_close_hidden": True,
            "reasoning_separated": True,
            "finish_reason": "length",
        }

        scored = await _score(task, [_response(self._instance(), ("A partial answ", metadata))])

        assert len(calls) == 2
        assert scored[0].scores["win_rate"] == 0.5

    @pytest.mark.anyio
    async def test_the_original_sends_the_trace_to_the_judge(self, monkeypatch):
        task = get_task("arena_hard_v2:hard_prompt")
        judge, calls = scripted_judge("[[B>A]]", "[[B>A]]")
        use_judge(monkeypatch, judge)
        trace = "Okay, the user wants X."

        scored = await _score(task, [_response(self._instance(), (trace, PREFILLED))])

        assert all(trace in prompt for prompt in calls)
        assert scored[0].scores["win_rate"] == 0.5


class TestOmniscience:
    @pytest.mark.anyio
    async def test_an_unfinished_trace_is_not_attempted_without_the_judge(self, monkeypatch):
        task = get_task("omniscience:judge:v2")
        judge, calls = scripted_judge()
        use_judge(monkeypatch, judge)
        instance = Instance(question="q", gold_answer="g", metadata={"domain": "Law", "topic": "t"})

        scored = await _score(task, [_response(instance, ("We are asked which", PREFILLED))])

        assert calls == []
        assert scored[0].outputs[0].metadata["judge_result"] == {
            "grade": "NOT_ATTEMPTED",
            "empty_response": True,
            "raw_judge_response": None,
        }


class TestHLE:
    @pytest.mark.parametrize(
        ("answer", "stated"),
        [
            ("Explanation: x\nAnswer: 42\nConfidence: 85%", True),
            ("Answer: 42\n**Confidence:** 0.9", True),
            ("I have 70 percent confidence.\nconfidence level - 70", True),
            ("The final answer is \\boxed{2}", False),
            ("Answer: 42\nConfidence: high", False),
        ],
    )
    def test_states_confidence(self, answer, stated):
        assert states_confidence(answer) is stated

    @pytest.mark.anyio
    async def test_calibration_is_read_from_stated_confidences(self, monkeypatch):
        task = get_task("hle:text:verified:v2")

        def reply(correct: str, confidence: str) -> str:
            return f"extracted_final_answer: 42\ncorrect: {correct}\nconfidence: {confidence}"

        # The judge reads a missing confidence as 100, as its prompt directs.
        judge, calls = scripted_judge(reply("yes", "60"), reply("no", "100"))
        use_judge(monkeypatch, judge)

        def instance() -> Instance:
            return Instance(question="q", gold_answer="42", metadata={"category": "Math"})

        responses = [
            _response(instance(), ("x</think>Answer: 42\nConfidence: 60%", PREFILLED)),
            _response(instance(), ("x</think>Answer: 41", PREFILLED)),
            # Cut off inside the trace: no judge call, wrong at an imputed 100%.
            _response(instance(), ("still working, confidence: 90", PREFILLED)),
        ]

        scored = await _score(task, responses)

        assert len(calls) == 2
        stated = [r.outputs[0].metadata["judge_result"]["confidence_stated"] for r in scored]
        assert stated == [True, False, False]
        metrics = _metrics(task, scored)
        assert metrics["accuracy"] == pytest.approx(1 / 3)
        # One stated confidence: 60% on a correct answer.
        assert metrics["calibration_error_stated"] == pytest.approx(0.4)
        assert metrics["confidence_stated_rate"] == pytest.approx(1 / 3)
        assert "calibration_error_all_bins" not in metrics

    @pytest.mark.anyio
    async def test_calibration_is_left_out_when_no_output_states_a_confidence(self, monkeypatch):
        task = get_task("hle:text:verified:v2")
        judge, _ = scripted_judge("extracted_final_answer: 41\ncorrect: no\nconfidence: 100")
        use_judge(monkeypatch, judge)
        instance = Instance(question="q", gold_answer="42", metadata={"category": "Math"})

        scored = await _score(task, [_response(instance, ("x</think>Answer: 41", PREFILLED))])

        metrics = _metrics(task, scored)
        # 0.0 would read as perfect calibration.
        assert "calibration_error_stated" not in metrics
        assert metrics["confidence_stated_rate"] == 0.0
        assert metrics["accuracy"] == 0.0

    def test_the_original_keeps_its_metrics_and_judge_result(self):
        names = [metric.name for metric in get_task("hle:text:verified").config.metrics]
        assert "calibration_error_all_bins" in names
        assert "calibration_error_stated" not in names

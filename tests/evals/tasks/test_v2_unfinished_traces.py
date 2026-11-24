"""Tests for the ``v2`` variants, which treat an unfinished reasoning trace as no answer.

A chat template that opens ``<think>`` in the prompt leaves no opening tag in the
generation, so a trace that never closed looks like an answer. Providers record such a
prompt on each output and, where they have evidence, whether the trace closed. ``OPEN``
and ``CLOSED`` are those records; ``NO_EVIDENCE`` is a prompt-opened trace the provider
could not vouch for either way.
"""

from __future__ import annotations

import json

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks import arena_hard
from olmo_eval.evals.tasks.common import get_task, register_sampling_variant, registry
from olmo_eval.runners.io.builders import build_predictions
from tests.evals.tasks.judge_stubs import scripted_judge, use_judge

pytestmark = pytest.mark.usefixtures("judge_test_env")

OPEN = {"prompt_opens_think": True, "think_closed": False}
CLOSED = {"prompt_opens_think": True, "think_closed": True}
NO_EVIDENCE = {"prompt_opens_think": True}
# Most cases below are a trace the provider knows stayed open, or text whose own
# ``</think>`` shows that it closed.
PREFILLED = OPEN
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
    "aime_2025:pass_at_32:v2",
    "aime_2026:pass_at_32:v2",
    "livecodebench_hidden:v2",
    "livecodebench:lite:v2",
    "bbq:mcq:v2",
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

    @pytest.mark.parametrize("spec", V2_SPECS)
    @pytest.mark.parametrize("record", [NO_EVIDENCE, {"prompt_opens_think": None}])
    @pytest.mark.parametrize("finish_reason", ["stop", "length"])
    def test_an_output_without_evidence_is_not_scored(self, spec, record, finish_reason):
        """Unknown is neither an answer nor an unfinished trace, for every v2 scorer.

        Stripping runs before every scorer and judge, so the output never reaches one:
        the runner records the instance as unscored and counts it against the task's
        failure budget.
        """
        from olmo_eval.common.scorers.execution import ScoringIncompleteError

        task = get_task(spec)
        response = _response(
            Instance(question="q", gold_answer="A"),
            ("The answer is (A).", {**record, "finish_reason": finish_reason}),
        )

        with pytest.raises(ScoringIncompleteError, match="no evidence"):
            task.strip_thinking_traces([response])

        assert response.outputs[0].text == "The answer is (A)."
        assert "unclosed_thinking" not in response.outputs[0].metadata

    @pytest.mark.parametrize("spec", V2_SPECS)
    @pytest.mark.parametrize("finish_reason", ["stop", "length"])
    def test_evidence_decides_whatever_stopped_the_generation(self, spec, finish_reason):
        task = get_task(spec)
        response = _response(
            Instance(question="q", gold_answer="A"),
            ("a trace that ended", {**OPEN, "finish_reason": finish_reason}),
            ("an answer, perhaps cut sho", {**CLOSED, "finish_reason": finish_reason}),
        )

        task.strip_thinking_traces([response])

        assert [o.text for o in response.outputs] == ["", "an answer, perhaps cut sho"]

    @pytest.mark.anyio
    async def test_the_runner_leaves_an_output_without_evidence_unscored(self):
        from olmo_eval.runners.asynq.results import _score_response
        from tests.evals.tasks.judge_stubs import score_like_the_runner

        task = get_task("wmdp:mcq:v2")
        instance = Instance(question="q", choices=("w", "x", "y", "z"), gold_answer="A")
        response = _response(instance, ("ANSWER: B", NO_EVIDENCE))

        class _Stripping:
            async def score_responses(self, responses):
                return [await _score_response(task, responses[0], ScoringContext())]

        scored = await score_like_the_runner(_Stripping(), response)

        assert scored.scores == {}
        error = scored.outputs[0].metadata["scoring_errors"]["__response__"]
        assert error["type"] == "ScoringIncompleteError"

    def test_v2_alone_is_the_same_measurement_as_the_stacked_spec(self):
        # A variant is also a spec of its base task; it must not be a meaningless one.
        for stacked, alone in (
            ("zebralogic:chat:v2", "zebralogic:v2"),
            ("omniscience:judge:v2", "omniscience:v2"),
            ("bbq:mcq:v2", "bbq:v2"),
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
        assert metrics["subcategory__wmdp-bio"] == 0.5
        # A subset with no answered question has no accuracy; 0.0 would read as unsafe.
        assert "subcategory__wmdp-cyber" not in metrics
        accuracy = task.config.primary_metric
        assert [accuracy.compute_instance(r) for r in scored] == [None, 0.0, 1.0, None]
        assert scored[0].outputs[0].metadata["unclosed_thinking"] is True
        saved = build_predictions(scored)[0]["model_output"][0]
        assert saved["scorer_results"]["wmdp_multiple_choice"] == {"no_answer": True}

    @pytest.mark.anyio
    @pytest.mark.parametrize("unfinished_first", [True, False])
    async def test_several_samples_are_counted_one_by_one_in_any_order(self, unfinished_first):
        """Each sample carries its own record, so the order of samples changes nothing."""
        task = get_task("wmdp:mcq:v2")
        samples = [
            ("maybe (A), or (B)", OPEN),  # no answer
            ("weighing</think>ANSWER: B", OPEN),  # answered, not the gold letter
            ("weighing</think>ANSWER: A", OPEN),  # answered with the gold letter
        ]
        if not unfinished_first:
            samples.reverse()
        instance = Instance(
            question="q",
            choices=("w", "x", "y", "z"),
            gold_answer="A",
            metadata={"subcategory": "wmdp-bio"},
        )

        scored = await _score(task, [_response(instance, *samples)])

        metrics = _metrics(task, scored)
        assert metrics["accuracy"] == 0.5
        assert metrics["no_answer_rate"] == pytest.approx(1 / 3)
        assert task.config.primary_metric.compute_instance(scored[0]) == 0.5

    @pytest.mark.anyio
    @pytest.mark.parametrize("unfinished_first", [True, False])
    async def test_one_unfinished_and_one_safe_sample_score_the_same_in_both_orders(
        self, unfinished_first
    ):
        task = get_task("wmdp:mcq:v2")
        samples = [("maybe (A), or (B)", OPEN), ("weighing</think>ANSWER: B", OPEN)]
        if not unfinished_first:
            samples.reverse()
        instance = Instance(question="q", choices=("w", "x", "y", "z"), gold_answer="A")

        scored = await _score(task, [_response(instance, *samples)])

        metrics = _metrics(task, scored)
        assert (metrics["accuracy"], metrics["no_answer_rate"]) == (1.0, 0.5)

    @pytest.mark.anyio
    async def test_no_answered_question_is_an_error_not_a_promoted_metric(self):
        """With accuracy undefined, no other metric may stand in as the task's score."""
        from olmo_eval.runners.asynq.preparation import compute_task_metrics
        from olmo_eval.runners.processing.utils import get_primary_metric

        task = get_task("wmdp:mcq:v2")
        scored = await _score(task, self._responses()[:1])

        metrics = _metrics(task, scored)
        assert "accuracy" not in metrics
        assert metrics["no_answer_rate"] == 1.0

        result = compute_task_metrics(
            spec="wmdp:mcq:v2",
            task=task,
            scored_responses=scored,
            failed_instances={},
            total_instances=1,
            duration_seconds=1.0,
        )

        assert result.metrics == {}
        assert "Primary metric 'accuracy' is undefined" in result.error
        assert "no_answer_rate=1" in result.error
        assert result.failure_budget_exceeded is False
        assert get_primary_metric(result.metrics, result.primary_metric) is None
        # The outputs are still saved for inspection.
        assert result.predictions

    @pytest.mark.anyio
    async def test_the_original_counts_the_same_questions_as_safe(self):
        task = get_task("wmdp:mcq")
        scored = await _score(task, self._responses())

        # The unfinished trace is read for a letter and the refusal counts as safe.
        assert _metrics(task, scored)["accuracy"] == 0.75
        assert [r.scores["wmdp_multiple_choice"] for r in scored] == [1.0, 0.0, 1.0, 1.0]


class TestMMLUPro:
    @pytest.mark.anyio
    async def test_an_unfinished_trace_is_not_an_answer(self):
        def instance() -> Instance:
            return Instance(question="q", gold_answer="C")

        texts = [
            ("is it B? no, (C) fits the facts", OPEN),
            ("hm</think>Therefore, the answer is (C).", OPEN),
            # Only the unfinished trace differs from ``cot``: a list of letters is
            # still read as its first letter here.
            ("hm</think>Therefore, the answer is (C, E).", OPEN),
        ]
        v2, original = get_task("mmlu_pro_law:cot:v2"), get_task("mmlu_pro_law:cot")

        new = await _score(v2, [_response(instance(), text) for text in texts])
        old = await _score(original, [_response(instance(), text) for text in texts])

        assert [r.scores["exact_match"] for r in new] == [0.0, 1.0, 1.0]
        assert [o.extracted_answer for r in new for o in r.outputs] == [None, "C", "C"]
        # The original reads "C" out of the unfinished trace.
        assert [r.scores["exact_match"] for r in old] == [1.0, 1.0, 1.0]
        assert v2.config.metrics == original.config.metrics


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
            {**OPEN, "finish_reason": "length"},
            # The trace ended at end of turn without closing.
            {**OPEN, "finish_reason": "stop"},
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
    async def test_an_answer_cut_off_after_a_closed_trace_is_judged(self, monkeypatch):
        task = get_task("arena_hard_v2:hard_prompt:v2")
        judge, calls = scripted_judge("[[A=B]]", "[[A=B]]")
        use_judge(monkeypatch, judge)
        # A server's reasoning parser removed the closed trace; the answer hit the limit.
        metadata = {**CLOSED, "finish_reason": "length"}

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
    @pytest.mark.anyio
    async def test_a_trace_that_ended_at_end_of_turn_is_not_judged(self, monkeypatch):
        """The original finds an unfinished response by its finish reason alone."""

        def responses() -> list[Response]:
            instance = Instance(question="q", gold_answer="42", metadata={"category": "Math"})
            return [
                _response(instance, ("so it is 42? Answer: 42", {**OPEN, "finish_reason": "stop"}))
            ]

        reply = "extracted_final_answer: 42\ncorrect: yes\nconfidence: 90"
        judge, calls = scripted_judge(reply)
        use_judge(monkeypatch, judge)

        v2 = get_task("hle:text:verified:v2")
        new = await _score(v2, responses())
        assert calls == []
        assert new[0].scores["accuracy"] == 0.0
        assert new[0].outputs[0].metadata["judge_result"]["no_final_response"] is True

        original = get_task("hle:text:verified")
        old = await _score(original, responses())
        assert len(calls) == 1
        assert old[0].scores["accuracy"] == 1.0
        assert v2.config.metrics == original.config.metrics


class TestAIME:
    @pytest.mark.parametrize("year", [2025, 2026])
    def test_a_boxed_candidate_in_an_unfinished_trace_is_not_extracted(self, year):
        trace = "Try 7. So maybe \\boxed{42}? Let me re-check the case n=3"

        def extracted(spec: str, text: str, record: dict) -> str | None:
            task = get_task(spec)
            response = _response(Instance(question="q", gold_answer="42"), (text, record))
            task.strip_thinking_traces([response])
            task._extract_answers([response])
            return response.outputs[0].extracted_answer

        assert extracted(f"aime_{year}:pass_at_32", trace, OPEN) == "42"
        assert not extracted(f"aime_{year}:pass_at_32:v2", trace, OPEN)
        # A finished response is extracted from exactly as before.
        finished = "Try 7.</think>The answer is \\boxed{42}."
        assert extracted(f"aime_{year}:pass_at_32:v2", finished, CLOSED) == "42"
        assert extracted(f"aime_{year}:pass_at_32:v2", finished, CLOSED) == extracted(
            f"aime_{year}:pass_at_32", finished, CLOSED
        )

    def test_v2_keeps_the_sampling_and_metrics_it_is_stacked_on(self):
        stacked, base = get_task("aime_2025:pass_at_32:v2"), get_task("aime_2025:pass_at_32")
        assert stacked.config.sampling_params == base.config.sampling_params
        assert stacked.config.metrics == base.config.metrics


class TestLiveCodeBench:
    @pytest.mark.parametrize("task_name", ["livecodebench_hidden", "livecodebench"])
    def test_a_draft_in_an_unfinished_trace_is_not_the_solution(self, task_name):
        draft = "First attempt:\n```python\nprint(1)\n```\nThat fails when n is"
        finished = "plan\n```python\nprint(1)\n```\n</think>\n```python\nprint(2)\n```"

        def extracted(spec: str, text: str, record: dict) -> str | None:
            task = get_task(spec)
            response = _response(Instance(question="q"), (text, record))
            task.strip_thinking_traces([response])
            return task.extract_answer(response.outputs[0])

        assert extracted(task_name, draft, OPEN) == "print(1)"
        assert extracted(f"{task_name}:v2", draft, OPEN) is None
        assert extracted(f"{task_name}:v2", finished, CLOSED) == "print(2)"
        assert extracted(task_name, finished, CLOSED) == "print(2)"


class TestBBQ:
    @pytest.mark.anyio
    async def test_a_letter_in_an_unfinished_trace_is_a_parsing_error(self):
        def response(text: str) -> Response:
            instance = Instance(
                question="q",
                choices=("x", "y", "z"),
                gold_answer="A",
                metadata={
                    "bias_label": "B",
                    "unknown_label": "A",
                    "question_polarity": "neg",
                    "context_condition": "ambig",
                    "category": "Age",
                },
            )
            return _response(instance, (text, OPEN))

        trace, answer = "it could be (A), though", "hm</think>ANSWER: A"
        v2, original = get_task("bbq:mcq:v2"), get_task("bbq:mcq")

        new = await _score(v2, [response(trace), response(answer)])
        old = await _score(original, [response(trace), response(answer)])

        assert [r.scores["bbq_bias"] for r in new] == [0.0, 1.0]
        assert [r.instance.metadata["is_parsing_error"] for r in new] == [True, False]
        assert [r.scores["bbq_bias"] for r in old] == [1.0, 1.0]

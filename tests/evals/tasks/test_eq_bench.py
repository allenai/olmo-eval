"""Tests for the EQ-Bench tasks."""

import math

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import LMOutput, RequestType, Response, Split
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.eq_bench import (
    EQ_BENCH_REPO,
    EQ_BENCH_REVISION,
    item_score,
    parse_ratings,
)
from olmo_eval.runners.asynq.preparation import compute_task_metrics

PROMPT = (
    "\nYour task is to predict the likely emotional responses of a character in this "
    "dialogue:\n\nRobert: Claudia, you've always been the idealist.\n[End dialogue]\n\n"
    "At the end of this dialogue, Robert would feel...\nRemorseful\nIndifferent\n"
    "Affectionate\nAnnoyed\n\nRemorseful: <score>\nIndifferent: <score>\nAffectionate: "
    "<score>\nAnnoyed: <score>\n\n\n[End of answer]\n\nYour answer:\n"
)
REFERENCE = {
    "emotion1": "Remorseful",
    "emotion2": "Indifferent",
    "emotion3": "Affectionate",
    "emotion4": "Annoyed",
    "emotion1_score": 0,
    "emotion2_score": "6",
    "emotion3_score": 0,
    "emotion4_score": "7",
}
ROW = {
    "prompt": PROMPT,
    "reference_answer": "{'emotion1': 'Remorseful', 'emotion1_score': 2}",
    "reference_answer_fullscale": repr(REFERENCE),
}
EXACT = "Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7"


def _s(d: float) -> float:
    """The reference's scaled difference for one emotion."""
    if d == 0:
        return 0.0
    return 6.5 / (1 + math.e ** (-1.2 * (d - 4))) if d <= 5 else d


@pytest.fixture
def task():
    return get_task("eq_bench")


@pytest.fixture
def loaded_sources(monkeypatch):
    sources = []

    def load(self, source, streaming=False):
        sources.append(source)
        return iter([ROW, ROW])

    monkeypatch.setattr(DataLoader, "load", load)
    return sources


def _response(task, text: str) -> Response:
    instance = task.process_doc(ROW)
    return Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(text=text)]
    )


class TestRegistration:
    def test_specs_are_registered(self):
        assert task_exists("eq_bench")
        assert task_exists("eq_bench:thinking")

    def test_data_source_is_pinned(self, task):
        source = task.config.get_data_source()

        assert task.config.split == Split.VALIDATION
        assert (source.path, source.revision, source.split) == (
            EQ_BENCH_REPO,
            EQ_BENCH_REVISION,
            "validation",
        )

    def test_sampling_follows_the_reference(self, task):
        params = task.config.sampling_params

        assert (params.temperature, params.max_tokens) == (0.01, 60)

    def test_thinking_variant_lifts_the_token_limit(self):
        params = get_task("eq_bench:thinking").config.sampling_params

        assert (params.temperature, params.top_p, params.max_tokens) == (0.6, 0.95, None)

    def test_metrics(self, task):
        metrics = {m.name: m for m in task.config.metrics}

        assert set(metrics) == {"eq_bench_score", "parse_failure_rate"}
        assert task.config.get_primary_metric().name == "eq_bench_score"
        assert metrics["eq_bench_score"].display_format() == "raw"
        assert metrics["parse_failure_rate"].higher_is_better() is False


class TestInstances:
    def test_rows_become_instances(self, task, loaded_sources):
        instances = list(task.instances)

        assert [i.metadata["id"] for i in instances] == [1, 2]
        assert instances[0].question == PROMPT
        assert instances[0].metadata["reference"] == REFERENCE
        assert loaded_sources[0].revision == EQ_BENCH_REVISION

    def test_prompt_is_the_only_message(self, task):
        request = task.format_request(task.process_doc(ROW))

        assert request.request_type == RequestType.CHAT
        assert request.messages == ({"role": "user", "content": PROMPT},)


class TestParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (EXACT, {"Remorseful": "0", "Indifferent": "6", "Affectionate": "0", "Annoyed": "7"}),
            ("**Remorseful:** 2\n# Annoyed: 7", {"Remorseful": "2", "Annoyed": "7"}),
            ("Annoyed: 7.5", {"Annoyed": "7"}),
            ("Annoyed: 2\nAnnoyed: 9", {"Annoyed": "9"}),
            ("Annoyed:\n7", {"Annoyed": "7"}),
            ("Annoyed:7", {}),
            ("Sure! Here you go.", {}),
        ],
    )
    def test_ratings_are_read_as_the_reference_reads_them(self, text, expected):
        assert parse_ratings(text) == expected


class TestItemScore:
    def test_an_exact_match_scores_ten(self):
        assert item_score(REFERENCE, parse_ratings(EXACT)) == 10.0

    def test_distances_use_the_s_curve_up_to_five_then_grow_linearly(self):
        ratings = {"Remorseful": "1", "Indifferent": "6", "Affectionate": "6", "Annoyed": "0"}

        expected = 10 - (_s(1) + _s(6) + _s(7)) * 0.7477
        assert item_score(REFERENCE, ratings) == pytest.approx(expected)
        assert _s(6) == 6 and _s(7) == 7

    # The reference's curve evaluated by hand: a distance of 5 is still on the curve
    @pytest.mark.parametrize(
        "d,scaled",
        [(1, 0.17288046), (2, 0.54062253), (3, 1.50458891), (4, 3.25), (5, 4.99541109), (6, 6.0)],
    )
    def test_one_distance_costs_what_the_reference_charges(self, d, scaled):
        ratings = {"Remorseful": str(d), "Indifferent": "6", "Affectionate": "0", "Annoyed": "7"}

        assert item_score(REFERENCE, ratings) == pytest.approx(10 - scaled * 0.7477)

    def test_scores_can_be_negative(self):
        ratings = {"Remorseful": "10", "Indifferent": "0", "Affectionate": "10", "Annoyed": "0"}

        assert item_score(REFERENCE, ratings) < 0

    def test_emotion_names_ignore_case(self):
        ratings = {"remorseful": "0", "INDIFFERENT": "6", "Affectionate": "0", "annoyed": "7"}

        assert item_score(REFERENCE, ratings) == 10.0

    @pytest.mark.parametrize(
        "ratings",
        [
            {"Remorseful": "0", "Indifferent": "6", "Affectionate": "0"},
            {
                "Remorseful": "0",
                "Indifferent": "6",
                "Affectionate": "0",
                "Annoyed": "7",
                "Sad": "1",
            },
            {"Remorseful": "0", "Indifferent": "6", "Affectionate": "0", "Sad": "7"},
            {"Remorseful": "0", "Indifferent": "0", "Affectionate": "0", "Annoyed": "0"},
            {"Remorseful": "0", "remorseful": "6", "Affectionate": "0", "Annoyed": "7"},
        ],
        ids=["three-emotions", "five-emotions", "wrong-emotion", "all-zero", "duplicate-case"],
    )
    def test_answers_the_reference_rejects_do_not_count(self, ratings):
        assert item_score(REFERENCE, ratings) is None


class TestScoring:
    @pytest.mark.anyio
    async def test_scores_and_aggregates_a_run(self, task):
        texts = [
            EXACT,
            "<think>Robert is angry.</think>\nRemorseful: 1\nIndifferent: 6\nAffectionate: 0\n"
            "Annoyed: 7",
            *([EXACT] * 4),
        ]
        responses = [_response(task, text) for text in texts]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        second = 10 - _s(1) * 0.7477
        assert responses[1].scores == {"eq_bench": pytest.approx(second)}
        assert responses[1].outputs[0].metadata["eq_bench_result"]["ratings"]["Remorseful"] == "1"
        assert metrics["eq_bench_score"]["eq_bench"] == pytest.approx(10 * (50 + second) / 6)
        assert metrics["parse_failure_rate"]["eq_bench"] == 0.0

    @pytest.mark.anyio
    async def test_unparseable_answers_are_left_out_without_a_scoring_error(self, task):
        responses = [_response(task, EXACT) for _ in range(5)] + [_response(task, "I can't.")]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        failed = responses[-1]
        assert failed.scores == {}
        assert "scoring_errors" not in failed.outputs[0].metadata
        assert failed.outputs[0].metadata["eq_bench_result"] == {"ratings": {}, "score": None}
        assert metrics["eq_bench_score"]["eq_bench"] == 100.0
        assert metrics["parse_failure_rate"]["eq_bench"] == pytest.approx(1 / 6)

    @pytest.mark.anyio
    async def test_a_run_below_the_parse_threshold_has_no_score(self, task):
        responses = [_response(task, EXACT) for _ in range(4)] + [_response(task, "?")]

        # The runner scores one response at a time
        for response in responses:
            await task.score_responses([response], ScoringContext())
        result = compute_task_metrics(
            spec="eq_bench",
            task=task,
            scored_responses=responses,
            failed_instances={},
            total_instances=len(responses),
            duration_seconds=0.0,
        )

        assert result.metrics == {}
        assert result.failure_budget_exceeded
        assert "only 4 of 5 EQ-Bench answers parse" in (result.error or "")
        assert (
            "infrastructure" not in responses[0].outputs[0].metadata["scoring_errors"]["eq_bench"]
        )

    @pytest.mark.anyio
    async def test_a_run_at_the_parse_threshold_has_a_score(self, task):
        responses = [_response(task, EXACT) for _ in range(5)] + [_response(task, "?")]

        for response in responses:
            await task.score_responses([response], ScoringContext())
        result = compute_task_metrics(
            spec="eq_bench",
            task=task,
            scored_responses=responses,
            failed_instances={},
            total_instances=len(responses),
            duration_seconds=0.0,
        )

        assert result.error is None
        assert result.metrics["eq_bench_score"]["eq_bench"] == 100.0
        assert result.metrics["parse_failure_rate"]["eq_bench"] == pytest.approx(1 / 6)

    @pytest.mark.anyio
    async def test_samples_of_a_prompt_average_over_those_that_parse(self, task):
        response = _response(task, EXACT)
        response.outputs.extend(
            [
                LMOutput(text="Remorseful: 1\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7"),
                LMOutput(text="?"),
            ]
        )

        await task.score_responses([response], ScoringContext())
        metrics = {m.name: m for m in task.config.metrics}

        expected = 10 * (10 + 10 - _s(1) * 0.7477) / 2
        assert metrics["eq_bench_score"].compute_instance(response) == pytest.approx(expected)
        assert metrics["parse_failure_rate"].compute_instance(response) == pytest.approx(1 / 3)

    @pytest.mark.anyio
    async def test_an_unfinished_trace_gives_no_answer(self, task):
        responses = [_response(task, EXACT) for _ in range(5)] + [
            _response(task, "<think>Remorseful: 0\nIndifferent: 6\nAffectionate: 0\nAnnoyed: 7")
        ]

        await task.score_responses(responses, ScoringContext())

        assert responses[-1].scores == {}

"""Tests for HardReasoning task logic."""

import json
from typing import Any

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import hard_reasoning
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.hard_reasoning import (
    HARD_REASONING_TASKS,
    _extract_last_valid_json,
    _json_values,
)

GOLD = '{"solution": [2, 6, 7, 8]}'


class _ListScenario:
    """Stand-in for an np_hard_reasoning scenario whose answer is a list of ints."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.valid = data["valid"]

    @classmethod
    def load_from_json(cls, data: dict[str, Any]) -> "_ListScenario":
        return cls(data)

    @staticmethod
    def load_answer_from_json(payload: Any) -> list[Any]:
        if isinstance(payload, dict) and "solution" in payload:
            payload = payload["solution"]
        if not isinstance(payload, list):
            raise ValueError("Expected list answer")
        return payload

    def check(self, answer: list[Any]) -> bool:
        return [self.valid[i] for i in answer] == [True] * len(answer) and bool(answer)


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def scenario_class(subset: str) -> Any:
        if subset != "bringing_toys":
            raise ValueError(f"Unknown hard_reasoning subset: {subset!r}")
        return _ListScenario

    monkeypatch.setattr(hard_reasoning, "_scenario_class", scenario_class)


def _score(task_spec: str, text: str, scenario_data: dict | None = None) -> Response:
    task = get_task(task_spec)
    instance = Instance(
        question="q",
        metadata={
            "id": 0,
            "subset": "bringing_toys",
            "scenario_data": scenario_data or {"valid": [False, False, True] + [True] * 6},
        },
    )
    request = LMRequest(request_type=RequestType.CHAT, prompt="")
    response = Response(instance=instance, request=request, outputs=[LMOutput(text=text)])
    task.strip_thinking_traces([response])
    task._extract_answers([response])
    task._apply_scorers([response])
    return response


class TestJsonValues:
    def test_top_level_values_in_order(self):
        assert _json_values('a {"x": 1} b [2, 3] c') == [{"x": 1}, [2, 3]]

    def test_invalid_outer_braces_still_find_inner_values(self):
        assert _json_values('{weight <= {"x": 1}}') == [{"x": 1}]

    def test_braces_inside_strings(self):
        assert _json_values('{"a": "}{"}') == [{"a": "}{"}]

    def test_multiline_json(self):
        assert _json_values('{"a":\n1}') == [{"a": 1}]

    @pytest.mark.parametrize("text", ["", "no json here", '{"incomplete": '])
    def test_no_values(self, text):
        assert _json_values(text) == []


class TestExtractLastValidJson:
    def test_prefers_last_valid(self):
        text = '{"solution": [1]} then {"solution": [2]} then {"other": 3}'
        assert _extract_last_valid_json(text, lambda v: "solution" in v) == {"solution": [2]}

    def test_prefers_objects_over_later_arrays(self):
        assert _extract_last_valid_json('{"solution": [1]} toy [3]', lambda v: True) == {
            "solution": [1]
        }

    def test_falls_back_to_array(self):
        assert _extract_last_valid_json("The answer is [1, 2]", lambda v: True) == [1, 2]

    def test_none_when_nothing_valid(self):
        assert _extract_last_valid_json('{"a": 1}', lambda v: False) is None


@pytest.mark.usefixtures("fake_registry")
class TestScoring:
    @pytest.mark.parametrize(
        "suffix",
        [
            "\n\nNote: I used {weight <= capacity}.",
            " (the empty set {} would fail)",
        ],
    )
    def test_trailing_braces_do_not_hide_the_answer(self, suffix):
        response = _score("hard_reasoning_bringing_toys", GOLD + suffix)
        assert response.scores == {"hard_reasoning_check": 1.0, "parsed": 1.0}

    def test_last_well_formed_answer_wins(self):
        response = _score("hard_reasoning_bringing_toys", GOLD + '\nActually: {"solution": []}')
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 1.0}

    def test_bare_list_in_prose(self):
        response = _score("hard_reasoning_bringing_toys", "The answer is [2, 6, 7, 8]")
        assert response.scores["hard_reasoning_check"] == 1.0

    def test_wrong_shape_is_a_parse_failure(self):
        response = _score("hard_reasoning_bringing_toys", '{"answer": [2, 6]}')
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 0.0}
        assert response.outputs[0].metadata["answer_format_correct"] is False

    @pytest.mark.parametrize(
        "spec", ["hard_reasoning_bringing_toys", "hard_reasoning_bringing_toys:chat"]
    )
    def test_unterminated_think_has_no_answer(self, spec):
        response = _score(spec, "<think>Let me try " + GOLD + " ... wait, and")
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 0.0}

    @pytest.mark.parametrize(
        "spec", ["hard_reasoning_bringing_toys", "hard_reasoning_bringing_toys:chat"]
    )
    def test_draft_inside_think_is_not_scored(self, spec):
        response = _score(spec, "<think>Try " + GOLD + "</think>\nI could not find one.")
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 0.0}

    def test_answer_after_think_is_scored(self):
        response = _score(
            "hard_reasoning_bringing_toys:chat", '<think>{"solution": [0]}</think>' + GOLD
        )
        assert response.scores["hard_reasoning_check"] == 1.0

    def test_out_of_range_index_is_wrong(self):
        response = _score("hard_reasoning_bringing_toys", '{"solution": [99]}')
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 1.0}

    def test_malformed_scenario_data_raises(self):
        with pytest.raises(KeyError):
            _score("hard_reasoning_bringing_toys", GOLD, scenario_data={"not_valid": []})

    def test_unknown_subset_raises(self):
        task = get_task("hard_reasoning_bringing_toys")
        output = LMOutput(text=GOLD, extracted_answer=GOLD)
        instance = Instance(question="q", metadata={"subset": "nope", "scenario_data": {}})
        scorer = task._get_scorers()["hard_reasoning_check"]
        with pytest.raises(ValueError, match="Unknown hard_reasoning subset"):
            scorer.score(instance, output)


class TestTaskConfig:
    def test_primary_metric_is_check_accuracy(self):
        primary = get_task("hard_reasoning_bringing_toys").config.get_primary_metric()
        assert primary is not None
        assert primary.scorer().name == "hard_reasoning_check"

    def test_chat_uses_reference_system_prompt_and_strips_thinking(self):
        config = get_task("hard_reasoning_bringing_toys:chat").config
        assert config.formatter is not None
        assert config.formatter.to_dict()["system_prompt"] == "You are a problem solver."
        assert config.strip_thinking

    def test_every_task_and_variant_serializes_distinctly(self):
        specs = [
            f"hard_reasoning_{subset}{variant}"
            for subset in HARD_REASONING_TASKS
            for variant in ("", ":chat", ":dev")
        ]
        serialized = {json.dumps(get_task(s).config.to_dict(), sort_keys=True) for s in specs}
        assert len(serialized) == len(specs)

    def test_dependencies_are_pinned(self):
        for dep in get_task("hard_reasoning_bringing_toys").config.dependencies or []:
            assert "==" in dep or "@" in dep, dep


class TestWithUpstreamPackage:
    """Run the real verifier when np_hard_reasoning is installed."""

    def test_gold_answer_scores_one(self):
        pytest.importorskip("np_hard_reasoning")
        pytest.importorskip("z3")
        data = {
            "toys": [
                {"name": "ball", "value": 5, "weight": 4},
                {"name": "kite", "value": 4, "weight": 3},
                {"name": "yo-yo", "value": 3, "weight": 2},
            ],
            "capacity": 5,
            "target": 7,
        }
        response = _score("hard_reasoning_bringing_toys", '{"solution": [2, 3]}', data)
        assert response.scores == {"hard_reasoning_check": 1.0, "parsed": 1.0}
        response = _score("hard_reasoning_bringing_toys", '{"solution": [1, 2]}', data)
        assert response.scores == {"hard_reasoning_check": 0.0, "parsed": 1.0}

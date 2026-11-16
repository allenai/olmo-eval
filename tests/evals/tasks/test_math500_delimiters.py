"""Opt-in delimiter normalization preserves the reference MATH-500 measurement."""

import copy

import pytest

from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import LMOutput
from olmo_eval.evals.tasks.common.registry import get_task


def score(spec: str, gold: str, answer: str) -> tuple[float, str | None]:
    task = get_task(spec, {"num_fewshot": 0})
    instance = task.process_doc({"problem": "Give the coordinates.", "solution": gold})
    assert instance is not None
    output = LMOutput(text=answer)
    output.extracted_answer = task.extract_answer(output)
    return MinervaMathScorer().process_score(instance, output), instance.gold_answer


@pytest.mark.parametrize("sized_gold", [False, True])
@pytest.mark.parametrize("sized_answer", [False, True])
def test_coordinate_equivalence_in_both_directions(sized_gold, sized_answer):
    plain = r"\boxed{(3,\frac{\pi}{2})}"
    sized = r"\boxed{\left(3,\frac{\pi}{2}\right)}"
    value, gold = score(
        "math500:latex_delimiters",
        sized if sized_gold else plain,
        sized if sized_answer else plain,
    )
    assert value == 1.0
    assert gold == r"(3,\frac{\pi}{2})"


def test_reference_spec_keeps_historical_false_negative():
    value, gold = score(
        "math500", r"\boxed{\left(3,\frac{\pi}{2}\right)}", r"\boxed{(3,\frac{\pi}{2})}"
    )
    assert gold == r"\le(3,\frac{\pi}{2}\right)"
    assert value == 0.0


def test_corrected_spec_still_rejects_wrong_coordinates():
    value, _ = score(
        "math500:latex_delimiters",
        r"\boxed{\left(3,\frac{\pi}{2}\right)}",
        r"\boxed{(4,\frac{\pi}{2})}",
    )
    assert value == 0.0


@pytest.mark.parametrize(
    "answer",
    [r"\boxed{42}", r"\boxed{3\text{ ft}}", r"\leftarrow", r"\leftrightarrow", r"\rightarrow"],
)
def test_other_commands_and_legacy_unit_rules_are_unchanged(answer):
    reference = get_task("math500")
    corrected = get_task("math500:latex_delimiters")
    old_output, new_output = LMOutput(text=answer), LMOutput(text=answer)
    assert corrected.extract_answer(new_output) == reference.extract_answer(old_output)
    assert new_output.metadata == old_output.metadata
    assert new_output.text == answer


def test_requests_and_source_text_are_preserved_with_distinct_spec_identity():
    reference = get_task("math500", {"num_fewshot": 0})
    corrected = get_task("math500:latex_delimiters", {"num_fewshot": 0})
    original = {
        "problem": "Convert to polar coordinates.",
        "solution": r"The coordinates are \boxed{\left(3,\frac{\pi}{2}\right)}.",
        "index": 0,
        "level": 2,
        "subject": "Precalculus",
    }
    doc = copy.deepcopy(original)
    old_instance, new_instance = reference.process_doc(doc), corrected.process_doc(doc)
    assert old_instance is not None and new_instance is not None
    assert doc == original
    assert new_instance.metadata["solution_text"] == original["solution"]
    assert reference.format_request(old_instance) == corrected.format_request(new_instance)
    assert reference.config.sampling_params == corrected.config.sampling_params
    assert reference.config.metrics == corrected.config.metrics
    assert reference.config.to_dict()["name"] == "math500"
    assert corrected.config.to_dict()["name"] == "math500:latex_delimiters"
    assert corrected.config.to_dict()["data_source"]["revision"] == (
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )

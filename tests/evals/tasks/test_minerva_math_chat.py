"""Zero-shot chat variants of Minerva MATH and MATH-500 for reasoning models (``:chat``)."""

import pytest

from olmo_eval.common.scorers.base import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.minerva_math import MATH_SUBSETS


def _score(task, instance, text):
    response = Response(
        instance=instance, request=task.format_request(instance), outputs=[LMOutput(text=text)]
    )
    task.strip_thinking_traces([response])
    task._extract_answers([response])
    output = response.outputs[0]
    return MinervaMathScorer().process_score(instance, output), output


@pytest.mark.parametrize(
    "spec", [f"minerva_math_{s}:chat" for s in MATH_SUBSETS] + ["math500:chat"]
)
def test_chat_variant_config(spec):
    config = get_task(spec).config
    assert config.num_fewshot == 0
    assert not config.sampling_params.stop_sequences
    assert config.strip_thinking and config.strip_unclosed_thinking and config.thinking_prefilled


def test_prompt_is_one_chat_turn_with_the_r1_instruction():
    task = get_task("minerva_math_algebra:chat")
    instance = Instance(
        question="What is $1+1$?", gold_answer="2", metadata={"all_gold_answers": ["2"]}
    )
    request = task.format_request(instance)
    assert len(request.messages) == 1
    assert request.messages[0]["content"] == (
        "What is $1+1$?\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def test_answer_after_prefilled_thinking_is_scored_and_the_trace_is_not():
    task = get_task("minerva_math_algebra:chat")
    instance = Instance(question="q", gold_answer="2", metadata={"all_gold_answers": ["2"]})
    # The template pre-filled "<think>", so the output opens inside the trace.
    right, _ = _score(task, instance, "Maybe \\boxed{3}? No.</think>\nSo the answer is \\boxed{2}.")
    wrong, _ = _score(
        task, instance, "Maybe \\boxed{2}? No, 3.</think>\nSo the answer is \\boxed{3}."
    )
    assert right == 1.0 and wrong == 0.0


def test_unclosed_thinking_scores_as_no_answer():
    task = get_task("minerva_math_algebra:chat")
    instance = Instance(question="q", gold_answer="2", metadata={"all_gold_answers": ["2"]})
    score, output = _score(task, instance, "still thinking, maybe \\boxed{2}")
    assert score == 0.0 and not output.extracted_answer


def test_suite_is_the_seven_subjects_macro_averaged():
    suite = get_suite("minerva_math:chat")
    assert suite.expand() == tuple(f"minerva_math_{s}:chat" for s in MATH_SUBSETS)
    assert sum(sum(1 for _ in get_task(spec).instances) for spec in suite.expand()) == 5000


def test_reference_solutions_score_as_correct():
    """Each MATH reference solution, given as the reply after </think>, is graded correct."""
    misses, total = [], 0
    for spec in get_suite("minerva_math:chat").expand():
        task = get_task(spec)
        for instance in task.instances:
            total += 1
            score, _ = _score(
                task, instance, "reasoning</think>\n" + instance.metadata["solution_text"]
            )
            if score != 1.0:
                misses.append((spec, instance.metadata["id"]))
    assert total == 5000
    assert not misses, misses[:25]  # 5,000/5,000 at d507236d

"""The ``final_answer`` variant scores a thinking model's maths answer only after ``</think>``."""

import pytest

from olmo_eval.common.scorers import MinervaMathScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_SPECS = (
    "aime_2025",
    "aime_2026:pass_at_32:16k",
    "hmmt_feb_2026:pass_at_32",
    "minerva_math_algebra:olmo3",
    "math500",
)

# The prompt opens the trace (Olmo Think templates end in "<think>"), so outputs start inside it.
_CASES = (
    # Answer only inside a trace that ran to the token cap.
    ("Try 70. So \\boxed{70}. Wait, check again", 0.0),
    # Right answer inside the trace, wrong final answer.
    ("So \\boxed{70}.</think>\nThe answer is \\boxed{71}.", 0.0),
    # Wrong guess inside the trace, right final answer.
    ("Maybe \\boxed{71}? No.</think>\nThe answer is \\boxed{70}.", 1.0),
)


def _score(spec: str, text: str) -> float:
    """Strip as the runner does, then extract and score in-process (no scoring pool)."""
    task = get_task(spec)
    response = Response(
        instance=Instance(question="Q", gold_answer="70"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q<think>"),
        outputs=[LMOutput(text=text)],
    )
    task.strip_thinking_traces([response])
    output = response.outputs[0]
    output.extracted_answer = task.extract_answer(output)
    return MinervaMathScorer().process_score(response.instance, output)


@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize(("text", "expected"), _CASES)
def test_final_answer_scores_only_text_after_thinking(spec, text, expected):
    assert _score(f"{spec}:final_answer", text) == expected


@pytest.mark.parametrize("spec", _SPECS)
def test_existing_specs_still_credit_answers_inside_thinking(spec):
    assert _score(spec, _CASES[0][0]) == 1.0


@pytest.mark.parametrize("spec", _SPECS)
def test_final_answer_composes_without_changing_the_rest_of_the_spec(spec):
    base = get_task(spec).config
    variant = get_task(f"{spec}:final_answer").config

    assert (variant.strip_thinking, variant.strip_unclosed_thinking) == (True, True)
    assert variant.thinking_prefilled is True
    assert variant.sampling_params == base.sampling_params
    assert variant.formatter == base.formatter
    assert variant.metrics == base.metrics

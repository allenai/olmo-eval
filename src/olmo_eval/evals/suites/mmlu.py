"""MMLU evaluation suites."""

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register
from olmo_eval.evals.tasks.mmlu import _HUMANITIES, _OTHER, _SOCIAL_SCIENCES, _STEM


def _task_names(subjects: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}" for s in subjects)


MMLU_STEM = make_suite(
    "mmlu:stem",
    _task_names(_STEM),
)

MMLU_HUMANITIES = make_suite(
    "mmlu:humanities",
    _task_names(_HUMANITIES),
)

MMLU_SOCIAL_SCIENCES = make_suite(
    "mmlu:social_sciences",
    _task_names(_SOCIAL_SCIENCES),
)

MMLU_OTHER = make_suite(
    "mmlu:other",
    _task_names(_OTHER),
)

MMLU = register(
    Suite(
        name="mmlu",
        tasks=(MMLU_STEM, MMLU_HUMANITIES, MMLU_SOCIAL_SCIENCES, MMLU_OTHER),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)
from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, make_suite, register
from olmo_eval.evals.tasks.mmlu import _HUMANITIES, _OTHER, _SOCIAL_SCIENCES, _STEM, MMLU_SUBJECTS


def _task_names(subjects: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}" for s in subjects)


def _task_names_variant(subjects: tuple[str, ...], variant: str) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}:{variant}" for s in subjects)


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

MMLU_STEM_CHAT = make_suite("mmlu:stem:chat", _task_names_variant(_STEM, "chat"))
MMLU_HUMANITIES_CHAT = make_suite("mmlu:humanities:chat", _task_names_variant(_HUMANITIES, "chat"))
MMLU_SOCIAL_SCIENCES_CHAT = make_suite(
    "mmlu:social_sciences:chat", _task_names_variant(_SOCIAL_SCIENCES, "chat")
)
MMLU_OTHER_CHAT = make_suite("mmlu:other:chat", _task_names_variant(_OTHER, "chat"))
MMLU_CHAT = register(
    Suite(
        name="mmlu:chat",
        tasks=(
            MMLU_STEM_CHAT,
            MMLU_HUMANITIES_CHAT,
            MMLU_SOCIAL_SCIENCES_CHAT,
            MMLU_OTHER_CHAT,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

MMLU_BPB = make_suite(
    "mmlu:bpb",
    tuple(f"mmlu_{s}:rc:bpb" for s in MMLU_SUBJECTS),
)

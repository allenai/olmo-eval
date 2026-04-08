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

# mc::olmo3base suites (for parity with oe-eval-internal mmlu_*:mc::olmes)
make_suite("mmlu:stem:mc::olmo3base", _task_names_variant(_STEM, "mc:olmo3base"))
make_suite("mmlu:humanities:mc::olmo3base", _task_names_variant(_HUMANITIES, "mc:olmo3base"))
make_suite(
    "mmlu:social_sciences:mc::olmo3base",
    _task_names_variant(_SOCIAL_SCIENCES, "mc:olmo3base"),
)
make_suite("mmlu:other:mc::olmo3base", _task_names_variant(_OTHER, "mc:olmo3base"))


# rc (cloze) variants — for parity with oe-eval-internal mmlu_*:rc::olmes
def _rc_task_names_variant(subjects: tuple[str, ...], variant: str) -> tuple[str, ...]:
    return tuple(f"mmlu_{s}:rc:{variant}" for s in subjects)


_MMLU_RC_STEM = make_suite(
    "mmlu:stem:rc::olmo3base",
    _rc_task_names_variant(_STEM, "olmo3base"),
)
_MMLU_RC_HUMANITIES = make_suite(
    "mmlu:humanities:rc::olmo3base",
    _rc_task_names_variant(_HUMANITIES, "olmo3base"),
)
_MMLU_RC_SOCIAL_SCIENCES = make_suite(
    "mmlu:social_sciences:rc::olmo3base",
    _rc_task_names_variant(_SOCIAL_SCIENCES, "olmo3base"),
)
_MMLU_RC_OTHER = make_suite(
    "mmlu:other:rc::olmo3base",
    _rc_task_names_variant(_OTHER, "olmo3base"),
)

MMLU_RC = register(
    Suite(
        name="mmlu:rc::olmo3base",
        tasks=(
            _MMLU_RC_STEM,
            _MMLU_RC_HUMANITIES,
            _MMLU_RC_SOCIAL_SCIENCES,
            _MMLU_RC_OTHER,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
)

MMLU_BPB = make_suite(
    "mmlu:bpb",
    tuple(f"mmlu_{s}:rc:bpb" for s in MMLU_SUBJECTS),
)

"""Every spec's primary metric must be one of the metrics it computes.

The runner scores a task by looking its primary metric key up in the metrics the task
computed, so a spec whose primary metric is not computed never yields a primary score,
and a suite average silently leaves it out. Nothing in the registry prevents this:
variants merge field by field and a later variant overwrites an earlier one, so a prompt
preset that also sets ``metrics`` (``arc_easy:olmo3base``) can undo a measurement
variant (``arc_easy:bpb``) depending on the order in the spec (#593).

The violations on ``main`` are pinned so that fixing one must remove it from the list,
and introducing one fails here instead of in a suite's score.
"""

from __future__ import annotations

from olmo_eval.common.configs import expand_tasks
from olmo_eval.evals.suites import list_suites
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.runners.processing.utils import get_metric_metadata
from tests.evals.tasks.test_task_hashes import _specs

#: Specs whose primary metric is not among their computed metrics. Each is a known
#: defect kept for result compatibility; a fix lands as a new variant or suite.
KNOWN_UNSCORED_PRIMARY = frozenset(
    {
        # ``:bpb:olmo3base``: the ``bpb`` variant sets ``primary_metric`` to BPB and the
        # later ``olmo3base`` variant overwrites ``metrics`` with accuracy (#593).
        "csqa:bpb:olmo3base",
        # Class-level accuracy ``primary_metric`` survives the ``bpb`` variant.
        "gpqa_diamond:bpb",
        "gpqa_diamond_biology:bpb",
        "gpqa_diamond_chemistry:bpb",
        "gpqa_diamond_physics:bpb",
        "gpqa_extended:bpb",
        "gpqa_extended_biology:bpb",
        "gpqa_extended_chemistry:bpb",
        "gpqa_extended_physics:bpb",
        "gpqa_main:bpb",
        "gpqa_main_biology:bpb",
        "gpqa_main_chemistry:bpb",
        "gpqa_main_physics:bpb",
        "medqa:bpb",
        # Class-level primary is the multiple-choice scorer; the variant switches the
        # metric to a logprob scorer under the same metric name.
        "lab_bench_cloning_scenarios:olmo3base",
        "lab_bench_dbqa:olmo3base",
        "lab_bench_litqa2:olmo3base",
        "lab_bench_protocolqa:olmo3base",
        "lab_bench_seqqa:olmo3base",
        "lab_bench_suppqa:olmo3base",
        # Class-level pass@1 ``primary_metric`` survives the ``bpb`` variant.
        "mbpp:olmo3base:bpb",
        "mbpp:olmo3base:v2:bpb",
    }
)

#: Suites named for BPB whose average includes a task that does not score BPB.
KNOWN_MIXED_BPB_SUITES = frozenset(
    {
        "arc:bpb:olmo3base",  # both tasks score accuracy (#593)
        "gpqa:bpb",  # every task's primary is accuracy, which it does not compute
        "olmobase:easy:qa:bpb",  # arc and piqa score accuracy, csqa scores nothing (#593)
    }
)


def _computed_metric_keys(spec: str) -> set[str]:
    """The ``metric:scorer`` keys a task publishes, as the runner names them."""
    keys = set()
    for metric in get_task(spec).config.metrics:
        scorer = getattr(metric, "scorer", None)
        keys.add(f"{metric.name}:{scorer().name if scorer is not None else 'default'}")
    return keys


def _primary_metric_key(spec: str) -> str | None:
    return get_metric_metadata(get_task(spec))


def _assert_known(found: set[str], known: frozenset[str], what: str) -> None:
    new = sorted(found - known)
    fixed = sorted(known - found)
    assert not new, f"new {what}: {new}"
    assert not fixed, f"{what} no longer occur; remove them from the known list: {fixed}"


def test_primary_metric_is_computed() -> None:
    unscored = set()
    for spec in _specs():
        primary = _primary_metric_key(spec)
        if primary is not None and primary not in _computed_metric_keys(spec):
            unscored.add(spec)

    _assert_known(unscored, KNOWN_UNSCORED_PRIMARY, "specs whose primary metric is not computed")


def test_bpb_suites_average_only_bpb() -> None:
    bpb_suites = [name for name in list_suites() if "bpb" in name.split(":")]
    assert bpb_suites, "no BPB suites registered"

    mixed = set()
    for suite in bpb_suites:
        for spec in expand_tasks([suite]):
            primary = _primary_metric_key(spec)
            if (
                primary is None
                or not primary.startswith("bits_per_byte:")
                or primary not in _computed_metric_keys(spec)
            ):
                mixed.add(suite)

    _assert_known(mixed, KNOWN_MIXED_BPB_SUITES, "BPB suites averaging non-BPB tasks")

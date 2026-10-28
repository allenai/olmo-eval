"""OLMoBase math and code suites at 256 samples per problem, for pass@k curves.

Each leaf task of `olmobase:math` and `olmobase:code:v2` gets an `n256` variant that
draws 256 samples and reports pass@k for k = 1, 2, 4, ..., 256 with the scorer its
pass@k metrics already use. Prompts, sampling temperature, top-p, token budgets and the
primary metric are unchanged, so pass@1 estimates the same quantity as the base suites.
"""

from dataclasses import replace

import olmo_eval.evals.suites.code  # noqa: F401 – register olmobase:code:v2
import olmo_eval.evals.suites.olmobase  # noqa: F401 – register olmobase:math
from olmo_eval.common.metrics.base import PassAtKMetric
from olmo_eval.evals.suites.registry import Suite, get_suite, register
from olmo_eval.evals.tasks.common.registry import get_task, list_tasks, register_variant

NUM_SAMPLES = 256
KS = tuple(2**i for i in range(9))  # 1 .. 256
VARIANT = "n256"


def _base_task(spec: str) -> str:
    """The registered task a spec's variants apply to (the longest registered prefix)."""
    tasks = set(list_tasks())
    parts = spec.split(":")
    for i in range(len(parts), 0, -1):
        if ":".join(parts[:i]) in tasks:
            return ":".join(parts[:i])
    raise KeyError(spec)


def _overrides(spec: str) -> dict:
    config = get_task(spec).config
    passk = [m for m in config.metrics if isinstance(m, PassAtKMetric)]
    scorers = {m.scorer for m in passk}
    assert len(scorers) == 1, f"{spec}: expected one pass@k scorer, got {scorers}"
    (scorer,) = scorers
    others = tuple(m for m in config.metrics if not isinstance(m, PassAtKMetric))
    return {
        "sampling_params": replace(config.sampling_params, num_samples=NUM_SAMPLES),
        "metrics": others + tuple(PassAtKMetric(k=k, scorer=scorer) for k in KS),
    }


_registered: dict[str, dict] = {}


def _leaf(spec: str) -> str:
    base, overrides = _base_task(spec), _overrides(spec)
    if base in _registered:
        # Leaves sharing a base task (gsm_symbolic's p1/p2) must agree on the variant.
        assert _registered[base] == overrides, f"{spec}: conflicting {VARIANT} for {base}"
    else:
        register_variant(base, VARIANT, **overrides)
        _registered[base] = overrides
    return f"{spec}:{VARIANT}"


def _mirror(suite: Suite) -> Suite:
    return Suite(
        name=f"{suite.name}:{VARIANT}",
        tasks=tuple(_mirror(t) if isinstance(t, Suite) else _leaf(t) for t in suite.tasks),
        aggregation=suite.aggregation,
        description=f"{suite.description} ({NUM_SAMPLES} samples, pass@1-{KS[-1]})",
    )


for _name in ("olmobase:math", "olmobase:code:v2"):
    register(_mirror(get_suite(_name)))

"""The exclusion must remove exactly the earlier run's instances, not a positional slice."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

sys.path.insert(0, "src")

from olmo_eval.runners.asynq.preparation import _drop_already_sampled  # noqa: E402


@dataclass
class Config:
    limit: int
    seed: int = 42


def run(n, limit, earlier):
    population = [f"inst_{i}" for i in range(n)]
    sampled = random.Random(42).sample(population, limit)
    prior = random.Random(42).sample(population, earlier)
    os.environ["OLMO_EVAL_SKIP_SAMPLE_OF_SIZE"] = str(earlier)
    remaining = _drop_already_sampled(population, sampled, Config(limit=limit))
    return population, sampled, prior, remaining


failures = 0
for n in (597, 599, 600, 500, 100):
    limit = 100
    population, sampled, prior, remaining = run(n, limit, 40)
    expected = len(sampled) - len(set(prior) & set(sampled))
    checks = {
        "count is the sample minus the earlier run": len(remaining) == expected,
        "nothing from the earlier run survives": not (set(remaining) & set(prior)),
        "everything kept was in the sample": set(remaining) <= set(sampled),
        "union restores the whole sample": set(remaining) | set(prior) >= set(sampled),
        # Only meaningful when the sample is a real subset. When limit == population size
        # the sample is a permutation of everything and the earlier draw *is* its prefix,
        # so the positional slice coincides with the right answer -- DeepResearch, at 100
        # instances, is exactly that case.
        **({"not a positional slice": list(remaining) != list(sampled[40:])}
           if n > limit else {}),
    }
    bad = [name for name, ok in checks.items() if not ok]
    failures += len(bad)
    print("n=%-4d limit=100 earlier=40 -> %3d to run   %s"
          % (n, len(remaining), "OK" if not bad else "FAILED: " + "; ".join(bad)))

os.environ.pop("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", None)
population = [f"inst_{i}" for i in range(597)]
sampled = random.Random(42).sample(population, 100)
untouched = _drop_already_sampled(population, sampled, Config(limit=100))
print("unset -> unchanged:", untouched == sampled)
failures += untouched != sampled

os.environ["OLMO_EVAL_SKIP_SAMPLE_OF_SIZE"] = "100"
same = _drop_already_sampled(population, sampled, Config(limit=100))
print("earlier >= limit -> unchanged:", same == sampled)
failures += same != sampled

print("\n%s" % ("all checks passed" if not failures else f"{failures} FAILURES"))
sys.exit(1 if failures else 0)

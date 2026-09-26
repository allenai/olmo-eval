"""Post-training hill-climb dev pass: the dev tiers and their guards in one run.

Launching ``hillclimb:dev`` on a checkpoint evaluates, in one job:

- the hills' dev tiers: LiveCodeBench release_v3 at one sample per problem, and
  the OMEGA pair, reported as its in and out scores and the gap between them;
- the guards, which are watched for regressions and never hill-climbed:
  IFEval, IFBench and GPQA main.

The suite has no aggregate of its own. A mean over code, math, instruction
following and knowledge would hide which one moved, so every task reports its
own score and the OMEGA pair reports its gap.
"""

from olmo_eval.evals.suites.math import OMEGA_DEV
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

HILLCLIMB_DEV_TIERS = ("livecodebench:lite", OMEGA_DEV)
HILLCLIMB_GUARDS = ("ifeval", "ifeval_ood", "gpqa_main:cot")

make_suite(
    "hillclimb:dev",
    (*HILLCLIMB_DEV_TIERS, *HILLCLIMB_GUARDS),
    aggregation=AggregationStrategy.NONE,
    description=(
        "Hill-climb dev pass: LiveCodeBench v3 (one sample), the OMEGA in/out pair "
        "with its gap, and the IFEval, IFBench and GPQA main guards."
    ),
)

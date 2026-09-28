"""SWE-Bench Pro external evaluation.

SWE-Bench Pro evaluates LLM agents on long-horizon software engineering tasks
drawn from real repositories. Each task is graded by applying the agent's diff
to a pristine container and running hidden tests.

Repository: https://github.com/scaleapi/SWE-bench_Pro-os
"""

from olmo_eval.evals.external.benchmarks.swe_bench_pro.eval import SWEBenchProExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(SWEBenchProExternalEval("v2"))
register_external_eval(SWEBenchProExternalEval("v1"))

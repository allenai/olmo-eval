"""Tau2-bench external evaluation.

tau2_bench is a benchmark for evaluating language model agents on realistic
customer service tasks. It measures both task completion and constraint
satisfaction. We use the "verified" version of the benchmark, which fixes
some issues in the original version. For more details, see https://arxiv.org/abs/2512.07850.

Repository: https://github.com/amazon-agi/tau2-bench-verified
"""

from olmo_eval.evals.external.benchmarks.tau2.eval import Tau2ExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(Tau2ExternalEval())

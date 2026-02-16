"""Tau2-bench external evaluation.

tau2_bench is a benchmark for evaluating language model agents on realistic
customer service tasks. It measures both task completion and constraint
satisfaction.

Repository: https://github.com/sierra-research/tau2-bench
"""

from olmo_eval.evals.external.benchmarks.tau2.eval import Tau2ExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(Tau2ExternalEval())

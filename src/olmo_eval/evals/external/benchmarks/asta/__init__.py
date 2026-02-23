"""ASTA-bench external evaluation.

ASTA-bench is a benchmark for evaluating AI scientist agents on tasks including
literature search, code execution, data analysis, and end-to-end discovery.
Uses Inspect AI as its evaluation harness.

Repository: https://github.com/allenai/asta-bench
"""

from olmo_eval.evals.external.benchmarks.asta.eval import AstaExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(AstaExternalEval())

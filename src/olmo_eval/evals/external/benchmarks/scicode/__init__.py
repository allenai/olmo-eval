"""SciCode external evaluation.

Reference: https://scicode-bench.github.io/
Dataset: https://huggingface.co/datasets/SciCode1/SciCode
"""

from olmo_eval.evals.external.benchmarks.scicode.eval import SciCodeExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(SciCodeExternalEval())

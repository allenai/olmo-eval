"""BixBench-Verified-50 external evaluation.

BixBench-Verified-50 is an expert-revised 50-question subset of FutureHouse's
BixBench. An agent answers each open question by analysing the data of a capsule
in a Jupyter notebook, using upstream's notebook tools, and its answer is graded by
string match or an LLM judge as upstream grades it.

Dataset: https://huggingface.co/datasets/phylobio/BixBench-Verified-50
Benchmark: https://github.com/Future-House/BixBench
Agent environment: https://github.com/Future-House/data-analysis-crow
"""

from olmo_eval.evals.external.benchmarks.bixbench.eval import BixBenchVerifiedExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(BixBenchVerifiedExternalEval())

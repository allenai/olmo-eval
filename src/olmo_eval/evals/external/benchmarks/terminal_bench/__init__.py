"""Terminal-Bench external evaluation.

Terminal-Bench evaluates LLM agents on diverse terminal tasks requiring
software engineering skills, system administration, and problem-solving.
The loader pins the 2.1 release of the benchmark.

Repository: https://github.com/harbor-framework/terminal-bench-2-1
"""

from olmo_eval.evals.external.benchmarks.terminal_bench.eval import TerminalBenchExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(TerminalBenchExternalEval())

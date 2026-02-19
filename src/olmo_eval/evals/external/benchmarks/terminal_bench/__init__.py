"""Terminal-Bench 2.0 external evaluation.

Terminal-Bench 2.0 evaluates LLM agents on diverse terminal tasks requiring
software engineering skills, system administration, and problem-solving.

Repository: https://github.com/laude-institute/terminal-bench-2
"""

from olmo_eval.evals.external.benchmarks.terminal_bench.eval import TerminalBenchExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(TerminalBenchExternalEval())

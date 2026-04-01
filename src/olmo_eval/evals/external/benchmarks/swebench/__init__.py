"""SWE-bench external evaluation.

SWE-bench evaluates LLM coding agents on real GitHub issues. Agents must produce
a git patch that resolves the described issue, verified by running the repository's
test suite. We use mini-swe-agent (https://github.com/allenai/mini-swe-agent) for
patch generation and the official SWE-bench harness for scoring.

For more details, see https://www.swebench.com.
"""

from olmo_eval.evals.external.benchmarks.swebench.eval import SWEBenchExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(SWEBenchExternalEval())

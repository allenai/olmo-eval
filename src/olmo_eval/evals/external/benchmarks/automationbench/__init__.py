"""AutomationBench-AA external evaluation.

AutomationBench (Zapier) gives an agent a business workflow to carry out in
simulated SaaS apps through a REST-style API toolset, and checks the apps' final
state with programmatic assertions. Artificial Analysis runs it as
AutomationBench-AA on a private split. This eval runs the public tasks with the
same environment, turn cap and toolset, and scores them the AA way.

Benchmark: https://github.com/zapier/AutomationBench
Methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking
"""

from olmo_eval.evals.external.benchmarks.automationbench.eval import AutomationBenchAAExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(AutomationBenchAAExternalEval())

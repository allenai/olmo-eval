"""OpenAgentSafety external evaluation.

OpenAgentSafety evaluates AI agent safety in workplace scenarios with NPC
interactions on TheAgentCompany services.

Repository: https://github.com/OpenHands/benchmarks/tree/main/benchmarks/openagentsafety
Dataset: https://huggingface.co/datasets/mgulavani/openagentsafety_full_updated_v3
"""

from olmo_eval.evals.external.benchmarks.openagentsafety.eval import (
    OpenAgentSafetyExternalEval,
)
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(OpenAgentSafetyExternalEval())

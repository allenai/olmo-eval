"""Codex HumanEval task (alias for HumanEval with codex_humaneval name)."""

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetric
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.humaneval import HumanEval


@register("codex_humaneval")
class CodexHumanEval(HumanEval):
    pass


register_variant(
    "codex_humaneval",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    metrics=(BPBMetric(),),
)

register_variant(
    "codex_humaneval",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(),
)

register_variant("codex_humaneval", "olmo3base", num_fewshot=3, fewshot_seed=1234)

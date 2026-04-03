"""Codex HumanEval task (alias for HumanEval with codex_humaneval name)."""

from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetric
from olmo_eval.common.types import Instance
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.humaneval import HumanEval


@register("codex_humaneval")
class CodexHumanEval(HumanEval):
    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Override to prepend space to gold_answer.

        This matches the old oe-eval doc_to_target which returns " " + canonical_solution.
        """
        instance = super().process_doc(doc, index)
        return Instance(
            question=instance.question,
            gold_answer=" " + (instance.gold_answer or ""),
            metadata=instance.metadata,
        )


register_variant(
    "codex_humaneval",
    "bpb",
    formatter=PPLFormatter(leading_space=False, answer_prefix=""),
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

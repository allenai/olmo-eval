"""Codex HumanEval task (alias for HumanEval with codex_humaneval name)."""

from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetric, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, SamplingParams
from olmo_eval.evals.constants.code import OLMO3_HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code_before_fence
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


# =============================================================================
# OLMo3 base variant (```python code block prompt wrapping)
# =============================================================================


@register("codex_humaneval:olmo3base")
class CodexHumanEvalOlmo3Base(HumanEval):
    """CodexHumanEval with OLMo3 prompt wrapping (```python code block)."""

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        stop_sequences=OLMO3_HUMANEVAL_STOP_SEQUENCES,
    )
    fewshot_split: str = "test"

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prompt = "```python\n" + doc["prompt"]
        unit_tests = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=prompt,
            gold_answer=doc["canonical_solution"] + "```",
            metadata={
                "id": doc["task_id"],
                "entry_point": doc["entry_point"],
                "answer_prefix": doc["prompt"],
                "test": unit_tests,
            },
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return extract_code_before_fence(output.text)


register_variant(
    "codex_humaneval:olmo3base",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
)

register_variant(
    "codex_humaneval:olmo3base",
    "n32",
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_HUMANEVAL_STOP_SEQUENCES,
    ),
    metrics=(
        PassAtKMetric(k=1, scorer=CodeExecutionScorer),
        PassAtKMetric(k=2, scorer=CodeExecutionScorer),
        PassAtKMetric(k=4, scorer=CodeExecutionScorer),
        PassAtKMetric(k=8, scorer=CodeExecutionScorer),
        PassAtKMetric(k=16, scorer=CodeExecutionScorer),
    ),
)

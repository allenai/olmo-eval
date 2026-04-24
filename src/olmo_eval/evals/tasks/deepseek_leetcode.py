"""DeepSeek LeetCode code generation task.

LeetCode-style problems from the DeepSeek Coder paper.

Paper: https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation/LeetCode
Dataset: davidheineman/deepseek-leetcode
"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import OLMO3_HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.tasks.common import Task, register, register_variant


@dataclass(frozen=True, slots=True)
class CodeExecutionScorer3s(CodeExecutionScorer):
    timeout: float = 3.0
    separator: str = "\n"


@register("deepseek_leetcode")
class DeepSeekLeetCode(Task):
    """DeepSeek LeetCode code generation task."""

    data_source = DataSource(path="davidheineman/deepseek-leetcode")
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=32,
        stop_sequences=OLMO3_HUMANEVAL_STOP_SEQUENCES,
    )
    metrics = (
        PassAtKMetric(k=1, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=2, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=4, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=8, scorer=CodeExecutionScorer3s),
        PassAtKMetric(k=16, scorer=CodeExecutionScorer3s),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prompt = doc["prompt"]
        test_code = doc.get("test", "")

        return Instance(
            question=prompt,
            gold_answer="",
            metadata={
                "id": doc.get("questionId", str(index)),
                "prompt": prompt,
                "test": test_code,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        text = output.text
        # Try markdown code block extraction first
        p_code = re.compile(r"```python\n?(.*?)\n?```", flags=re.DOTALL)
        code_blocks = p_code.findall(text)
        if code_blocks:
            return code_blocks[0]
        # Fallback: split on common stop patterns
        codelist = re.split(r"\ndef|\nclass|\nif|\n#|\nprint", text)
        if codelist:
            return codelist[0]
        return text

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        for response in responses:
            prompt = response.instance.metadata["prompt"]
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    output.extracted_answer = prompt + code
                else:
                    output.extracted_answer = None


register_variant(
    "deepseek_leetcode",
    "olmo3base",
    num_fewshot=3,
    fewshot_seed=1234,
    primary_metric=PassAtKMetric(k=1, scorer=CodeExecutionScorer),
)

"""DS-1000 data science code generation task.

DS-1000 contains 1000 data science questions spanning seven Python libraries
with reliable metrics and perturbation-based defenses against memorization.

Paper: https://arxiv.org/abs/2211.11501
Dataset: xlangai/DS-1000
"""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import DS1000_STOP_SEQUENCES
from olmo_eval.evals.tasks.common import Task, register, register_variant

# DS-1000 requires longer timeout for data science libraries
_DS1000_SCORER = type(
    "DS1000Scorer",
    (CodeExecutionScorer,),
    {"timeout": 120.0, "__module__": __name__, "__qualname__": "DS1000Scorer"},
)


@register("ds1000")
class DS1000(Task):
    """DS-1000 data science code generation task."""

    data_source = DataSource(path="xlangai/DS-1000")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.6,
        top_p=0.6,
        do_sample=True,
        num_samples=5,
        stop_sequences=DS1000_STOP_SEQUENCES,
    )
    fewshot_split: str = "test"

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
        prompt = self._format_prompt(doc["prompt"])
        reference_code = doc.get("reference_code", "")
        code_context = doc.get("code_context", "")

        return Instance(
            question=prompt,
            gold_answer=reference_code.rstrip("\n") + "\n```",
            metadata={
                "id": doc.get("metadata", {}).get("problem_id", str(index)),
                "code_context": code_context,
                "test": "",
                "lib": doc.get("metadata", {}).get("library", ""),
            },
        )

    @staticmethod
    def _format_prompt(prompt_text: str) -> str:
        """Process DS-1000 prompt: replace code tags, mask solution lines."""
        text = prompt_text
        text = text.replace("\nBEGIN SOLUTION\n<code>\n", "\n")
        text = text.replace("    ### BEGIN SOLUTION", "")
        text = text.replace("<code>", "```python")
        text = text.replace("</code>", "```")
        text = text + "\n"

        # Mask "put solution in this variable" lines
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "put solution" in line:
                lines[i] = "\n# " + line
                # Remove trailing ``` if present before the masked line
                if i > 0 and lines[i - 1].strip() == "```":
                    lines.pop(i - 1)
        text = "\n".join(lines)
        text = text.rstrip("\n") + "\n"
        return text

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return output.text

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Assemble test program from code_context and model continuation."""
        for response in responses:
            code_context = response.instance.metadata.get("code_context", "")
            for output in response.outputs:
                continuation = self.extract_answer(output)
                if continuation and code_context:
                    # DS-1000 test harness format:
                    # code_context defines test_execution() and optionally test_string()
                    test_program = (
                        code_context
                        + "\n"
                        + f"code = {repr(continuation)}\n"
                        + "test_execution(code)\n"
                        + ("test_string(code)\n" if "test_string(" in code_context else "\n")
                    )
                    output.extracted_answer = test_program
                else:
                    output.extracted_answer = None


register_variant(
    "ds1000",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(answer_prefix=""),
)

register_variant(
    "ds1000",
    "bpb",
    formatter=PPLFormatter(leading_space=False),
    metrics=(BPBMetricByteAvg(),),
)

register_variant(
    "ds1000",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=_DS1000_SCORER),),
)

register_variant(
    "ds1000",
    "olmo3base",
    metrics=(PassAtKMetric(k=1, scorer=_DS1000_SCORER),),
)

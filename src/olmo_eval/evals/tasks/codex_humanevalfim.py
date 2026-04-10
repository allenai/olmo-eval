"""HumanEval Fill-in-the-Middle (FIM) infilling tasks.

Benchmarks for code infilling adapted from the 164-problem HumanEval dataset
by masking portions of the code. Three variants with different masking strategies:
single line, multi line, and random span.

Paper: https://arxiv.org/abs/2207.14255
Dataset: loubnabnl/humaneval_infilling
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import OLMO_FIM
from olmo_eval.evals.tasks.common import Task, register, register_variant

# The loubnabnl/humaneval_infilling dataset uses a loading script which is no
# longer supported by datasets>=4.  Load the raw JSONL files directly instead.
_HF_DATA_BASE = "hf://datasets/loubnabnl/humaneval_infilling/data"


@dataclass(frozen=True, slots=True)
class _FIMCodeScorer(CodeExecutionScorer):
    """Code execution scorer with 3s timeout matching oe-eval-internal."""

    timeout: float = 3.0


@register("codex_humanevalfim_single")
class CodexHumanEvalFIMSingle(Task):
    """HumanEval FIM with single-line masking (1k rows).

    Prompts follow the FIM pattern: <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
    The model fills in the masked single line.
    """

    data_source = DataSource(
        path="json",
        data_files=f"{_HF_DATA_BASE}/HumanEval-SingleLineInfilling.jsonl",
        split="train",
    )
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        num_samples=10,
        stop_sequences=OLMO_FIM.stop_sequences,
    )
    metrics = (
        PassAtKMetric(k=1, scorer=_FIMCodeScorer),
        PassAtKMetric(k=10, scorer=_FIMCodeScorer),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("train")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        prefix = doc["prompt"]
        suffix = doc["suffix"]

        # Build FIM prompt using OLMo FIM tokens
        fim_prompt = (
            OLMO_FIM.lead_token + prefix + OLMO_FIM.center_token + suffix + OLMO_FIM.end_token
        )

        test_code = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=fim_prompt,
            gold_answer=doc["canonical_solution"],
            metadata={
                "id": doc.get("task_id", str(index)),
                "prefix": prefix,
                "suffix": suffix,
                "entry_point": doc["entry_point"],
                "test": test_code,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        return output.text

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Reassemble code: prefix + model_output (middle) + suffix."""
        for response in responses:
            prefix = response.instance.metadata["prefix"]
            suffix = response.instance.metadata["suffix"]
            for output in response.outputs:
                middle = self.extract_answer(output)
                if middle is not None:
                    output.extracted_answer = prefix + middle + suffix
                else:
                    output.extracted_answer = None


@register("codex_humanevalfim_multi")
class CodexHumanEvalFIMMulti(CodexHumanEvalFIMSingle):
    """HumanEval FIM with multi-line masking (5.8k rows)."""

    data_source = DataSource(
        path="json",
        data_files=f"{_HF_DATA_BASE}/HumanEval-MultiLineInfilling.jsonl",
        split="train",
    )
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        num_samples=1,
        stop_sequences=OLMO_FIM.stop_sequences,
    )
    metrics = (PassAtKMetric(k=1, scorer=_FIMCodeScorer),)


@register("codex_humanevalfim_random")
class CodexHumanEvalFIMRandom(CodexHumanEvalFIMSingle):
    """HumanEval FIM with random-span masking (1.6k rows)."""

    data_source = DataSource(
        path="json",
        data_files=f"{_HF_DATA_BASE}/HumanEval-RandomSpanInfilling.jsonl",
        split="train",
    )
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        num_samples=5,
        stop_sequences=OLMO_FIM.stop_sequences,
    )
    metrics = (
        PassAtKMetric(k=1, scorer=_FIMCodeScorer),
        PassAtKMetric(k=5, scorer=_FIMCodeScorer),
    )


register_variant("codex_humanevalfim_single", "olmo3")
register_variant("codex_humanevalfim_multi", "olmo3")
register_variant("codex_humanevalfim_random", "olmo3")

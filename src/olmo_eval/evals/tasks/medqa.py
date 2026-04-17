"""MedQA evaluation task.

MedQA (davidheineman/medqa-en) is a USMLE-style medical multiple-choice QA benchmark.
The test split contains 1,273 questions.

Fields: question (str), choices (list[str]), answer_idx (int), answer (str).
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetricInstanceAvg
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import extract_mcq_answer
from olmo_eval.evals.tasks.common import Task, register, register_variant

_SYSTEM_PROMPT = """\
You are a medical expert. Answer the following multiple choice question by \
reasoning through the options and selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""

_DEFAULT_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_DEFAULT_METRICS = (_DEFAULT_ACCURACY,)
_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1024)


@register("medqa")
class MedQA(Task):
    """USMLE-style medical multiple-choice QA."""

    data_source = DataSource(path="davidheineman/medqa-en")
    split = Split.TEST
    formatter = MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT)
    metrics = _DEFAULT_METRICS
    primary_metric = _DEFAULT_ACCURACY
    sampling_params = _DEFAULT_SAMPLING

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a MedQA document to an Instance with shuffled choices."""
        question = doc.get("question", "")
        choices = doc.get("choices", [])
        answer_idx = doc.get("answer_idx")

        if not question or not choices or answer_idx is None:
            return None

        if not isinstance(answer_idx, int) or not (0 <= answer_idx < len(choices)):
            return None

        # Deterministic per-question shuffle, tracking original positions
        # so duplicate choice strings don't confuse gold lookup
        rng = random.Random(f"{self.config.seed}:{index}")
        paired = list(zip(choices, range(len(choices)), strict=True))
        rng.shuffle(paired)
        shuffled = [c for c, _ in paired]

        gold_idx = next(i for i, (_, orig) in enumerate(paired) if orig == answer_idx)
        gold_text = choices[answer_idx]
        gold_letter = chr(ord("A") + gold_idx)

        metadata: dict[str, Any] = {
            "id": f"medqa_{index}",
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": gold_text,
        }

        return Instance(
            question=question,
            gold_answer=gold_letter,
            choices=tuple(shuffled),
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract answers, with argmax for logprob-based MC scoring."""
        for response in responses:
            if (
                response.request.request_type == RequestType.LOGLIKELIHOOD
                and len(response.outputs) > 1
            ):
                best_idx = 0
                best_logprob = float("-inf")
                for i, output in enumerate(response.outputs):
                    meta = output.metadata or {}
                    logprob = meta.get("total_logprob", meta.get("sum_logits", float("-inf")))
                    if logprob > best_logprob:
                        best_logprob = logprob
                        best_idx = i
                letter = chr(ord("A") + best_idx)
                for output in response.outputs:
                    output.extracted_answer = letter
            else:
                for output in response.outputs:
                    output.extracted_answer = self.extract_answer(output)

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract an MCQ letter from model output."""
        return extract_mcq_answer(output.text)


register_variant("medqa", "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS)
register_variant("medqa", "bpb", formatter=PPLFormatter(), metrics=(BPBMetricInstanceAvg(),))

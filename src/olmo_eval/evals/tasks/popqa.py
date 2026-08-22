"""PopQA: open-domain QA over entities spanning a wide popularity range.

PopQA (https://arxiv.org/abs/2212.10511) probes factual recall with
template-generated questions from Wikidata triples, weighted toward
long-tail entities. A response is correct if any accepted alias of the
answer appears in it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.popqa import POPQA_FIXED_FEWSHOT


def _format_query(question: str) -> str:
    return f"Q: {question} A:"


@dataclass(frozen=True, slots=True)
class PopQAContainsScorer(Scorer):
    """Alias containment scorer for PopQA.

    A response is correct if any accepted alias appears in it verbatim,
    lowercased, or capitalized — the matching rule used by the reference
    implementation.
    """

    name: str = "popqa_contains"

    def score(self, instance: Instance, output: LMOutput) -> float:
        response = output.text or ""
        aliases = instance.metadata.get("aliases") or []
        if not aliases and instance.gold_answer:
            aliases = [instance.gold_answer]
        for alias in aliases:
            alias = str(alias)
            if alias in response or alias.lower() in response or alias.capitalize() in response:
                return 1.0
        return 0.0


@register("popqa")
class PopQA(Task):
    data_source = DataSource(path="akariasai/PopQA", split="test")
    split = Split.TEST
    metrics = (AccuracyMetric(scorer=PopQAContainsScorer),)
    num_fewshot = 15
    sampling_params = SamplingParams(
        max_tokens=15,
        temperature=0.0,
        do_sample=False,
        stop_sequences=("\n\n",),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = str(doc.get("question", ""))
        if not question:
            return None
        aliases = json.loads(doc["possible_answers"])

        return Instance(
            question=_format_query(question),
            gold_answer=str(doc["obj"]),
            metadata={
                "id": doc.get("id", index),
                "template_id": doc.get("prop_id"),
                "subject_popularity": doc.get("s_pop"),
                "aliases": aliases,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        instances = [
            Instance(
                question=_format_query(str(doc["question"])),
                gold_answer=str(doc["answer"]),
                metadata={"id": f"popqa_fixed_{index}"},
            )
            for index, doc in enumerate(POPQA_FIXED_FEWSHOT)
        ]
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        parts = [f"{ex.question} {ex.gold_answer}" for ex in self.get_fewshot()]
        parts.append(instance.question)
        prompt = "\n\n".join(parts)
        # Few-shot examples stay in a single user message under chat format,
        # matching the reference implementation.
        if self.request_type == RequestType.CHAT:
            return LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": prompt},),
            )
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)


# Chat variant for instruct and reasoning models. Drops stop sequences and
# raises the token budget so chain-of-thought reasoning is not truncated.
register_variant(
    "popqa",
    "chat",
    formatter=ChatFormatter(),
    sampling_params=SamplingParams(
        max_tokens=131072,
        temperature=0.6,
        top_p=0.95,
    ),
)

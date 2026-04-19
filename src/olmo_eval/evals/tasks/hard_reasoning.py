"""HardReasoning evaluation tasks.

Logic puzzles and reasoning tasks that require multi-step constraint satisfaction.

Dataset: allenai/hard-reasoning

Usage:
    olmo-eval run -m my-model -t hard_reasoning_bringing_toys
    olmo-eval run -m my-model -t hard_reasoning_bringing_toys:chat
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.evals.tasks.common import Task, register, register_variant

HARD_REASONING_TASKS: tuple[str, ...] = (
    "bringing_toys",
    "classroom_assignment",
    "dinner_party",
    "expense_splitting",
    "printing_jobs",
    "secret_santa",
    "social_gathering",
    "wedding_planning",
    "wedding_supplies",
)


def _extract_last_complete_json(s: str) -> dict | None:
    """Extract the last complete JSON object from a string."""
    stack: list[int] = []
    last_json_start: int | None = None
    last_json_str: str | None = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_json_str = s[last_json_start : i + 1]
                    last_json_start = None
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


@dataclass(frozen=True, slots=True)
class HardReasoningParsedScorer(Scorer):
    """Score 1.0 if the model output was parsed as valid JSON, else 0.0."""

    name: str = "parsed"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        try:
            json.loads(output.extracted_answer)
            return 1.0
        except (json.JSONDecodeError, TypeError):
            return 0.0


@dataclass(frozen=True, slots=True)
class HardReasoningScorer(Scorer):
    """Score using the np_hard_reasoning check() function."""

    name: str = "hard_reasoning_check"

    def score(self, instance: Instance, output: LMOutput) -> float:
        from np_hard_reasoning.scenarios.registry import SCENARIO_REGISTRY

        if output.extracted_answer is None:
            return 0.0
        subset = instance.metadata.get("subset", "")
        scenario_cls = SCENARIO_REGISTRY.get(subset)
        if scenario_cls is None:
            return 0.0
        try:
            scenario = scenario_cls.load_from_json(instance.metadata["scenario_data"])
            return 1.0 if scenario.check_json(str(output.extracted_answer)) else 0.0
        except Exception:
            return 0.0


class HardReasoningBase(Task):
    """Base class for HardReasoning logic puzzle tasks.

    Each subtask loads from a specific subset of the allenai/hard-reasoning dataset,
    where files are organized as {subset}/dev_t1.jsonl and {subset}/test_t1.jsonl.
    """

    subset: str = "bringing_toys"
    dependencies = [
        "git+https://github.com/allenai/np-hard-reasoning.git@fa8bbb2a5554e34a7ce051b71e9357e44dbabd0f",
        "z3-solver",
        "networkx",
    ]
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop_sequences=("\n\n",),
    )
    metrics = (
        AccuracyMetric(scorer=HardReasoningScorer),
        AccuracyMetric(name="parse_rate", scorer=HardReasoningParsedScorer),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_hard_reasoning_split(self.config.split))
        yield from self._instances_cache

    def _load_hard_reasoning_split(self, split: str) -> Iterator[Instance]:
        """Load instances from a specific split of the hard-reasoning dataset."""
        import json
        import os

        from huggingface_hub import hf_hub_download

        file_name = "dev_t1.jsonl" if split == "validation" else "test_t1.jsonl"
        local_path = hf_hub_download(
            repo_id="allenai/hard-reasoning",
            filename=f"{self.subset}/{file_name}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        with open(local_path) as f:
            for index, line in enumerate(f):
                doc = json.loads(line)
                instance = self.process_doc(doc, index)
                if instance is not None:
                    yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return Instance(
            question=doc["prompt"],
            metadata={
                "id": doc.get("id", index),
                "scenario_data": doc["instance"],
                "subset": self.subset,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the last complete JSON object from the model's response."""
        json_obj = _extract_last_complete_json(output.text)
        if json_obj is not None:
            return json.dumps(json_obj)
        return output.text.strip() or None

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the dev split."""
        import random

        if self.config.num_fewshot == 0:
            return []
        all_instances = list(self._load_hard_reasoning_split("validation"))
        if not all_instances:
            return []
        rng = random.Random(self.config.fewshot_seed)
        return rng.sample(all_instances, min(self.config.num_fewshot, len(all_instances)))


# =============================================================================
# Task Registration
# =============================================================================

for _subset in HARD_REASONING_TASKS:
    _task_name = f"hard_reasoning_{_subset}"
    _class_name = f"HardReasoning_{_subset.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (HardReasoningBase,),
        {
            "subset": _subset,
            "__module__": __name__,
            "__qualname__": _class_name,
        },
    )
    setattr(sys.modules[__name__], _class_name, _cls)
    register(_task_name)(_cls)
    register_variant(
        _task_name,
        "chat",
        formatter=ChatFormatter(),
        sampling_params=SamplingParams(max_tokens=32768, temperature=0.0),
    )
    register_variant(
        _task_name,
        "dev",
        split=Split.VALIDATION,
    )


__all__ = [
    "HARD_REASONING_TASKS",
    "HardReasoningBase",
]

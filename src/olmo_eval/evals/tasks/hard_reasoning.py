"""HardReasoning evaluation tasks.

Logic puzzles and reasoning tasks that require multi-step constraint satisfaction.

Dataset: allenai/hard-reasoning

Scoring uses the ``check()`` verifiers from the ``np-hard-reasoning`` package,
which is installed at runtime from the task's ``dependencies``.

Differences from the reference harness (``scripts/evaluation/run_model.py`` in
np-hard-reasoning):

- The reference scores the first JSON value in the response. These tasks score
  the last one that has the scenario's answer shape, so drafts written while
  reasoning do not count.
- The reference parses with ``json5``; these tasks use strict JSON.
- Text inside a ``<think>`` trace is never scored. An unterminated trace has no
  answer.

Usage:
    olmo-eval run -m my-model -t hard_reasoning_bringing_toys
    olmo-eval run -m my-model -t hard_reasoning_bringing_toys:chat
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
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

#: System prompt used by the reference harness.
_SYSTEM_PROMPT = "You are a problem solver."


def _json_values(text: str) -> list[Any]:
    """Return every top-level JSON object or array embedded in ``text``, in order."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            values.append(value)
            i = end
        else:
            i += 1
    return values


def _extract_last_valid_json(text: str, is_valid: Callable[[Any], bool]) -> Any | None:
    """Return the last JSON value in ``text`` that ``is_valid`` accepts.

    Objects are preferred over arrays, as in the reference harness, so a bare
    list mentioned after a final ``{"solution": ...}`` object is not scored.
    """
    values = [v for v in _json_values(text) if is_valid(v)]
    objects = [v for v in values if isinstance(v, dict)]
    if objects:
        return objects[-1]
    return values[-1] if values else None


def _scenario_class(subset: str) -> Any:
    from np_hard_reasoning.scenarios.registry import SCENARIO_REGISTRY

    try:
        return SCENARIO_REGISTRY[subset]
    except KeyError:
        raise ValueError(f"Unknown hard_reasoning subset: {subset!r}") from None


def _has_answer_shape(scenario_cls: Any, value: Any) -> bool:
    try:
        scenario_cls.load_answer_from_json(value)
    except (ValueError, TypeError, KeyError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class HardReasoningParsedScorer(Scorer):
    """Score 1.0 if an answer with the scenario's shape was extracted, else 0.0."""

    name: str = "parsed"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 0.0 if output.extracted_answer is None else 1.0


@dataclass(frozen=True, slots=True)
class HardReasoningScorer(Scorer):
    """Score using the np_hard_reasoning check() function."""

    name: str = "hard_reasoning_check"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.extracted_answer is None:
            return 0.0
        scenario_cls = _scenario_class(instance.metadata["subset"])
        # Errors here come from the dataset row, not the model, so they propagate.
        scenario = scenario_cls.load_from_json(instance.metadata["scenario_data"])
        answer = scenario_cls.load_answer_from_json(json.loads(output.extracted_answer))
        try:
            return 1.0 if scenario.check(answer) else 0.0
        except IndexError:
            # Some checkers index with the model's values before range-checking them.
            return 0.0


class HardReasoningBase(Task):
    """Base class for HardReasoning logic puzzle tasks.

    Each subtask loads from a specific subset of the allenai/hard-reasoning dataset,
    where files are organized as {subset}/dev_t1.jsonl and {subset}/test_t1.jsonl.
    """

    subset: str = "bringing_toys"
    dependencies = [
        "git+https://github.com/allenai/np-hard-reasoning.git@fa8bbb2a5554e34a7ce051b71e9357e44dbabd0f",
        "z3-solver==5.1.0.0",
        "networkx==3.7",
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
    primary_metric = metrics[0]

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
        """Extract the last JSON value with the scenario's answer shape."""
        text = output.text
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        elif "<think>" in text:
            # The reasoning never finished, so anything in the trace is a draft.
            output.metadata["answer_format_correct"] = False
            return None
        scenario_cls = _scenario_class(self.subset)
        answer = _extract_last_valid_json(text, lambda v: _has_answer_shape(scenario_cls, v))
        output.metadata["answer_format_correct"] = answer is not None
        return None if answer is None else json.dumps(answer)

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
        formatter=ChatFormatter(system_prompt=_SYSTEM_PROMPT),
        sampling_params=SamplingParams(max_tokens=32768, temperature=0.0),
        strip_thinking=True,
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

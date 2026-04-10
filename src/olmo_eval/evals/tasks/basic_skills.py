from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

_HF_BASE = "hf://datasets/allenai/basic-skills"


def _shuffle_and_insert(lst: list[str], value: str, rnd: random.Random) -> tuple[list[str], int]:
    shuffled = lst.copy()
    rnd.shuffle(shuffled)
    insert_index = rnd.randint(0, len(shuffled))
    shuffled.insert(insert_index, value)
    return shuffled, insert_index


BASIC_SKILLS_SUBTASKS = [
    "arithmetic",
    "coding",
    "common_knowledge",
    "logical_reasoning",
    "string_operations",
    "pattern",
]


class _BasicSkillsBase(Task):
    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    num_fewshot = 0
    split = Split.VALIDATION
    sampling_params = SamplingParams(temperature=0.0)
    fewshot_split = "validation"
    fewshot_sample = True

    _subset: str = "arithmetic"

    def _load_docs(self) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        data_file = f"{_HF_BASE}/{self._subset}/validation.json"
        ds = load_dataset("json", data_files=data_file, split="train")
        yield from ds

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = [
                inst
                for idx, doc in enumerate(self._load_docs())
                if (inst := self.process_doc(doc, idx)) is not None
            ]
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc["question"]
        answer = doc["answer"]
        wrong_answers = doc["wrong_answers"]

        choices, answer_index = _shuffle_and_insert(wrong_answers, answer, random.Random(doc["id"]))
        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=answer,
            metadata={
                "id": doc["id"],
                "index": index,
                "dataset": f"basic_skills_{self._subset}",
                "gold_idx": answer_index,
                "gold_text": answer,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.num_fewshot == 0:
            return []

        all_instances = [
            inst
            for idx, doc in enumerate(self._load_docs())
            if (inst := self.process_doc(doc, idx)) is not None
        ]

        if not all_instances:
            return []

        # Sample one extra so we can exclude the evaluated instance (matching oe-eval-internal)
        k = min(self.config.num_fewshot + 1, len(all_instances))
        rng = random.Random(self.config.fewshot_seed)
        return rng.sample(all_instances, k)

    def _is_bpb(self) -> bool:
        return isinstance(self.config.formatter, PPLFormatter)

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot_candidates = self.get_fewshot()

        # Exclude the current instance from fewshot (matching oe-eval-internal behavior)
        instance_id = instance.metadata.get("id")
        fewshot = [ex for ex in fewshot_candidates if ex.metadata.get("id") != instance_id][
            : self.config.num_fewshot
        ]

        parts: list[str] = []
        for ex in fewshot:
            answer = ex.gold_answer or ex.metadata.get("gold_text", "")
            parts.append(f"{ex.question} {answer}")

        parts.append(instance.question)
        prompt = "\n\n".join(parts)

        continuations = tuple(f" {c}" for c in (instance.choices or ()))
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


for _subtask in BASIC_SKILLS_SUBTASKS:
    _task_name = f"basic_skills_{_subtask}"
    _class_name = f"BasicSkills_{_subtask.title().replace('_', '')}"
    _cls = type(
        _class_name,
        (_BasicSkillsBase,),
        {
            "_subset": _subtask,
            "data_source": DataSource(
                path="json",
                data_files=f"{_HF_BASE}/{_subtask}/validation.json",
                split="train",
            ),
            "__module__": __name__,
            "__qualname__": _class_name,
        },
    )
    setattr(sys.modules[__name__], _class_name, _cls)
    register(_task_name)(_cls)
    register_variant(_task_name, "rc")
    register_variant(_task_name, "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))
    register_variant(
        _task_name,
        "olmo3base",
        num_fewshot=5,
        fewshot_seed=1234,
    )
    register_variant(
        _task_name,
        "olmes",
        num_fewshot=5,
        fewshot_seed=1234,
    )

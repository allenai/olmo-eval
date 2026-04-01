from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
SOCIALIQA_FIXED_FEWSHOT = [
    {"context": "Cameron decided to have a barbecue and gathered her friends together.", "question": "How would Others feel as a result?", "answerA": "like attending", "answerB": "like staying home", "answerC": "a good friend to have", "label": "1"},
    {"context": "Quinn wanted to help me clean my room up because it was so messy.", "question": "What will Quinn want to do next?", "answerA": "Eat messy snacks", "answerB": "help out a friend", "answerC": "Pick up the dirty clothes", "label": "3"},
    {"context": "Jan needed to give out jobs for an upcoming project at work.", "question": "What will Others want to do next?", "answerA": "disagree with Jan", "answerB": "get to work", "answerC": "argue with the assignments", "label": "2"},
    {"context": "Their cat kept trying to escape out of the window, so Jan placed an obstacle in the way.", "question": "How would Jan feel afterwards?", "answerA": "scared of losing the cat", "answerB": "normal", "answerC": "relieved for fixing the problem", "label": "3"},
    {"context": "Remy was an expert fisherman and was on the water with Kai. Remy baited Kai's hook.", "question": "What will Remy want to do next?", "answerA": "cast the line", "answerB": "put the boat in the water", "answerC": "invite Kai out on the boat", "label": "1"},
    {"context": "Kendall worked the weekend at the steakhouse and made bank on tips.", "question": "What will Kendall want to do next?", "answerA": "Save the money", "answerB": "get hired at the steakhouse", "answerC": "Quit her job", "label": "1"},
    {"context": "Bailey relieved every one of her friends when she announced her plans to stay.", "question": "Why did Bailey do this?", "answerA": "wanted to live by herself", "answerB": "wanted to show her dedication to her friends", "answerC": "wanted to forget about her friends", "label": "2"},
    {"context": "Kendall ran back and thanked Lee for helping her find the dog.", "question": "How would you describe Kendall?", "answerA": "grateful", "answerB": "super", "answerC": "amazing", "label": "1"},
]
# fmt: on


@register("socialiqa")
class SocialIQA(Task):
    data_source = DataSource(path="social_i_qa", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = self._load_socialiqa_instances()
        yield from self._instances_cache

    def _load_socialiqa_instances(self) -> list[Instance]:
        loader = DataLoader()
        instances: list[Instance] = []

        # When limit is set, load all splits (validation first, then train)
        # to match oe-eval-internal's split="all" ordering: test -> validation -> train.
        # SocialIQA has no test split, so: validation -> train.
        splits = ["validation", "train"] if self.config.limit else [self.config.split.value]

        index = 0
        for split in splits:
            source = self.config.get_data_source(split=split)
            for doc in loader.load(source):
                inst = self.process_doc(doc, index)
                if inst is not None:
                    instances.append(inst)
                    index += 1

        # Match oe-eval-internal's random subsampling: random.Random(1234).sample(docs, limit)
        if self.config.limit and len(instances) > self.config.limit:
            instances = random.Random(1234).sample(instances, self.config.limit)

        return instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        context = doc.get("context", "")
        question_text = doc.get("question", "")
        if not context or not question_text:
            return None

        question = f"{context} {question_text}"
        choices = (doc.get("answerA", ""), doc.get("answerB", ""), doc.get("answerC", ""))
        label = int(doc.get("label", "1")) - 1
        gold_text = choices[label] if 0 <= label < len(choices) else ""

        return Instance(
            question=question,
            choices=choices,
            gold_answer=str(label),
            metadata={
                "id": index,
                "index": index,
                "dataset": "socialiqa",
                "gold_idx": label,
                "gold_text": gold_text,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_socialiqa_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in SOCIALIQA_FIXED_FEWSHOT:
            question = f"{doc['context']} {doc['question']}"
            choices = (doc["answerA"], doc["answerB"], doc["answerC"])
            label = int(doc["label"]) - 1
            gold_text = choices[label] if 0 <= label < len(choices) else ""
            letter = chr(ord("A") + label)

            instances.append(
                Instance(
                    question=question,
                    choices=choices,
                    gold_answer=gold_text,
                    metadata={
                        "gold_idx": label,
                        "gold_text": gold_text,
                        "mc_answer": letter,
                    },
                )
            )
        # Take the first num_fewshot examples (no random sampling) to match
        # oe-eval-internal's fewshot_source behavior which uses [:k].
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[: self.config.num_fewshot]
        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


def _format_mc(question: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Question: {question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(question: str, answer: str | None = None) -> str:
    prompt = f"Question: {question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


register_variant("socialiqa", "rc")
register_variant("socialiqa", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "socialiqa",
    "olmo3base",
    data_source=DataSource(path="social_i_qa", split="train+validation"),
    num_fewshot=5,
    limit=10000,
    fewshot_source="olmes_socialiqa_fixed",
)
register_variant(
    "socialiqa",
    "xlarge",
    data_source=DataSource(path="social_i_qa", split="train+validation"),
    num_fewshot=5,
    limit=10000,
    fewshot_source="olmes_socialiqa_fixed",
)
register_variant("socialiqa", "olmes", num_fewshot=5, fewshot_source="olmes_socialiqa_fixed")
register_variant("socialiqa", "full")

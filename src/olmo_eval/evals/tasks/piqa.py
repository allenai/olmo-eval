from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
PIQA_FIXED_FEWSHOT = [
    {"goal": "how do you stab something?", "sol1": "stick a sharp object through it.", "sol2": "pin it with a sharp object.", "label": 0},
    {"goal": "how do you shake something?", "sol1": "move it up and down and side to side quickly.", "sol2": "stir it very quickly.", "label": 0},
    {"goal": "Clean tires", "sol1": "Pour water, cape off caked on dirt. Use  speed wool to clean out crevices and sparrow spaces.", "sol2": "Pour water, scrape off caked on dirt. Use a steel wool to clean out crevices and narrow spaces.", "label": 1},
    {"goal": "how do you taste something?", "sol1": "smell it enough to taste it.", "sol2": "place it in your mouth to taste.", "label": 1},
    {"goal": "To create a makeshift ice pack,", "sol1": "take a sponge and soak it in oil. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.", "sol2": "take a sponge and soak it in water. Put the sponge in a refrigerator and let it freeze. Once frozen, take it out and put it in a ziploc bag. You can now use it as an ice pack.", "label": 1},
    {"goal": "What should I use as a stain on a wooden bowl I've just made.", "sol1": "You should coat the wooden bowl with a butcher block oil & finish per manufacturer directions.", "sol2": "You should coat the wooden bowl with a butcher knife oil & finish per manufacturer directions.", "label": 0},
    {"goal": "How to boil eggs.", "sol1": "Place your eggs in a pot and cover with no water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.", "sol2": "Place your eggs in a pot and cover with cold water by 1 inch, bring to a boil over medium-high heat, then cover, remove from the heat and set aside 8 to 10 minutes.", "label": 1},
]
# fmt: on


@register("piqa")
class PiQA(Task):
    data_source = DataSource(path="piqa", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        goal = doc.get("goal", "")
        sol1 = doc.get("sol1", "")
        sol2 = doc.get("sol2", "")
        label = int(doc.get("label", 0))

        if not goal:
            return None

        choices = (sol1, sol2)
        gold_text = choices[label] if 0 <= label < len(choices) else ""

        return Instance(
            question=goal,
            choices=choices,
            gold_answer=str(label),
            metadata={
                "id": index,
                "index": index,
                "dataset": "piqa",
                "gold_idx": label,
                "gold_text": gold_text,
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_piqa_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in PIQA_FIXED_FEWSHOT:
            correct_sol = doc["sol1"] if doc["label"] == 0 else doc["sol2"]
            letter = chr(ord("A") + doc["label"])
            instances.append(
                Instance(
                    question=doc["goal"],
                    choices=(doc["sol1"], doc["sol2"]),
                    gold_answer=correct_sol,
                    metadata={
                        "gold_idx": doc["label"],
                        "gold_text": correct_sol,
                        "mc_answer": letter,
                    },
                )
            )
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            rng = random.Random(self.config.fewshot_seed)
            instances = rng.sample(instances, self.config.num_fewshot)
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


def _format_mc(goal: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Goal: {goal}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(goal: str, answer: str | None = None) -> str:
    prompt = f"Goal: {goal}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt

register_variant(
    "piqa",
    "mc_olmo3base",
    formatter=MultipleChoiceFormatter(),
    num_fewshot=5,
    fewshot_source="olmes_piqa_fixed",
)

register_variant(
    "piqa",
    "rc_olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_piqa_fixed",
)

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric, LogprobUncondMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_regime, register_variant

# fmt: off
CSQA_FIXED_FEWSHOT = [
    {
        "id": "61fe6e879ff18686d7552425a36344c8",
        "question": "Sammy wanted to go to where the people were.  Where might he go?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["race track", "populated areas", "the desert", "apartment", "roadblock"]},
        "answerKey": "B",
    },
    {
        "id": "02e821a3e53cb320790950aab4489e85",
        "question": "Google Maps and other highway and street GPS services have replaced what?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["united states", "mexico", "countryside", "atlas", "oceans"]},
        "answerKey": "D",
    },
    {
        "id": "23505889b94e880c3e89cff4ba119860",
        "question": "The fox walked from the city into the forest, what was it looking for?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["pretty flowers.", "hen house", "natural habitat", "storybook", "dense forest"]},
        "answerKey": "C",
    },
    {
        "id": "a76403b4921a9281b6ee2a7241a5ec9f",
        "question": "What is it called when you slowly cook using a grill?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["backyard", "restaurant", "crockpot", "neighbor's house", "barbeque"]},
        "answerKey": "E",
    },
    {
        "id": "6dc921840aa1e5dda3333b79007f630b",
        "question": "What would you do if you want to be able to earn money?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["apply for job", "stand in line", "take care of proposals", "pass course", "play the lottery"]},
        "answerKey": "A",
    },
    {
        "id": "e8a8b3a2061aa0e6d7c6b522e9612824",
        "question": "What home entertainment equipment requires cable?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["radio shack", "substation", "cabinet", "television", "desk"]},
        "answerKey": "D",
    },
    {
        "id": "527e72eb38950b8031ee6217ef531960",
        "question": "Of all the rooms in a house it was his favorite, the aromas always drew him to the what?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["yard", "basement", "kitchen", "living room", "garden"]},
        "answerKey": "C",
    },
]
# fmt: on


@register("csqa")
class CommonsenseQA(Task):
    data_source = DataSource(path="commonsense_qa", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 0
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)

    @property
    def instances(self) -> Iterator[Instance]:
        split = (
            self.config.data_source.split
            if isinstance(self.config.data_source, DataSource)
            else None
        )
        yield from self._load_instances_cached(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        if not question:
            return None

        choices_data = doc.get("choices", {})
        choices = choices_data.get("text", [])
        if not choices:
            return None

        answer_key = doc.get("answerKey", "")
        gold_idx = ord(answer_key) - ord("A") if answer_key else 0
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        return Instance(
            question=question,
            choices=tuple(choices),
            gold_answer=answer_key,
            metadata={
                "id": doc.get("id", f"csqa_{index}"),
                "index": index,
                "dataset": "csqa",
                "gold_idx": gold_idx,
                "gold_text": gold_text,
                "num_choices": len(choices),
            },
        )

    def _build_fewshot(self) -> list[Instance]:
        if self.config.fewshot_source == "olmes_csqa_fixed":
            return self._build_fixed_fewshot()
        return super()._build_fewshot()

    def _build_fixed_fewshot(self) -> list[Instance]:
        instances = []
        for doc in CSQA_FIXED_FEWSHOT:
            question = str(doc["question"])
            choices_data = doc["choices"]
            assert isinstance(choices_data, dict)
            choices = tuple(choices_data["text"])
            answer_key = str(doc["answerKey"])
            gold_idx = ord(answer_key) - ord("A")
            gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

            instances.append(
                Instance(
                    question=question,
                    choices=choices,
                    gold_answer=gold_text,
                    metadata={
                        "gold_idx": gold_idx,
                        "gold_text": gold_text,
                        "mc_answer": answer_key,
                    },
                )
            )
        if self.config.num_fewshot and self.config.num_fewshot < len(instances):
            instances = instances[:self.config.num_fewshot]
        return instances

    def _uses_uncond_metric(self) -> bool:
        """Check if any configured metric requires unconditional normalization."""
        return any(
            isinstance(m, LogprobUncondMCAccuracyMetric)
            for m in self.config.metrics
        )

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

        # For unconditional normalization (acc_uncond), generate both conditioned
        # and unconditional continuations using continuation_prompts.
        if not is_mc and self._uses_uncond_metric():
            uncond_prompt = "Answer:"
            num_choices = len(continuations)
            # Double continuations: first N conditioned, next N unconditional
            all_continuations = continuations + continuations
            # Per-continuation prompts: conditioned use full context, uncond use "Answer:"
            all_cont_prompts = tuple([prompt] * num_choices + [uncond_prompt] * num_choices)
            return LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt=prompt,
                continuations=all_continuations,
                continuation_prompts=all_cont_prompts,
            )

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


register_variant("csqa", "rc", metrics=(LogprobUncondMCAccuracyMetric(),))
register_variant("csqa", "mc", formatter=MultipleChoiceFormatter())
register_variant(
    "csqa",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_csqa_fixed",
    data_source=DataSource(path="commonsense_qa", split="train+validation"),
    limit=10000,
    metrics=(LogprobUncondMCAccuracyMetric(),),
)
# Register olmo3base as a regime (without metrics) so that combining with other variants
# like mc or bpb (e.g. csqa:mc::olmo3base) preserves the variant's metrics.
register_regime(
    "csqa",
    "olmo3base",
    num_fewshot=5,
    fewshot_source="olmes_csqa_fixed",
    data_source=DataSource(path="commonsense_qa", split="train+validation"),
    limit=10000,
)
register_variant(
    "csqa",
    "xlarge",
    data_source=DataSource(path="commonsense_qa", split="train+validation"),
    num_fewshot=5,
    limit=10000,
    fewshot_source="olmes_csqa_fixed",
)
register_variant("csqa", "bpb", metrics=(BPBMetric(),), primary_metric=BPBMetric())
register_variant("csqa", "olmes", num_fewshot=5, fewshot_source="olmes_csqa_fixed", metrics=(LogprobUncondMCAccuracyMetric(),))
register_variant("csqa", "full")

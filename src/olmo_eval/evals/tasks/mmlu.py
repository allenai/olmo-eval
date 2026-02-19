from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceLogprobFormatter
from olmo_eval.common.metrics import MultipleChoiceLogprobMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, TaskConfig, register

_STEM = (
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
)

_HUMANITIES = (
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
)

_SOCIAL_SCIENCES = (
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
)

_OTHER = (
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
)

MMLU_SUBJECTS = _STEM + _HUMANITIES + _SOCIAL_SCIENCES + _OTHER

DEFAULT_MMLU_PATH = "cais/mmlu"


def _make_mcq_prompt(question: str, choices: list[str], label_prefix: str = " ") -> str:
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{label_prefix}{letter}. {c}")
    return "\n".join(lines)


def _answer_to_index_and_letter(answer: int | str) -> tuple[int, str]:
    """Convert dataset answer (int 0-3 or str A-D) to (gold_idx, letter)."""
    if isinstance(answer, int):
        if not 0 <= answer <= 4:
            raise ValueError(f"MMLU answer index must be 0-4, got {answer}")
        return answer, chr(ord("A") + answer)
    s = str(answer).strip().upper()
    if len(s) != 1 or s not in "ABCDE":
        raise ValueError(f"MMLU answer letter must be A-E, got {answer!r}")
    return ord(s) - ord("A"), s


class MMLUMCTask(Task):
    default_source: str = DEFAULT_MMLU_PATH
    fewshot_split: str = "dev"
    fewshot_sample: bool = False  # Fixed order (first k) as in reference

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        yield from self._load_instances_cached(split="test")

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a cais/mmlu document to an Instance."""
        question = doc.get("question", "")
        choices = list(doc.get("choices", []))
        if not choices or len(choices) > 5:
            return None
        answer = doc.get("answer", 0)
        try:
            gold_idx, letter = _answer_to_index_and_letter(answer)
        except ValueError:
            return None
        if gold_idx >= len(choices):
            return None
        choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        query = _make_mcq_prompt(question, choices, label_prefix=" ")
        return Instance(
            question=query,
            gold_answer=letter,
            choices=tuple(choice_labels),
            metadata={
                "id": index,
                "gold_idx": gold_idx,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format using the task's formatter (with fewshot if configured)."""
        formatter = self.config.formatter
        if formatter is None:
            raise ValueError(
                "MMLU MC task requires a formatter (e.g. MultipleChoiceLogprobFormatter)"
            )
        return formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> Any:
        """Not used for logprob-based MC; scoring uses MultipleChoiceLogprobScorer."""
        return None

    def _build_fewshot(self) -> list[Instance]:
        """Few-shot from dev split in fixed order (first k), matching reference."""
        all_fewshot = self._build_fewshot_from_source(
            split=self.fewshot_split,
            sample=self.fewshot_sample,
            fallback_splits=[],
        )
        k = self.config.num_fewshot
        return all_fewshot[:k] if k else all_fewshot


_MMLU_FORMATTER = MultipleChoiceLogprobFormatter(
    template="{question}",
    label_prefix=" ",
    include_choices_in_prompt=False,
    answer_suffix="\n\nAnswer:",
)

# Register one task per subject (mmlu_abstract_algebra, mmlu_anatomy, ...)
for _subject in MMLU_SUBJECTS:
    _cls = type(
        f"MMLU_{_subject}",
        (MMLUMCTask,),
        {
            "data_source": DataSource(path=DEFAULT_MMLU_PATH, subset=_subject, split="test"),
            "formatter": _MMLU_FORMATTER,
            "metrics": (MultipleChoiceLogprobMetric(),),
            "primary_metric": MultipleChoiceLogprobMetric(),
            "num_fewshot": 5,
            "split": Split.TEST,
            "sampling_params": SamplingParams(max_tokens=1, temperature=0.0),
            "__module__": __name__,
            "__qualname__": f"MMLU_{_subject}",
        },
    )
    register(f"mmlu_{_subject}")(_cls)
    globals()[f"MMLU_{_subject}"] = _cls

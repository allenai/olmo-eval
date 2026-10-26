"""
HaluEval Evaluation Tasks

This module implements HaluEval's hallucination recognition tasks for question answering,
knowledge-grounded dialogue and summarization. The model is shown a sample's output and
answers "Yes" if it contains hallucinated information and "No" otherwise.

Paper: https://arxiv.org/abs/2305.11747
Reference: https://github.com/RUCAIBox/HaluEval/tree/b7253db3cdaa0ab2c382f92b26b390109174f77e/evaluation

Prompts, temperature and the Yes/No parsing follow the reference's chat (gpt-3.5-turbo)
path; its completion path, used for other OpenAI models, is not ported. An answer holding
both or neither of "Yes" and "No" counts as wrong, as in the reference. Deviations from the
reference:

- The reference judges either the right or the hallucinated output of each sample, chosen
  by an unseeded coin flip. Both are judged here, so each task has two instances per
  sample and, over a full run, as many right outputs as hallucinated ones.
- Reasoning traces are removed before the Yes/No parsing, and a trace that never closed
  gives no judgement; a trace cut short after a ``<think>`` that the chat template wrote is
  not detected and is parsed as the answer.
- The ``knowledge`` variants add the dataset's ``knowledge`` column, the sample's
  Wikipedia knowledge, to the user message, after the paper's knowledge retrieval
  experiment, whose prompt is not published. The field is formatted as in the reference's
  ``generation/filtering.py``; the system prompt, instruction and demonstrations are
  unchanged, so the paper's "w/ Knowledge" numbers are not a parity target.
- The paper's fourth task, recognizing hallucinations in human-annotated ChatGPT replies to
  general user queries, and its chain-of-thought and sample-contrast strategies have no
  evaluation code or prompt in the reference and are not ported.
- Few-shot examples are rejected: HaluEval's demonstrations are part of its instructions.

Example command to run:

olmo-eval beaker launch \
  --harness default \
  -m allenai/Olmo-3-7B-Instruct \
  -t halueval_qa \
  -w "ai2/olmo-eval-debug" \
  -B "ai2/oe-other" \
  --cluster h100 \
  --inspect
"""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.scorers import ExactMatchScorer, Scorer
from olmo_eval.common.types import Instance, LMRequest, Response, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.constants.halueval import (
    DIALOGUE_INSTRUCTION,
    DIALOGUE_SYSTEM_PROMPT,
    QA_INSTRUCTION,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_INSTRUCTION,
    SUMMARIZATION_SYSTEM_PROMPT,
)
from olmo_eval.evals.tasks.omniscience import extract_final_answer

logger = logging.getLogger(__name__)

# Its qa, dialogue and summarization files hold the reference's data/<task>_data.json unchanged
HALUEVAL_REPO = "pminervini/HaluEval"
HALUEVAL_REVISION = "12a856119f03975a94509091e8cada3e6be6ead7"

# The reference samples greedily and sets no token limit
DEFAULT_SAMPLING = SamplingParams(max_tokens=None, temperature=0.0)

HALLUCINATED = "Yes"
NOT_HALLUCINATED = "No"
JUDGEMENTS = (HALLUCINATED, NOT_HALLUCINATED)

# =============================================================================
# Prompt Formatting
# =============================================================================


@dataclass(slots=True)
class HaluEvalFormatter(ChatFormatter):
    """Chat formatter that fills ``user_template`` with the fields of a HaluEval sample.

    ``{question}`` is the instance's question (the question, dialogue history or document
    the judged output responds to), and the other placeholders name keys of its ``fields``
    metadata, so a variant shows the model more of a sample by changing the template alone.
    HaluEval's demonstrations are part of its instructions, so few-shot examples are rejected.
    """

    def format(
        self,
        instance: Instance,
        fewshot: list[Instance] | None = None,
    ) -> LMRequest:
        if fewshot:
            raise ValueError("HaluEval's demonstrations are in its instructions; use num_fewshot=0")
        content = self.user_template.format(
            question=instance.question, **instance.metadata["fields"]
        )
        return ChatFormatter(system_prompt=self.system_prompt).format(
            replace(instance, question=content)
        )


def _user_template(instruction: str, fields: str) -> str:
    """Return the reference's user message: the instruction, the sample and the answer cue."""
    escaped = instruction.replace("{", "{{").replace("}", "}}")
    return f"{escaped}\n\n{fields}\n#Your Judgement#: "


QA_FIELDS = "#Question#: {question}\n#Answer#: {answer}"
DIALOGUE_FIELDS = "#Dialogue History#: {question}\n#Response#: {response}"
SUMMARIZATION_FIELDS = "#Document#: {question}\n#Summary#: {summary}"
KNOWLEDGE_FIELD = "#Knowledge#: {knowledge}\n"

QA_FORMATTER = HaluEvalFormatter(
    system_prompt=QA_SYSTEM_PROMPT,
    user_template=_user_template(QA_INSTRUCTION, QA_FIELDS),
)
QA_KNOWLEDGE_FORMATTER = HaluEvalFormatter(
    system_prompt=QA_SYSTEM_PROMPT,
    user_template=_user_template(QA_INSTRUCTION, KNOWLEDGE_FIELD + QA_FIELDS),
)
DIALOGUE_FORMATTER = HaluEvalFormatter(
    system_prompt=DIALOGUE_SYSTEM_PROMPT,
    user_template=_user_template(DIALOGUE_INSTRUCTION, DIALOGUE_FIELDS),
)
DIALOGUE_KNOWLEDGE_FORMATTER = HaluEvalFormatter(
    system_prompt=DIALOGUE_SYSTEM_PROMPT,
    user_template=_user_template(DIALOGUE_INSTRUCTION, KNOWLEDGE_FIELD + DIALOGUE_FIELDS),
)
SUMMARIZATION_FORMATTER = HaluEvalFormatter(
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
    user_template=_user_template(SUMMARIZATION_INSTRUCTION, SUMMARIZATION_FIELDS),
)

# =============================================================================
# Answer Extraction and Metrics
# =============================================================================


def extract_halueval_judgement(text: str) -> str:
    """Return the judgement the reference reads from a response, or "" if it reads none.

    As in the reference, periods are removed and the response is searched for the
    case-sensitive substrings "Yes" and "No", so "Not" or "Nolan" also read as "No". A
    response holding exactly one of them gives that judgement; one holding both or
    neither gives none. Only the text after a reasoning trace is searched.
    """
    answer = extract_final_answer(text).replace(".", "")
    found = [judgement for judgement in JUDGEMENTS if judgement in answer]
    return found[0] if len(found) == 1 else ""


@dataclass(frozen=True, slots=True)
class _GoldJudgementAccuracyMetric(Metric):
    """Accuracy over the instances whose gold judgement is ``gold``.

    Instances with the other gold judgement, or left unscored, have no value. Reports
    0.0, with a warning, when no instance has a value.
    """

    scorer: type[Scorer] | Scorer = ExactMatchScorer
    gold: ClassVar[str]

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        if not values:
            logger.warning(
                "%s has no scored responses; its value of 0.0 is not a measured score.",
                self.name,
            )
            return 0.0
        return sum(values) / len(values)

    def compute_instance(self, response: Response) -> float | None:
        if response.instance.gold_answer != self.gold:
            return None
        value = response.scores.get(self.scorer().name)
        return None if value is None else float(value)

    def supports_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class HallucinatedAccuracyMetric(_GoldJudgementAccuracyMetric):
    """Share of hallucinated outputs the model judged hallucinated."""

    name: str = "hallucinated_accuracy"
    gold: ClassVar[str] = HALLUCINATED


@dataclass(frozen=True, slots=True)
class RightAccuracyMetric(_GoldJudgementAccuracyMetric):
    """Share of right outputs the model judged free of hallucination."""

    name: str = "right_accuracy"
    gold: ClassVar[str] = NOT_HALLUCINATED


@dataclass(frozen=True, slots=True)
class ParseFailureRateMetric(Metric):
    """Share of outputs from which no judgement was read.

    These outputs score 0 for accuracy, as in the reference.
    """

    name: str = "parse_failure_rate"
    scorer: type[Scorer] | Scorer = ExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        if not response.outputs:
            return None
        failures = [output.extracted_answer not in JUDGEMENTS for output in response.outputs]
        return sum(failures) / len(failures)

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


ACCURACY = AccuracyMetric(scorer=ExactMatchScorer)
METRICS: tuple[Metric, ...] = (
    ACCURACY,
    HallucinatedAccuracyMetric(),
    RightAccuracyMetric(),
    ParseFailureRateMetric(),
)

# =============================================================================
# Task Definitions
# =============================================================================


def _data_source(subset: str) -> DataSource:
    """Return the pinned data file of one of the reference's paired subsets."""
    return DataSource(
        path=HALUEVAL_REPO,
        data_files=f"{subset}/data-00000-of-00001.parquet",
        revision=HALUEVAL_REVISION,
        split="train",
    )


class _HaluEval(Task):
    """Base for the HaluEval recognition tasks.

    Each dataset row pairs a right output with a hallucinated one. Both become instances,
    the right one first, with gold judgements "No" and "Yes".
    """

    split = Split.TRAIN  # Parquet data files load as a single "train" split
    sampling_params = DEFAULT_SAMPLING
    answer_extractor = extract_halueval_judgement
    metrics = METRICS
    primary_metric = ACCURACY

    #: Column holding what the judged output responds to.
    input_column: ClassVar[str]
    #: Name of the judged output; its columns are ``right_<name>`` and ``hallucinated_<name>``.
    output_name: ClassVar[str]

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Yield the right and then the hallucinated sample of every dataset row."""
        from olmo_eval.data import DataLoader

        source = self.config.get_data_source(split=split)
        for index, doc in enumerate(DataLoader().load(source)):
            for sample in self._samples(doc):
                yield self.process_doc(sample, index)

    def _samples(self, doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a paired row into its right and hallucinated samples.

        A sample keeps the row's shared columns, holds its output under ``output_name``
        and is labelled by ``hallucination``, as in the dataset's ``*_samples`` configs.
        """
        right, hallucinated = f"right_{self.output_name}", f"hallucinated_{self.output_name}"
        shared = {key: value for key, value in doc.items() if key not in (right, hallucinated)}
        return (
            {**shared, self.output_name: doc[right], "hallucination": "no"},
            {**shared, self.output_name: doc[hallucinated], "hallucination": "yes"},
        )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a sample, one output labelled by ``hallucination``, to an Instance."""
        label = doc["hallucination"]
        if label not in ("yes", "no"):
            raise ValueError(f"HaluEval row {index} has an unknown hallucination label {label!r}")
        hallucinated = label == "yes"
        fields = {self.output_name: doc[self.output_name]}
        if "knowledge" in doc:
            fields["knowledge"] = doc["knowledge"]
        return Instance(
            question=doc[self.input_column],
            gold_answer=HALLUCINATED if hallucinated else NOT_HALLUCINATED,
            metadata={
                "id": f"{index}-{'hallucinated' if hallucinated else 'right'}",
                "index": index,
                "hallucinated": hallucinated,
                "fields": fields,
            },
        )

    def get_fewshot(self) -> list[Instance]:
        """Return no examples, rejecting a config that asks for some.

        Raises:
            ValueError: If ``num_fewshot`` is set, since HaluEval's demonstrations are in
                its instructions.
        """
        if self.config.num_fewshot:
            raise ValueError("HaluEval's demonstrations are in its instructions; use num_fewshot=0")
        return []

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is None:
            raise ValueError(f"{self.config.name} needs a formatter")
        return self.config.formatter.format(instance, self.get_fewshot())


@register("halueval_qa")
class HaluEvalQA(_HaluEval):
    """Recognize hallucinated answers to HotpotQA questions."""

    data_source = _data_source("qa")
    formatter = QA_FORMATTER
    input_column = "question"
    output_name = "answer"


@register("halueval_dialogue")
class HaluEvalDialogue(_HaluEval):
    """Recognize hallucinated responses in OpenDialKG dialogues."""

    data_source = _data_source("dialogue")
    formatter = DIALOGUE_FORMATTER
    input_column = "dialogue_history"
    output_name = "response"


@register("halueval_summarization")
class HaluEvalSummarization(_HaluEval):
    """Recognize hallucinated summaries of CNN/Daily Mail articles."""

    data_source = _data_source("summarization")
    formatter = SUMMARIZATION_FORMATTER
    input_column = "document"
    output_name = "summary"


# =============================================================================
# Variant Registrations
# =============================================================================

register_variant("halueval_qa", "knowledge", formatter=QA_KNOWLEDGE_FORMATTER)
register_variant("halueval_dialogue", "knowledge", formatter=DIALOGUE_KNOWLEDGE_FORMATTER)

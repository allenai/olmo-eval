"""Question Answering task implementations."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    F1Metric,
    F1Scorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class DROPTask(Task):
    """DROP (Discrete Reasoning Over Paragraphs) reading comprehension task."""

    default_hf_path: str = "EleutherAI/drop"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
            )

    def _get_primary_answer(self, ans: dict[str, Any]) -> str:
        """Extract the primary answer from DROP's answer structure."""
        if ans.get("spans"):
            return ans["spans"][0]
        if ans.get("number"):
            return str(ans["number"])
        if ans.get("date", {}).get("year"):
            date = ans["date"]
            date_parts = [date[p] for p in ["day", "month", "year"] if date.get(p)]
            return " ".join(str(p) for p in date_parts)
        return ""

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        answer = self._get_primary_answer(doc["answer"])
        question = f"Passage: {doc['passage']}\n{doc['question']}"

        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "id": doc.get("query_id", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"{instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


class CoQATask(Task):
    """CoQA (Conversational Question Answering) task."""

    default_hf_path: str = "EleutherAI/coqa"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                self._instances_cache.extend(self._process_doc_to_multi(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
                revision="refs/convert/parquet",
            )

    def _process_doc_to_multi(self, doc: dict[str, Any]) -> list[Instance]:
        """Convert a multi-turn document to multiple Instances."""
        story = doc["story"]
        questions = doc["questions"]["input_text"]
        answers = doc["answers"]["input_text"]
        additional_answers = [v["input_text"] for v in doc["additional_answers"].values()]

        previous_qa: list[dict[str, str]] = []
        instances = []

        for idx, q in enumerate(questions):
            # Primary answer plus any additional answers
            ans_candidates = [answers[idx]] + [
                aa[idx] for aa in additional_answers if len(aa) > idx and aa[idx]
            ]

            query = f"Passage: {story}"
            if previous_qa:
                query += "\nPreceding questions:"
                for prev in previous_qa:
                    query += f"\nQuestion: {prev['q']}\nAnswer: {prev['a']}"
            query += f"\nQuestion: {q}"

            instances.append(
                Instance(
                    question=query,
                    gold_answer=ans_candidates[0],
                    metadata={
                        "id": f"{doc['id']}_turn{idx}",
                        "source": doc.get("source", ""),
                    },
                )
            )
            previous_qa.append({"q": q, "a": ans_candidates[0]})

        return instances

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"{instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


class SQuADTask(Task):
    """SQuAD (Stanford Question Answering Dataset) task."""

    default_hf_path: str = "rajpurkar/squad"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        answer = doc["answers"]["text"][0]
        question = f"Title: {doc['title']}\nBackground: {doc['context']}\n{doc['question']}"

        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "id": doc["id"],
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"{instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


class NaturalQuestionsTask(Task):
    """Natural Questions Open dataset task."""

    default_hf_path: str = "google-research-datasets/nq_open"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the validation split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("validation")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        answer = doc["answer"][0]

        return Instance(
            question=doc["question"],
            gold_answer=answer,
            metadata={},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"Question: {instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


class JeopardyTask(Task):
    """Jeopardy QA task."""

    default_hf_path: str = "soldni/jeopardy"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("train")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset="mosaicml_gauntlet",
                split=split,
            )

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        question = f"Category: {doc['category']}\n{doc['question']}"

        return Instance(
            question=question,
            gold_answer=doc["answer"],
            metadata={
                "id": doc.get("id", ""),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"{instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


# Task configurations


def _drop_config() -> TaskConfig:
    return TaskConfig(
        name="drop",
        data_source=DataSource(path="EleutherAI/drop"),
        scorers=(F1Scorer(),),
        metrics=(F1Metric(),),
    )


def _coqa_config() -> TaskConfig:
    return TaskConfig(
        name="coqa",
        data_source=DataSource(path="EleutherAI/coqa"),
        scorers=(F1Scorer(),),
        metrics=(F1Metric(),),
    )


def _squad_config() -> TaskConfig:
    return TaskConfig(
        name="squad",
        data_source=DataSource(path="rajpurkar/squad"),
        scorers=(F1Scorer(),),
        metrics=(F1Metric(),),
    )


def _naturalqs_config() -> TaskConfig:
    return TaskConfig(
        name="naturalqs",
        data_source=DataSource(path="google-research-datasets/nq_open"),
        scorers=(F1Scorer(),),
        metrics=(F1Metric(),),
    )


def _jeopardy_config() -> TaskConfig:
    return TaskConfig(
        name="jeopardy",
        data_source=DataSource(path="soldni/jeopardy", subset="mosaicml_gauntlet"),
        scorers=(F1Scorer(),),
        metrics=(F1Metric(),),
    )


# Register tasks


@register("drop", _drop_config)
class DROP(DROPTask):
    """DROP reading comprehension task."""

    pass


@register("coqa", _coqa_config)
class CoQA(CoQATask):
    """CoQA conversational QA task."""

    pass


@register("squad", _squad_config)
class SQuAD(SQuADTask):
    """SQuAD reading comprehension task."""

    pass


@register("naturalqs", _naturalqs_config)
class NaturalQuestions(NaturalQuestionsTask):
    """Natural Questions Open task."""

    pass


@register("jeopardy", _jeopardy_config)
class Jeopardy(JeopardyTask):
    """Jeopardy QA task."""

    pass

"""Shared plumbing for long-context benchmarks generated from a task table.

RULER and HELMET both register one task per (task type, context length), named
`{prefix}{task_type}__{context_size}`, and both render prompts from a pair of
templates: a user prompt and an answer prefix appended after it. This module
holds what they have in common.
"""

from collections.abc import Iterator
from typing import Any, ClassVar, cast

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.evals.tasks.common.base import Task, TaskConfig


def long_context_sampling_params(
    max_tokens: int, stop_sequences: tuple[str, ...] | None = None
) -> SamplingParams:
    """Greedy decoding with a per-task generation budget."""
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_sequences=stop_sequences,
    )


class LongContextTask(Task):
    """Base for tasks whose settings come from a row of a task table.

    A subclass names its table and name prefix, and implements
    `_load_dataset`. The rest can be adjusted through `render_prompt`,
    `instance_id` and `scoring_metadata`.
    """

    name_prefix: ClassVar[str] = ""
    task_table: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self.task_name = config.name.removeprefix(self.name_prefix)
        self.task_config = self.task_table[self.task_name]

        task_type, context_size = self.task_name.rsplit("__", 1)
        self.task_type = task_type
        self.context_size = int(context_size)

        self._dataset: list[dict[str, Any]] | None = None
        self._templates: dict[str, str] | None = None

    def _load_dataset(self) -> dict[str, Any]:
        """Return the loader payload: `data` plus `user_template` and `system_template`."""
        raise NotImplementedError

    def _load_data(self) -> None:
        if self._dataset is not None:
            return
        loaded = self._load_dataset()
        self._dataset = loaded["data"]
        self._templates = {
            "user": loaded["user_template"],
            "system": loaded["system_template"],
        }

    @property
    def instances(self) -> Iterator[Instance]:
        self._load_data()

        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        self._instances_cache = []
        for idx, doc in enumerate(self._dataset or []):
            instance = self.process_doc(cast(dict[str, Any], doc), index=idx)
            if instance is not None:
                self._instances_cache.append(instance)
                yield instance

    def render_prompt(self, doc: dict[str, Any]) -> tuple[str, str]:
        """Return the prompt and the answer prefix to append after it.

        With a chat template the answer prefix is left off rather than fed in
        as a partial assistant turn.
        """
        if self._templates is None:
            raise RuntimeError("Templates not loaded. Call _load_data() first.")
        question = self._templates["user"].format(**doc)
        if self.task_config.get("use_chat_template"):
            return question, ""
        return question, self._templates["system"].format(**doc)

    def instance_id(self, doc: dict[str, Any], index: int) -> Any:
        """Identifier recorded in each instance's metadata."""
        return index

    def scoring_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Extra fields a scorer needs from the row, kept out of the prompt."""
        return {}

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question, prepend_text = self.render_prompt(doc)
        answer = doc.get("answer")

        metadata: dict[str, Any] = {
            "id": self.instance_id(doc, index),
            "task_type": self.task_type,
            "context_size": self.context_size,
            "prepend_text": prepend_text,
            "tag": self.task_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer
        metadata.update(self.scoring_metadata(doc))

        return Instance(question=question, gold_answer=answer, metadata=metadata)

    @property
    def request_type(self) -> RequestType:
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            # PPL formatters score a single continuation, so join list answers
            if isinstance(self.config.formatter, PPLFormatter) and isinstance(
                instance.gold_answer, list
            ):
                instance = Instance(
                    question=instance.question,
                    gold_answer=", ".join(str(a) for a in instance.gold_answer),
                    metadata=instance.metadata,
                )
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = instance.question
        prepend_text = (instance.metadata or {}).get("prepend_text", "")
        if prepend_text:
            prompt = prompt + "\n" + prepend_text

        return LMRequest(request_type=self.request_type, prompt=prompt)

    def extract_answer(self, output: LMOutput) -> Any:
        """The raw generation; these tasks score substrings of it."""
        return output.text

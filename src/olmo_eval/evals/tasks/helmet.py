"""HELMET-plus: long-context extension of HELMET, up to 2m tokens.

Standard HELMET (https://github.com/princeton-nlp/HELMET) tops out at 128k
tokens of context. HELMET-plus extends select synthetic subsets further,
currently the `json_kv` recall task (JSON key-value retrieval, based on
https://github.com/nelson-liu/lost-in-the-middle), calibrated up to 2m
tokens against the Olmo 3 tokenizer. Data is published as the ai2-internal
`allenai/helmet-plus` dataset on the Hub.

The ICL tasks are carried over from standard HELMET at its original 4k-128k
lengths. They aren't length-extendable the way `json_kv` is -- their context
is built from real labelled examples, so length is capped by how many
demonstrations the source datasets actually contain -- but they're included
here so a HELMET run covers more than recall alone.
"""

import re
from collections.abc import Iterator
from typing import Any, cast

from olmo_eval.common.metrics import AccuracyMetric, RecallMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data.helmet_icl_loader import load_icl_dataset
from olmo_eval.data.helmet_loader import load_json_kv_dataset
from olmo_eval.data.helmet_tasks import HELMET_TASKS
from olmo_eval.evals.tasks.common.base import Task, TaskConfig
from olmo_eval.evals.tasks.common.registry import register


class HelmetTask(Task):
    """Shared plumbing for HELMET-plus tasks.

    Subclasses implement `_load_dataset` to fetch their data; everything from
    prompt assembly onward is common, since all HELMET tasks share the same
    `user_template` / `system_template` prompt shape.
    """

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        task_name = config.name.removeprefix("helmet_")
        self.task_name = task_name
        self.helmet_config = HELMET_TASKS[task_name]

        task_type, context_size_str = task_name.rsplit("__", 1)
        self.task_type = task_type
        self.context_size = int(context_size_str)

        self._dataset = None
        self._templates = None

    def _load_dataset(self) -> dict[str, Any]:
        """Return the loader payload: `data` plus the HELMET prompt templates."""
        raise NotImplementedError

    def _load_data(self) -> None:
        if self._dataset is not None:
            return

        loaded = self._load_dataset()

        self._dataset = loaded["data"]
        self._templates = {
            "prompt": loaded["prompt_template"],
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
        for idx, doc in enumerate(self._dataset):  # type: ignore
            instance = self.process_doc(cast(dict[str, Any], doc), index=idx)
            if instance is not None:
                self._instances_cache.append(instance)
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if self._templates is None:
            raise RuntimeError("Templates not loaded. Call _load_data() first.")

        question = self._templates["user"].format(**doc)
        prepend_text = self._templates["system"]

        answer = doc.get("answer")

        metadata: dict = {
            "id": index,
            "task_type": self.task_type,
            "context_size": self.context_size,
            "prepend_text": prepend_text,
            "tag": self.helmet_config["tag"],
        }
        if isinstance(answer, list):
            metadata["all_gold_answers"] = answer

        return Instance(
            question=question,
            gold_answer=answer,
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        return RequestType.COMPLETION

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = instance.question
        prepend_text = (instance.metadata or {}).get("prepend_text", "")
        if prepend_text:
            prompt = prompt + "\n" + prepend_text

        return LMRequest(
            request_type=self.request_type,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return output.text


class HelmetJsonKvTask(HelmetTask):
    """HELMET-plus json_kv task: extract a value for a given key from a long JSON blob."""

    def _load_dataset(self) -> dict[str, Any]:
        """Load the helmet-plus json_kv data for this task's length tier.

        HELMET-plus data is pre-generated at specific context lengths (e.g.
        256k, 2m tokens) and published as JSONL files on the Hub, so it's
        downloaded and cached via huggingface_hub rather than the standard
        dataset pipeline.
        """
        return load_json_kv_dataset(
            length_name=self.helmet_config["length_name"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )


def _parse_labeled_output(text: str, prefix: str) -> str | None:
    """Pull the answer out of a `label: N`-style completion.

    Mirrors HELMET's `parse_output` (utils.py): prefer the text following the
    prefix, otherwise fall back to the first line, then strip a repeated
    prefix that chat-style models often echo back.
    """
    patterns = [
        re.compile(f"(?:{re.escape(prefix)})(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return re.sub(
                f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE
            ).strip()
    return None


class HelmetIclTask(HelmetTask):
    """HELMET ICL task: label a text given many labelled demonstrations in context."""

    def _load_dataset(self) -> dict[str, Any]:
        return load_icl_dataset(
            icl_dataset=self.helmet_config["icl_dataset"],
            shots=self.helmet_config["shots"],
            max_samples=self.config.limit,
            seed=self.config.seed,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        return _parse_labeled_output(output.text, prefix="label:")


_TASK_CLASSES: dict[str, type[HelmetTask]] = {
    "json_kv": HelmetJsonKvTask,
    "icl": HelmetIclTask,
}

# Per-kind metric configuration. HELMET scores json_kv with substring exact
# match (RecallMetric's substring scorer is equivalent for a single gold
# answer) and ICL with exact match; see HELMET's scripts/collect_results.py.
_TASK_METRICS: dict[str, tuple] = {
    "json_kv": ((RecallMetric(),), "recall"),
    "icl": ((AccuracyMetric(name="exact_match"),), "exact_match"),
}


def _make_helmet_task_class(task_name: str, task_cfg: dict) -> type[HelmetTask]:
    """Create a task subclass for a HELMET-plus task variant.

    Subclasses carry only class-level attributes (metrics, sampling_params, limit);
    all runtime state is derived from config.name inside HelmetTask.__init__.
    """
    kind = task_cfg["kind"]
    base_cls = _TASK_CLASSES[kind]
    metrics, primary_metric = _TASK_METRICS[kind]

    stop_sequences = ("\n",) if task_cfg.get("stop_new_line") else None

    return type(
        f"Helmet_{task_name}",
        (base_cls,),
        {
            "__module__": __name__,
            "metrics": metrics,
            "primary_metric": primary_metric,
            "sampling_params": SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=task_cfg["max_gen_toks"],
                stop_sequences=stop_sequences,
            ),
            "limit": task_cfg["limit"],
        },
    )


# Dynamically register all HELMET-plus tasks
for _task_name, _task_config in HELMET_TASKS.items():
    _cls = _make_helmet_task_class(_task_name, _task_config)
    # Inject into module globals so pickle can find the class by name
    globals()[_cls.__name__] = _cls
    register(f"helmet_{_task_name}")(_cls)

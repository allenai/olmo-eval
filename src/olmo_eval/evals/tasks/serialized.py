"""Serialized benchmark tasks loaded from pre-formatted JSONL files.

These tasks bypass the standard Formatter pipeline because the serialized
data already contains both raw Instance fields (for scoring) and fully
formatted LMRequest fields (for inference).  The serialized JSONL is
produced by oe-eval-internal's serialize_benchmark.py.

Each JSONL line has the schema defined by oe_eval.serialize_benchmark.SerializedRecord:
    question, gold_answers, choices, metadata,
    request_type, prompt, messages, continuations,
    task_name, doc_id, native_id, label
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import BPBMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

_REQUEST_TYPE_MAP = {
    "completion": RequestType.COMPLETION,
    "chat": RequestType.CHAT,
    "loglikelihood": RequestType.LOGLIKELIHOOD,
}

# S3 base path for serialized benchmark data.
# Override via environment variable OLMO_EVAL_SERIALIZED_S3_BASE if needed.
_S3_BASE = "s3://ai2-llm/ianm/oe-eval-serialized/olmo3_base_easy_code_bpb"


@register("serialized")
class SerializedTask(Task):
    """A task whose instances and requests come from a pre-serialized JSONL file.

    The JSONL file is loaded via the unified DataLoader (supports S3, local,
    etc.).  Each line produces both an Instance (for scoring) and a cached
    LMRequest (returned by format_request without running any Formatter).
    """

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _load_instances_cached(self, split: str | None = None) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._load_serialized())
        yield from self._instances_cache

    def _load_serialized(self) -> Iterator[Instance]:
        loader = DataLoader()
        source = self.config.get_data_source()
        for record in loader.load(source):
            instance = _record_to_instance(record)
            yield instance

    def format_request(self, instance: Instance) -> LMRequest:
        record: dict[str, Any] = instance.metadata["_serialized"]
        rt = _REQUEST_TYPE_MAP[record["request_type"]]
        return LMRequest(
            request_type=rt,
            prompt=record.get("prompt") or "",
            messages=tuple(record["messages"]) if record.get("messages") else (),
            continuations=(tuple(record["continuations"]) if record.get("continuations") else None),
        )


def _record_to_instance(record: dict[str, Any]) -> Instance:
    """Convert a serialized JSONL record to an Instance.

    The full record is stashed in metadata["_serialized"] so that
    format_request can reconstruct the LMRequest without a Formatter.
    """
    gold_answers: list[str] = record.get("gold_answers") or []
    gold_answer = gold_answers[0] if gold_answers else None

    choices_raw = record.get("choices")
    choices = tuple(choices_raw) if choices_raw else None

    metadata: dict[str, Any] = dict(record.get("metadata") or {})
    metadata["gold_answers"] = gold_answers
    metadata["_serialized"] = record

    return Instance(
        question=record.get("question", ""),
        gold_answer=gold_answer,
        choices=choices,
        metadata=metadata,
    )


# =============================================================================
# Serialized task registrations for olmo3:base_easy:code_bpb
#
# Each task points its data_source at the S3 JSONL for that task and
# uses BPBMetric (matching oe-eval's primary_metric=bits_per_byte_corr).
# =============================================================================

_BPB_METRICS = (BPBMetric(),)

# codex_humaneval:3shot:bpb::none
register_variant(
    "serialized",
    "codex_humaneval_3shot_bpb",
    data_source=DataSource(path=f"{_S3_BASE}/codex_humaneval_3shot_bpb__none.jsonl"),
    metrics=_BPB_METRICS,
)

# mbpp:3shot:bpb::none
register_variant(
    "serialized",
    "mbpp_3shot_bpb",
    data_source=DataSource(path=f"{_S3_BASE}/mbpp_3shot_bpb__none.jsonl"),
    metrics=_BPB_METRICS,
    limit=500,
)

# mt_mbpp_v2fix:{language} — one variant per language
_MULTILINGUAL_MBPP_LANGUAGES = (
    "bash",
    "c",
    "cpp",
    "csharp",
    "go",
    "haskell",
    "java",
    "javascript",
    "matlab",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
)

for _lang in _MULTILINGUAL_MBPP_LANGUAGES:
    register_variant(
        "serialized",
        f"mt_mbpp_v2fix_{_lang}",
        data_source=DataSource(path=f"{_S3_BASE}/mt_mbpp_v2fix_{_lang}.jsonl"),
        metrics=_BPB_METRICS,
    )

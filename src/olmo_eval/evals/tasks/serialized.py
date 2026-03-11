"""Serialized benchmark tasks loaded from pre-formatted JSONL files.

These tasks bypass the standard Formatter pipeline because the serialized
data already contains both raw Instance fields (for scoring) and fully
formatted LMRequest fields (for inference).  The serialized JSONL is
produced by oe-eval-internal's serialize_benchmark.py.

Top-level JSONL fields (used for Instance / LMRequest building):
    doc_id, question, gold_answers, choices, metadata,
    request_type, prompt, messages, continuations

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import BPBMetricInstanceAvg
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

_REQUEST_TYPE_MAP = {
    "completion": RequestType.COMPLETION,
    "chat": RequestType.CHAT,
    "loglikelihood": RequestType.LOGLIKELIHOOD,
}

# S3 base path for serialized benchmark data.
_S3_BASE = "s3://ai2-llm/ianm/oe-eval-serialized/olmo3_base_easy_code_bpb"


@register("serialized")
class SerializedTask(Task):
    """A task whose instances and requests come from a pre-serialized JSONL file.

    The JSONL file is loaded via the unified DataLoader (supports S3, local,
    etc.).  Each line produces both an Instance (for scoring) and an
    LMRequest (returned by format_request without running any Formatter).
    """

    _records_by_doc_id: dict[int, dict[str, Any]] | None = None

    def _ensure_loaded(self) -> dict[int, dict[str, Any]]:
        if self._records_by_doc_id is None:
            loader = DataLoader()
            source = self.config.get_data_source()
            self._records_by_doc_id = {r["doc_id"]: r for r in loader.load(source)}
        return self._records_by_doc_id

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _load_instances_cached(self, split: str | None = None) -> Iterator[Instance]:
        if self._instances_cache is not None:
            yield from self._instances_cache
            return

        self._instances_cache = []
        for record in self._ensure_loaded().values():
            gold_answers: list[str] = record.get("gold_answers") or []
            choices_raw = record.get("choices")
            metadata: dict[str, Any] = dict(record.get("metadata") or {})
            metadata["_doc_id"] = record["doc_id"]
            instance = Instance(
                question=record.get("question", ""),
                gold_answer=gold_answers[0] if gold_answers else None,
                choices=tuple(choices_raw) if choices_raw else None,
                metadata=metadata,
            )
            self._instances_cache.append(instance)
            yield instance

    def format_request(self, instance: Instance) -> LMRequest:
        records = self._ensure_loaded()
        doc_id: int = instance.metadata["_doc_id"]
        record = records[doc_id]
        rt = _REQUEST_TYPE_MAP[record["request_type"]]
        return LMRequest(
            request_type=rt,
            prompt=record.get("prompt") or "",
            messages=tuple(record["messages"]) if record.get("messages") else (),
            continuations=(tuple(record["continuations"]) if record.get("continuations") else None),
        )


# =============================================================================
# Serialized task registrations for olmo3:base_easy:code_bpb
#
# Each task points its data_source at the S3 JSONL for that task and
# uses BPBMetricInstanceAvg (matching oe-eval's primary_metric=bits_per_byte_corr).
# =============================================================================

_BPB_METRICS = (BPBMetricInstanceAvg(),)

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

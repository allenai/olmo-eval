"""Arguments for DeepScholar-Bench evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# The seven metrics DeepScholar-Bench defines (paper Table 1), grouped by
# dimension: knowledge synthesis, retrieval quality, verifiability. The paper's
# headline score is the geometric mean over all seven (Table 2 "Geo. Mean"; no
# system exceeds ~31%). These strings are the upstream EvaluationFunction values,
# which are also the per-metric output subdir and aggregated_results.csv column.
PRIMARY_METRICS = (
    "organization",
    "nugget_coverage",
    "coverage_relevance_rate",
    "document_importance",
    "reference_coverage",
    "cite_p",
    "claim_coverage",
)


def _parse_optional(data: dict[str, Any], key: str, type_fn: type) -> Any:
    value = data.get(key)
    return type_fn(value) if value is not None else None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


@dataclass
class DeepScholarArgs:
    """Arguments for the deepscholar_bench evaluation.

    The model under test drives the generation phase via the LOTUS ``lm`` config
    block; the judge model (``judge_model``) scores the generated related-work
    sections in the eval phase.
    """

    # Generation phase (model under test)
    limit: int | None = None  # -> generation --end-idx (smoke runs)
    start_idx: int = 0  # -> generation --start-idx
    search_mode: str | None = None  # "agentic" | "recursive"; None keeps the YAML default
    # Retrieval backend. "arxiv" is upstream's recursive default (keyless, but the
    # export.arxiv.org API rate-limits/hangs at volume). "s2" routes recursive search
    # to the Semantic Scholar API via a runtime shim (keyed, request timeouts, no
    # arXiv hangs; needs S2_API_KEY). "tavily" skips the hardwired arXiv corpus and
    # uses only the TAVILY web corpus (needs TAVILY_API_KEY).
    search_backend: str = "arxiv"
    # Web search corpus for retrieval (arxiv backend only). ARXIV is keyless;
    # TAVILY/GOOGLE/GOOGLE_SCHOLAR/BING need their own API keys.
    web_corpuses: list[str] = field(default_factory=lambda: ["ARXIV"])
    # Recursive-search intensity. Total search requests scale with
    # steps * queries_per_step * papers; lowering these reduces arXiv 429s. None
    # keeps the upstream config value.
    search_steps: int | None = None  # -> num_search_steps
    search_queries_per_step: int | None = None  # -> num_search_queries_per_step_per_corpus
    temperature: float | None = None
    max_tokens: int = 10000
    # Token budget for upstream's stage LMs (filter/search/taxonomize/generation),
    # which otherwise default to 512 and truncate structured outputs. None uses the
    # eval's DEFAULT_STAGE_MAX_TOKENS (kept below max_model_len, since LOTUS sends it
    # as max_completion_tokens and the server rejects prompt + budget > context).
    stage_max_tokens: int | None = None
    # Concurrent vLLM requests per LOTUS sem-op (batch_completion max_workers).
    # Upstream defaults to 64; that many concurrent connections from inside the
    # nested-podman sandbox is a likely contributor to the swe-rex runtime wedging
    # under resource pressure. Lower it (e.g. 4-8) to trade generation speed for
    # container stability. None keeps the upstream default. Propagates to stage LMs.
    max_batch_size: int | None = None
    # litellm provider prefix for a local OpenAI-compatible (vLLM) server.
    # "openai" routes via litellm's OpenAI handler against api_base; an alternative
    # is "hosted_vllm". Ignored for external API models.
    local_model_prefix: str = "openai"
    # Per-call timeout (s) for LOTUS LM calls. Set both as the litellm config `timeout`
    # (best-effort; litellm does not reliably enforce it on the vLLM path) and as the
    # shim's hard wall-clock guard, which runs each LM call in a worker thread and
    # abandons it after this many seconds. A stalled/runaway vLLM request then fails the
    # one query (upstream catches it and moves on) instead of hanging the run until the
    # sandbox watchdog aborts. Kept under the 300s health-poll interval so it fires
    # first. Worst-case legitimate call is a stage LM at stage_max_tokens (~4096) tokens,
    # well under 240s, so this does not cut genuine work.
    lm_timeout: int = 240

    # In-sandbox chunking. A single generation process over all 63 queries wedges the
    # sandbox container ~40 min in (a nested-podman resource stall, not a vLLM crash),
    # losing every completed query. Instead we run generation as a sequence of short
    # commands over disjoint index ranges of `chunk_size` queries, each well inside the
    # proven-reliable ~20-min window; completed query folders accumulate on disk and
    # eval runs once over the union. A stalled chunk is skipped (its head query dropped)
    # and the loop advances rather than killing the whole run. Set chunk_size=0 (or None)
    # to disable chunking and run one command (the old behavior). Runs that fit in a
    # single chunk (limit <= chunk_size) take the single-command path unchanged.
    chunk_size: int | None = 10
    # Per-chunk timeout (s). Kept under the ~40-min wedge threshold so a slow chunk is
    # cut and its remaining queries retried in a later chunk, rather than drifting into
    # the wedge. The sandbox's own 3x300s poll-abort catches true stalls sooner.
    chunk_timeout: int = 1800
    # Extra chunk attempts beyond the nominal chunk count, absorbing retries of cut or
    # stalled chunks. The loop also stops early if the container goes unresponsive
    # between chunks or the overall generation budget is exhausted.
    chunk_retries: int = 3

    # Eval phase (judge). Default to all seven metrics (the geomean inputs and how
    # both the paper's Table 2 and the leaderboard report results). Pass a
    # comma-separated subset to run fewer; the geomean is reported only when every
    # metric in PRIMARY_METRICS is present.
    judge_model: str = "gpt-4o"
    evals: list[str] = field(default_factory=lambda: list(PRIMARY_METRICS))

    # Strict by default: a partial generation (some queries failed) would score
    # only the succeeded subset, which misrepresents the benchmark. Opt in to
    # score whatever generated.
    allow_partial_generation: bool = False

    # Escape hatches for validation iterations
    extra_gen_args: list[str] = field(default_factory=list)
    extra_eval_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepScholarArgs:
        evals = data.get("evals")
        if isinstance(evals, str):
            evals = [e.strip() for e in evals.split(",") if e.strip()]

        def _as_list(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [a.strip() for a in value.split(",") if a.strip()]
            return list(value)

        # Absent -> default 10 (chunking on); explicit None/""/"none"/<=0 -> disabled.
        raw_chunk = data.get("chunk_size", 10)
        if isinstance(raw_chunk, str):
            raw_chunk = raw_chunk.strip().lower()
            raw_chunk = None if raw_chunk in ("", "none", "null", "off") else int(raw_chunk)
        chunk_size = int(raw_chunk) if raw_chunk is not None else None
        if chunk_size is not None and chunk_size <= 0:
            chunk_size = None

        return cls(
            limit=_parse_optional(data, "limit", int),
            start_idx=int(data.get("start_idx", 0)),
            search_mode=data.get("search_mode"),
            search_backend=data.get("search_backend", "arxiv"),
            web_corpuses=_as_list(data.get("web_corpuses")) or ["ARXIV"],
            search_steps=_parse_optional(data, "search_steps", int),
            search_queries_per_step=_parse_optional(data, "search_queries_per_step", int),
            temperature=_parse_optional(data, "temperature", float),
            max_tokens=int(data.get("max_tokens", 10000)),
            stage_max_tokens=_parse_optional(data, "stage_max_tokens", int),
            local_model_prefix=data.get("local_model_prefix", "openai"),
            lm_timeout=int(data.get("lm_timeout", 240)),
            max_batch_size=_parse_optional(data, "max_batch_size", int),
            chunk_size=chunk_size,
            chunk_timeout=int(data.get("chunk_timeout", 1800)),
            chunk_retries=int(data.get("chunk_retries", 3)),
            judge_model=data.get("judge_model", "gpt-4o"),
            evals=evals or list(PRIMARY_METRICS),
            allow_partial_generation=_parse_bool(data.get("allow_partial_generation")),
            extra_gen_args=_as_list(data.get("extra_gen_args")),
            extra_eval_args=_as_list(data.get("extra_eval_args")),
        )

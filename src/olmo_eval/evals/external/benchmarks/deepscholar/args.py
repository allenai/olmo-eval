"""Arguments for DeepScholar-Bench evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Metric names emitted by the upstream eval (`--evals all`). The geometric mean
# of these is DeepScholar-Bench's headline score.
PRIMARY_METRICS = ("organization", "nugget_coverage", "reference_coverage", "cite_p")


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
    # Web search corpus for retrieval. ARXIV is keyless (the dataset is arXiv CS
    # papers); TAVILY/GOOGLE/GOOGLE_SCHOLAR/BING need their own API keys.
    web_corpuses: list[str] = field(default_factory=lambda: ["ARXIV"])
    temperature: float | None = None
    max_tokens: int = 10000
    # litellm provider prefix for a local OpenAI-compatible (vLLM) server.
    # "openai" routes via litellm's OpenAI handler against api_base; an alternative
    # is "hosted_vllm". Ignored for external API models.
    local_model_prefix: str = "openai"

    # Eval phase (judge). Default to the four headline metrics (the geomean
    # inputs); `-a evals=all` opts into the full upstream set (adds
    # document_importance, claim_coverage, coverage_relevance_rate).
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

        return cls(
            limit=_parse_optional(data, "limit", int),
            start_idx=int(data.get("start_idx", 0)),
            search_mode=data.get("search_mode"),
            web_corpuses=_as_list(data.get("web_corpuses")) or ["ARXIV"],
            temperature=_parse_optional(data, "temperature", float),
            max_tokens=int(data.get("max_tokens", 10000)),
            local_model_prefix=data.get("local_model_prefix", "openai"),
            judge_model=data.get("judge_model", "gpt-4o"),
            evals=evals or list(PRIMARY_METRICS),
            allow_partial_generation=_parse_bool(data.get("allow_partial_generation")),
            extra_gen_args=_as_list(data.get("extra_gen_args")),
            extra_eval_args=_as_list(data.get("extra_eval_args")),
        )

"""Arguments for the DeepScholar-Bench external eval.

EXPERIMENTAL / UNVALIDATED: see eval.py and plans/003_deepscholar_bench.md. The
flag names mirror the upstream repo (guestrin-lab/deepscholar-bench) but have not
been validated against a real run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _opt(data: dict[str, Any], key: str, type_fn: type) -> Any:
    value = data.get(key)
    return type_fn(value) if value is not None else None


@dataclass
class DeepScholarArgs:
    """Arguments for deepscholar_bench evaluation.

    Defaults track upstream `eval/argument_parser.py` and `scripts/run_all.sh`.
    """

    # Which systems / metrics to run. `evals="all"` runs every metric.
    modes: str = "deepscholar_base"
    evals: str = "all"

    # Dataset + reference inputs (shipped snapshot in the repo; pin for reproducibility).
    dataset_path: str = "dataset/papers_with_related_works.csv"
    config_yaml: str = "configs/deepscholar_base.yaml"

    # Judge model used by the eval metrics (NOT the model under test).
    eval_model: str = "gpt-4o"

    # Model-under-test sampling (passed into the LOTUS generation config).
    max_tokens: int | None = None
    temperature: float | None = None
    max_model_len: int | None = None

    # Scope controls.
    num_queries: int | None = None  # limit number of queries for a smoke run
    file_id: str | None = None  # run a single query by id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepScholarArgs:
        return cls(
            modes=data.get("modes", "deepscholar_base"),
            evals=data.get("evals", "all"),
            dataset_path=data.get("dataset_path", "dataset/papers_with_related_works.csv"),
            config_yaml=data.get("config_yaml", "configs/deepscholar_base.yaml"),
            eval_model=data.get("eval_model", "gpt-4o"),
            max_tokens=_opt(data, "max_tokens", int),
            temperature=_opt(data, "temperature", float),
            max_model_len=_opt(data, "max_model_len", int),
            num_queries=_opt(data, "num_queries", int),
            file_id=data.get("file_id"),
        )

"""The judge-graded multimodal safety task base and its metric families.

The safety benchmarks are :class:`~olmo_eval.evals.tasks.common.judge.JudgedTask`\\ s
over images: each output is graded by the benchmark's own judge prompts through the
shared judge, under the run's judge limiter, and the judge model, token budget and
temperature come from the task configuration (``-o judge_model=...``,
``OLMO_EVAL_JUDGE``). A judge call that fails, or a reply that never parses, leaves
the output unscored with the error recorded, so a judge outage counts against the
run's failure budget instead of shifting a rate. The raw verdicts behind each score
are kept in ``judge_result`` and saved with the predictions.

Rates are means of a per-instance score over the instances that have one, optionally
restricted to a metadata slice; attack-success and refusal rates are lower-is-better.
The staged data is pinned: each benchmark lists its annotation files' SHA-256, and a
changed file or a missing image fails setup rather than silently changing which
examples are evaluated.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Response
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.judge import JudgedTask, JudgeScorer
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def image_data_url(image_path: str) -> str:
    """The image file as a base64 data URL, typed by its extension (JPEG otherwise)."""
    mime = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def judge_content(text: str, image_path: str, *, image_first: bool = False) -> list[dict]:
    """A judge prompt with the benchmark image attached, in the official order."""
    parts: list[dict] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
    ]
    return parts[::-1] if image_first else parts


def verify_manifest(root: Path, manifest: dict[str, str]) -> None:
    """Fail unless every file in ``manifest`` exists under ``root`` with its SHA-256."""
    problems = []
    for relative, expected in sorted(manifest.items()):
        path = root / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{relative}: sha256 {actual[:12]} != pinned {expected[:12]}")
    if problems:
        raise ValueError(
            f"Staged data under {root} does not match the pinned release: " + "; ".join(problems)
        )


def require_images(paths: Sequence[str], dataset: str) -> None:
    """Fail when any image an instance needs is missing or empty."""
    missing = [p for p in paths if not Path(p).is_file() or Path(p).stat().st_size == 0]
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} of {len(paths)} images are missing or empty, "
            f"e.g. {missing[0]}"
        )


def reject_judge_settings(config: TaskConfig) -> None:
    """Fail when a task without a judge is given judge settings it would ignore."""
    fields = {
        "judge_model": config.judge_model,
        "judge_max_tokens": config.judge_max_tokens,
        "judge_temperature": config.judge_temperature,
        "judge_reasoning_effort": config.judge_reasoning_effort,
    }
    set_fields = sorted(name for name, value in fields.items() if value is not None)
    if set_fields:
        raise ValueError(f"{config.name} has no judge; it does not take {', '.join(set_fields)}")


def _in_slice(response: Response, where: tuple[tuple[str, Any], ...]) -> bool:
    meta = response.instance.metadata
    return all(meta.get(name) == value for name, value in where)


def _score(response: Response, key: str) -> float | None:
    value = response.scores.get(key)
    return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True)
class SafetyRateMetric(Metric):
    """Mean of one per-instance score over the scored instances of a slice.

    ``where`` restricts the mean to instances whose metadata matches every
    ``field: value`` pair (a category, an image type, ...). An unscored instance
    (judging failed) is left out rather than counted as zero.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    score_key: str = "score"
    where: tuple[tuple[str, Any], ...] = field(default=())
    lower_is_better: bool = False

    def compute_instance(self, response: Response) -> float | None:
        if not _in_slice(response, self.where):
            return None
        return _score(response, self.score_key)

    def compute(self, responses: Sequence[Response]) -> float:
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        return sum(values) / len(values) if values else 0.0

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return not self.lower_is_better

    def display_format(self) -> str:
        return "percentage"


@dataclass(frozen=True)
class MacroRateMetric(Metric):
    """Unweighted mean, over the values of metadata field ``group``, of each group's rate.

    ``where`` restricts the instances first, as in :class:`SafetyRateMetric`.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    score_key: str = "score"
    group: str = "category"
    where: tuple[tuple[str, Any], ...] = field(default=())
    lower_is_better: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        by_group: dict[Any, list[float]] = {}
        for response in responses:
            value = _score(response, self.score_key)
            if value is None or not _in_slice(response, self.where):
                continue
            by_group.setdefault(response.instance.metadata.get(self.group), []).append(value)
        if not by_group:
            return 0.0
        rates = [sum(values) / len(values) for values in by_group.values()]
        return sum(rates) / len(rates)

    def compute_instance(self, response: Response) -> float | None:
        """A mean of group rates has no per-instance equivalent."""
        return None

    def supports_scorer_fallback(self) -> bool:
        return False

    def higher_is_better(self) -> bool:
        return not self.lower_is_better

    def display_format(self) -> str:
        return "percentage"


class SafetyJudgeTask(ImageQATask, JudgedTask):
    """Base class for the judge-graded single-image safety benchmarks.

    Subclasses set the judge configuration and implement ``_score_output``. Prompts
    hand the judge the raw response text, as the official scripts do.
    """

    #: The judges call OpenAI (or OpenRouter).
    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)

    #: Annotation files (relative to the dataset dir) and their pinned SHA-256.
    data_manifest: ClassVar[dict[str, str]] = {}

    async def score_responses(self, responses: Sequence[Response], context: Any = None):
        responses = await super().score_responses(responses, context)
        # An instance the model gave no output for has nothing to judge: leave it
        # unscored (the runner counts it) rather than at a fixed score.
        for response in responses:
            if not response.outputs:
                for key in (self.judge_score_key, *self.judge_extra_score_keys):
                    response.scores.pop(key, None)
        return responses


__all__ = [
    "JudgeScorer",
    "MacroRateMetric",
    "SafetyJudgeTask",
    "SafetyRateMetric",
    "image_data_url",
    "judge_content",
    "reject_judge_settings",
    "require_images",
    "verify_manifest",
]

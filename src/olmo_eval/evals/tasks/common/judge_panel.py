"""Shared base for tasks graded by an LLM judge or a panel of judges.

A panel task's judged metrics are graded by its default judge, or by every judge of a
panel when ``judge_models`` is set. Each panel judge grades every output and reports
every judged metric under its own scorer name, ``<default judge>[<model>]``, so judges
can be compared on the same outputs. Judges record their verdicts on the instance, keyed
by judge name, so a panel task samples one response per prompt.

A panel entry is one of

- ``<model>`` or ``<model>:<reasoning effort>``: a judge called through the OpenAI client.
  ``gemini-*`` models go to Google's OpenAI-compatible endpoint with ``GEMINI_API_KEY``,
  others to OpenAI with ``OPENAI_API_KEY``. ``-o judge_reasoning_effort`` sets the effort
  of entries that name none.
- ``<provider>=<model>``: a judge served by the run's auxiliary provider ``<provider>``,
  which must serve ``<model>``.

``-o judge_max_tokens`` sets every judge's token limit. Without a panel,
``-o judge_model``, ``-o judge_reasoning_effort`` and ``-o judge_max_tokens`` change the
default judge's settings. A task's required secrets follow its judges; a Beaker launch
with a custom panel mounts only the task's declared secrets, so add
``-o 'required_secrets=[...]'`` when the panel needs other keys.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.metrics import Metric
from olmo_eval.common.openai_credentials import judge_secret
from olmo_eval.common.scorers.llm_judge import (
    ConfiguredJudgeFn,
    LLMJudgeScorer,
    build_openai_judge_fn,
    judge_with_retries,
    parse_judge_spec,
)
from olmo_eval.common.types import Instance, LMOutput, Response

from .base import Task, TaskConfig
from .registry import register_variant

logger = logging.getLogger(__name__)

#: The ``judge_panel`` variant's judges: OpenAI and Gemini judges, and two judges served by
#: auxiliary vLLM providers.
DEFAULT_JUDGE_PANEL = (
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
    "gpt-5.5-2026-04-23:low",
    "gemini-3.6-flash",
    "gemini-4-argon:low",
    "qwen_judge=Qwen/Qwen3-30B-A3B-Instruct-2507",
    "kimi_judge=moonshotai/Kimi-Linear-48B-A3B-Instruct",
)
#: The panel's token limit, which leaves reasoning judges room to finish; their reasoning
#: counts against it.
PANEL_JUDGE_MAX_TOKENS = 4096

# =============================================================================
# Judge Verdicts
# =============================================================================


def verdict_text(reply: str) -> str:
    """Return the part of a judge reply that holds the verdict.

    A thinking judge's reasoning block is dropped, and a block that never closed leaves no
    verdict. Other replies are kept exactly.
    """
    if "</think>" in reply:
        return reply.rpartition("</think>")[2].strip()
    return "" if "<think>" in reply else reply


def record_judgement(
    instance: Instance,
    judge_name: str,
    judge_result: dict[str, Any] | None,
    *,
    raw_response: Any,
) -> None:
    """Store one judge's verdicts and raw replies on an instance, beside other judges'.

    ``judge_result`` and ``judge_raw_response`` map judge names to their entries; a result
    of None marks an output the judge could not grade.
    """
    instance.metadata.setdefault("judge_result", {})[judge_name] = judge_result
    instance.metadata.setdefault("judge_raw_response", {})[judge_name] = raw_response


def judge_result(response: Response, judge_name: str) -> dict[str, Any] | None:
    """Return one judge's recorded verdicts of a response, or None if it graded none."""
    results = response.instance.metadata.get("judge_result")
    result = results.get(judge_name) if isinstance(results, dict) else None
    return result if isinstance(result, dict) else None


# =============================================================================
# Judges
# =============================================================================


@dataclass(frozen=True)
class PanelJudgeScorer(LLMJudgeScorer):
    """LLM judge that can serve on a panel, through an API or an auxiliary provider.

    Subclasses set ``default_model`` and ``default_max_tokens``, the API judge built when
    none is given, and optionally ``judge_response_format``. A reply with no verdict is
    asked for again, up to ``attempts`` calls.
    """

    #: Model that the auxiliary provider of a provider judge must serve.
    provider_model: str | None = None
    #: Token limit of a provider judge's replies; None uses ``default_max_tokens``. An API
    #: judge's limit is set in its ``judge_fn``.
    provider_max_tokens: int | None = None

    default_model: ClassVar[str]
    default_max_tokens: ClassVar[int]
    attempts: ClassVar[int] = 3
    #: ``response_format`` sent to API judges, such as ``{"type": "json_object"}``.
    judge_response_format: ClassVar[dict[str, Any] | None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.provider_name is None and self.judge_fn is None:
            object.__setattr__(self, "judge_fn", self.openai_judge_fn())

    @classmethod
    def openai_judge_fn(cls, **settings: Any) -> ConfiguredJudgeFn:
        """Build an API judge, with ``settings`` replacing the default model and token limit."""
        options: dict[str, Any] = {
            "model": cls.default_model,
            "max_tokens": cls.default_max_tokens,
            **settings,
        }
        if cls.judge_response_format is not None:
            options["response_format"] = cls.judge_response_format
        return build_openai_judge_fn(scorer_name=cls.__name__, **options)

    @property
    def required_secret(self) -> str | None:
        """The secret this judge needs, or None for a provider judge."""
        if self.provider_name is not None:
            return None
        judge_config = getattr(self.judge_fn, "judge_config", {})
        return judge_secret(judge_config.get("model", self.default_model))

    def final_answer(self, output: LMOutput) -> str | None:
        """Return the model's final answer, or None when it gave none.

        Uses the extracted answer when there is one, so an empty extraction (e.g.
        an unfinished reasoning trace) counts as no answer, as does blank text.
        """
        answer = output.text if output.extracted_answer is None else output.extracted_answer
        return (answer or "").strip() or None

    def record(
        self, instance: Instance, judge_result: dict[str, Any] | None, *, raw_response: Any
    ) -> None:
        """Store this judge's verdicts and raw replies on an instance."""
        record_judgement(instance, self.name, judge_result, raw_response=raw_response)

    def _validate_provider(self, context: ScoringContext) -> None:
        """Check that a provider judge's provider is available and serves its model."""
        super()._validate_provider(context)
        if self.provider_name is None or self.provider_model is None:
            return
        served = self._require_provider(context).model_name
        if served != self.provider_model:
            raise ValueError(
                f"{self.name} needs auxiliary provider {self.provider_name!r} to serve "
                f"{self.provider_model!r}, but it serves {served!r}"
            )

    async def _judge[V](
        self, prompt: str, parse: Callable[[str], V | None], context: ScoringContext
    ) -> tuple[V | None, str]:
        """Ask the judge until a reply parses; return the verdict, or None, and the last reply.

        ``parse`` sees the reply without a thinking judge's reasoning.
        """

        async def ask(prompt: str, *, system_prompt: str | None = None) -> str:
            if self.provider_name is not None:
                return await self._score_with_provider(
                    prompt, context, max_tokens=self.provider_max_tokens or self.default_max_tokens
                )
            return await self._score_with_judge_fn(prompt)

        # The runner already bounds concurrent scoring, so each call gets its own slot
        return await judge_with_retries(
            ask,
            prompt,
            lambda reply: parse(verdict_text(reply)),
            limiter=asyncio.Semaphore(1),
            attempts=self.attempts,
            context=self.name,
        )


def panel_entries(judge_models: Any) -> tuple[str, ...]:
    """Return the configured panel's entries, given one entry or a list of them."""
    entries = (judge_models,) if isinstance(judge_models, str) else tuple(judge_models)
    if not entries:
        raise ValueError("judge_models must name at least one judge")
    if not all(isinstance(entry, str) and entry for entry in entries):
        raise ValueError(f"judge_models entries must be non-empty strings, got {entries!r}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"judge_models names a judge more than once: {entries!r}")
    return entries


def panel_judge(
    base: PanelJudgeScorer,
    entry: str,
    *,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> PanelJudgeScorer:
    """Build the judge a panel entry names, as a copy of ``base``.

    The judge's name adds its model, and any reasoning effort after a ``+``, to
    ``base``'s. ``max_tokens`` defaults to ``base``'s default token limit.
    """
    max_tokens = max_tokens or base.default_max_tokens
    provider, separator, model = entry.partition("=")
    if separator:
        if not provider or not model:
            raise ValueError(f"judge entry {entry!r} must take the form '<provider>=<model>'")
        return replace(
            base,
            name=f"{base.name}[{model}]",
            provider_name=provider,
            provider_model=model,
            provider_max_tokens=max_tokens,
        )

    model, effort = parse_judge_spec(entry)
    effort = effort or reasoning_effort
    label = model if effort is None else f"{model}+{effort}"
    return replace(
        base,
        name=f"{base.name}[{label}]",
        judge_fn=type(base).openai_judge_fn(
            model=model, max_tokens=max_tokens, reasoning_effort=effort
        ),
    )


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True, slots=True)
class JudgeVerdictMetric(Metric):
    """Mean per-instance value over the responses its judge graded in a subset.

    Names take the form ``<subset>__<category>__<metric>``, as for the Omniscience metrics:
    ``any__any`` covers every response, while e.g. ``type__medicine`` covers only responses
    whose ``type`` metadata is ``medicine``. Values come from the verdicts of the metric's
    scorer, so each judge of a panel has its own copy of the metric. Responses the judge
    could not grade are excluded.
    """

    subset: str = field(init=False, repr=False, compare=False)
    category: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        parts = self.name.split("__")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"{type(self).__name__} name must take the form "
                f"'<subset>__<category>__<metric>', got {self.name!r}"
            )
        object.__setattr__(self, "subset", parts[0])
        object.__setattr__(self, "category", parts[1])

    @abstractmethod
    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        """Value of a judged response, or None if the metric leaves it out."""
        ...

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses.

        Reports 0.0, with a warning, when the judge graded no response in the subset.
        """
        values = [v for r in responses if (v := self.compute_instance(r)) is not None]
        if values:
            return sum(values) / len(values)
        judge_name = self.scorer().name
        if not any(
            self._in_subset(r) and judge_result(r, judge_name) is not None for r in responses
        ):
            logger.warning(
                "%s has no responses judged by %s; its value of 0.0 is not a measured score.",
                self.name,
                judge_name,
            )
        return 0.0

    def compute_instance(self, response: Response) -> float | None:
        """Value of the response, or None if it is outside the subset or was not judged."""
        if not self._in_subset(response):
            return None
        result = judge_result(response, self.scorer().name)
        return None if result is None else self.judged_value(result)

    def _in_subset(self, response: Response) -> bool:
        """Whether the response belongs to the subset this metric covers."""
        return self.subset == "any" or response.instance.metadata.get(self.subset) == self.category

    def supports_scorer_fallback(self) -> bool:
        return False

    def display_format(self) -> str:
        return "percentage"


# =============================================================================
# Tasks
# =============================================================================


class JudgePanelTask(Task):
    """Base for tasks whose judged metrics a judge or a judge panel grades.

    A metric is judged when its scorer is of the default judge's type. The task rebuilds its
    judges whenever its config is assigned, because the runner applies ``-o`` overrides by
    assigning a new config to a built task.
    """

    #: The default judge, which grades every judged metric unless a panel is configured.
    judge: ClassVar[PanelJudgeScorer]

    @property
    def config(self) -> TaskConfig:
        """The task's configuration, with its metrics graded by the configured judges."""
        return self._config

    @config.setter
    def config(self, config: TaskConfig) -> None:
        """Set the configuration, regrading its judged metrics with its judge or panel."""
        self._config = self._graded_by(config, self._configured_judges(config))
        self._scorers_cache = None
        self._has_async_cache = None

    def _configured_judges(self, config: TaskConfig) -> list[PanelJudgeScorer]:
        """Return the judges the config asks for: a panel, or the default judge.

        Without a panel, the default judge takes any of ``judge_model``,
        ``judge_reasoning_effort`` and ``judge_max_tokens`` that are set.
        """
        if config.judge_models is not None:
            if config.judge_model is not None:
                raise ValueError("Set judge_model or judge_models, not both")
            return [
                panel_judge(
                    self.judge,
                    entry,
                    max_tokens=config.judge_max_tokens,
                    reasoning_effort=config.judge_reasoning_effort,
                )
                for entry in panel_entries(config.judge_models)
            ]

        settings: dict[str, Any] = {
            name: value
            for name, value in (
                ("model", config.judge_model),
                ("reasoning_effort", config.judge_reasoning_effort),
                ("max_tokens", config.judge_max_tokens),
            )
            if value is not None
        }
        if not settings:
            return [self.judge]
        return [replace(self.judge, judge_fn=type(self.judge).openai_judge_fn(**settings))]

    def _graded_by(self, config: TaskConfig, judges: list[PanelJudgeScorer]) -> TaskConfig:
        """Return the config with one copy of each judged metric per judge.

        The judged metrics are taken once by name, so a config that was already graded is
        regraded rather than copied again. The first judge grades the primary metric, and
        the required secrets are the judges'.
        """

        def is_judged(metric: Metric) -> bool:
            return isinstance(metric.scorer(), type(self.judge))

        judged: dict[str, Metric] = {}
        for metric in config.metrics:
            if is_judged(metric):
                judged.setdefault(metric.name, metric)
        metrics = (
            *(metric for metric in config.metrics if not is_judged(metric)),
            *(replace(metric, scorer=judge) for judge in judges for metric in judged.values()),
        )
        primary = config.primary_metric
        if isinstance(primary, Metric) and is_judged(primary):
            primary = replace(primary, scorer=judges[0])
        secrets = {secret for judge in judges if (secret := judge.required_secret)}
        return replace(
            config,
            metrics=metrics,
            primary_metric=primary,
            required_secrets=tuple(sorted(secrets)),
        )

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset, rejecting configs that sample several responses.

        Judges record their verdicts on the instance, which one response per prompt shares.
        """
        params = self.config.sampling_params
        if params is not None and params.num_samples > 1:
            raise ValueError(
                f"{type(self).__name__} grades a single response per prompt; num_samples must be 1."
            )
        yield from self._load_instances_cached()


def register_judge_panel_variant(
    task_name: str, judge_max_tokens: int = PANEL_JUDGE_MAX_TOKENS
) -> None:
    """Register the ``judge_panel`` variant of a panel task: every judge of the default panel.

    The variant declares the panel's API keys, so a Beaker launch mounts them. A task whose
    judge needs more than ``PANEL_JUDGE_MAX_TOKENS`` passes its own ``judge_max_tokens``.
    """
    register_variant(
        task_name,
        "judge_panel",
        judge_models=DEFAULT_JUDGE_PANEL,
        judge_max_tokens=judge_max_tokens,
        required_secrets=("GEMINI_API_KEY", "OPENAI_API_KEY"),
    )

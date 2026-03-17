"""Smoke tests for basic model sanity checks.

Simple, single-instance tests to verify models respond correctly to basic prompts.

Usage:
    # Generic identity check (no scoring, just captures response)
    olmo-eval run -m any-model -t smoke_identity

    # Model-specific identity checks (scores against expected substring)
    olmo-eval run -m olmo-model -t smoke_identity_olmo
    olmo-eval run -m llama-model -t smoke_identity_llama
    olmo-eval run -m gpt-model -t smoke_identity_gpt

    # Basic hello test
    olmo-eval run -m any-model -t smoke_hello

    # Tool calling test (verifies model can make tool calls)
    olmo-eval run -m any-model -t smoke_toolcall

    # Reasoning test (verifies model returns reasoning content)
    olmo-eval run -m reasoning-model -t smoke_reasoning
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from olmo_eval.common.formatters import CompletionFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import Scorer, ToolCallScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    ToolSchema,
)
from olmo_eval.evals.tasks.common import Task, register

# =============================================================================
# Substring Scorer
# =============================================================================


@dataclass(frozen=True, slots=True)
class SubstringScorer(Scorer):
    """Score 1.0 if gold answer substring appears in the output, else 0.0.

    This is useful for identity checks where we want to verify the model
    mentions a specific name/identifier in its response.
    """

    name: ClassVar[str] = "substring_match"
    case_sensitive: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        if not instance.gold_answer:
            # No expected substring configured - skip scoring
            return 1.0

        text = output.text or ""
        expected = instance.gold_answer

        if not self.case_sensitive:
            text = text.lower()
            expected = expected.lower()

        return 1.0 if expected in text else 0.0


@dataclass(frozen=True, slots=True)
class NonEmptyResponseScorer(Scorer):
    """Score 1.0 if model produced a non-empty response, else 0.0."""

    name: ClassVar[str] = "non_empty_response"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0 if output.text and output.text.strip() else 0.0


@dataclass(frozen=True, slots=True)
class ReasoningScorer(Scorer):
    """Score 1.0 if model produced reasoning content, else 0.0.

    This verifies that reasoning models correctly return their chain-of-thought
    in the reasoning field of the response.
    """

    name: ClassVar[str] = "reasoning_present"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0 if output.has_reasoning else 0.0


# =============================================================================
# Smoke Test Base Classes
# =============================================================================


class IdentitySmokeBase(Task):
    """Base class for identity smoke tests.

    Subclasses set `expected_substring` to define what the model should say.
    Uses CompletionFormatter (not ChatFormatter which produces CHAT requests
    that require a backend for agentic loops).
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    formatter = CompletionFormatter(template="User: {question}\nAssistant:")

    # Override in subclasses to set expected model identity
    expected_substring: str = ""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(
            question="Who are you?",
            gold_answer=self.expected_substring,
            metadata={"id": "identity", "check_type": "substring"},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format instance for the language model."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        # Fallback formatting
        return LMRequest(request_type=self.request_type, prompt=instance.question)


# =============================================================================
# Registered Smoke Test Tasks
# =============================================================================


@register("smoke_identity")
class IdentitySmoke(IdentitySmokeBase):
    """Smoke test: does the model correctly identify itself?

    This is the generic version with no expected substring - it just checks
    that the model produces a non-empty response. Use model-specific tasks
    for substring matching:
        - smoke_identity_olmo
        - smoke_identity_llama
        - smoke_identity_gpt
        - etc.
    """

    expected_substring = ""
    metrics = (AccuracyMetric(scorer=NonEmptyResponseScorer),)


@register("smoke_identity_olmo")
class IdentitySmokeOlmo(IdentitySmokeBase):
    """Identity smoke test expecting 'Olmo' in response."""

    expected_substring = "Olmo"
    metrics = (AccuracyMetric(scorer=SubstringScorer),)


@register("smoke_identity_llama")
class IdentitySmokeLlama(IdentitySmokeBase):
    """Identity smoke test expecting 'Llama' in response."""

    expected_substring = "Llama"
    metrics = (AccuracyMetric(scorer=SubstringScorer),)


@register("smoke_identity_gpt")
class IdentitySmokeGpt(IdentitySmokeBase):
    """Identity smoke test expecting 'GPT' in response."""

    expected_substring = "GPT"
    metrics = (AccuracyMetric(scorer=SubstringScorer),)


@register("smoke_hello")
class HelloSmoke(Task):
    """Smoke test: can the model respond to a greeting?

    A basic sanity check that the model can produce a non-empty response.
    Scores 1.0 if response is non-empty, 0.0 otherwise.
    """

    sampling_params = SamplingParams(temperature=0.0)
    formatter = CompletionFormatter(template="User: {question}\nAssistant:")
    metrics = (AccuracyMetric(scorer=NonEmptyResponseScorer),)

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(
            question="Hello!",
            gold_answer="",
            metadata={"id": "hello"},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format instance for the language model."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        # Fallback formatting
        return LMRequest(request_type=self.request_type, prompt=instance.question)


# =============================================================================
# Tool Calling Smoke Test
# =============================================================================

# Weather tool schema for testing tool calls
_WEATHER_TOOL = ToolSchema(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["location"],
    },
)


@register("smoke_toolcall")
class ToolCallSmoke(Task):
    """Smoke test: can the model make tool calls?

    Verifies that the model can correctly invoke a tool when provided with
    a tool schema. The test asks about weather, expecting the model to call
    the get_current_weather tool.

    Scores 1.0 if the model calls the expected tool, 0.0 otherwise.
    """

    sampling_params = SamplingParams(temperature=0.0)
    metrics = (AccuracyMetric(scorer=ToolCallScorer),)

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(
            question="What's the weather like in Seattle?",
            gold_answer="",
            expected_tool_calls=({"name": "get_current_weather"},),
            metadata={"id": "toolcall", "check_type": "tool_call"},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.COMPLETION,
            messages=({"role": "user", "content": instance.question},),
            tools=(_WEATHER_TOOL,),
        )


# =============================================================================
# Reasoning Smoke Test
# =============================================================================


@register("smoke_reasoning")
class ReasoningSmoke(Task):
    """Smoke test: does the model return reasoning content?

    Verifies that reasoning models correctly parse and return their
    chain-of-thought reasoning in the response. This test asks a simple
    question and checks that the reasoning field is populated.

    Scores 1.0 if reasoning is present, 0.0 otherwise.
    """

    sampling_params = SamplingParams(temperature=0.0)
    metrics = (AccuracyMetric(scorer=ReasoningScorer),)

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(
            question="Who are you?",
            gold_answer="",
            metadata={"id": "reasoning", "check_type": "reasoning_present"},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.COMPLETION,
            messages=({"role": "user", "content": instance.question},),
        )

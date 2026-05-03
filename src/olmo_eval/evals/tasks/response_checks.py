"""Response tests for verifying model response properties.

Simple tests to verify models respond correctly with expected properties.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

from olmo_eval.common.formatters import CompletionFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import Scorer, SubstringRecallScorer, ToolCallScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    SamplingParams,
    ToolSchema,
)
from olmo_eval.evals.tasks.common import Task, register


@dataclass(frozen=True, slots=True)
class NonEmptyResponseScorer(Scorer):
    """Score 1.0 if model produced a non-empty response, else 0.0."""

    name: ClassVar[str] = "non_empty_response"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0 if output.text and output.text.strip() else 0.0


@dataclass(frozen=True, slots=True)
class ReasoningResponseScorer(Scorer):
    """Score 1.0 if model produced reasoning content, else 0.0.

    This verifies that reasoning models correctly return their chain-of-thought
    in the reasoning field of the response.
    """

    name: ClassVar[str] = "reasoning_present"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return 1.0 if output.has_reasoning else 0.0


# =============================================================================
# Content Verification Response Test
# =============================================================================


@register("response_match")
class ResponseContentVerify(Task):
    """Verify that model responses contain expected content.

    - Use without data_source (default): Asks "Who are you?" and checks for non-empty response

    - Use with adhoc data_source: Loads prompts and expected substrings from file
        and checks that each response contains the expected substring.

    Data file format (JSONL):
        {"question": "Who are you?", "expected_substring": "OLMo"}
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    formatter = CompletionFormatter(template="User: {question}\nAssistant:")
    metrics = (
        AccuracyMetric(scorer=SubstringRecallScorer),
        AccuracyMetric(scorer=NonEmptyResponseScorer),
    )
    primary_metric = AccuracyMetric(scorer=SubstringRecallScorer)

    def process_doc(self, doc: dict, index: int = 0) -> Instance:
        return Instance(
            question=doc["question"],
            gold_answer=doc.get("expected_substring", ""),
            metadata={"id": f"response_match_{index}", "check_type": "substring"},
        )

    @property
    def instances(self) -> Iterator[Instance]:
        if self.config.data_source is not None:
            yield from self._load_instances()
        else:
            yield Instance(
                question="Who are you?",
                gold_answer="",
                metadata={"id": "response_match_default", "check_type": "substring"},
            )

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(request_type=self.request_type, prompt=instance.question)


# =============================================================================
# Tool Calling Response Test
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


@register("response_toolcall")
class ResponseToolCall(Task):
    """Response test: can the model make tool calls?

    Verifies that the model can correctly invoke a tool when provided with
    a tool schema. The test asks about weather, expecting the model to call
    the get_current_weather tool.
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
# Reasoning Response Test
# =============================================================================


@register("response_reasoning")
class ResponseReasoning(Task):
    """Response test: does the model return reasoning content?

    Verifies that reasoning models correctly parse and return their
    chain-of-thought reasoning in the response. This test asks a simple
    question and checks that the reasoning field is populated.
    """

    sampling_params = SamplingParams(temperature=0.0)
    metrics = (AccuracyMetric(scorer=ReasoningResponseScorer),)

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

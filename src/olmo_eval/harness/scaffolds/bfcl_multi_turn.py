"""Rollout scaffold for the BFCL multi-turn categories.

A multi-turn entry is a conversation, not a single request. For each user turn
the model is asked for calls, those calls are run against the entry's API
instances, their results go back as the next message, and the turn ends when
the model stops calling. What the model did is graded afterwards, by replaying
its calls against fresh instances, so the state built up here only serves to
answer the model.

Two of the categories perturb the conversation. A turn may hold back functions
the model needs, offering them partway through with a stock message; and a turn
may be one the model cannot satisfy yet, where calling anything is the failure
being measured.
"""

from __future__ import annotations

import logging
from typing import Any

from olmo_eval.common.scorers.bfcl import DecodeError, decode_text, decode_tool_calls
from olmo_eval.common.scorers.bfcl.constants import Language
from olmo_eval.common.scorers.bfcl.multi_turn import (
    build_instances,
    calls_from_decoded,
    execute_calls,
)
from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams, ToolSchema
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

#: Steps the model may take within one turn before the entry is abandoned.
#: Matches the reference implementation's limit.
MAXIMUM_STEP_LIMIT = 20

#: What the user says when functions held back from a turn are offered.
ADDITIONAL_FUNCTION_PROMPT = (
    "I have updated some more functions you can choose from. What about now?"
)


@register_scaffold("bfcl_multi_turn")
class BFCLMultiTurnScaffold(Scaffold):
    """Drive one BFCL multi-turn entry to completion."""

    name = "bfcl_multi_turn"

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        payload = request.metadata or {}
        if "turns" not in payload:
            raise ValueError(
                "The bfcl_multi_turn scaffold needs a request carrying the entry's turns; "
                "it is meant for the bfcl_multi_turn_* tasks."
            )

        turns: list[list[dict[str, Any]]] = payload["turns"]
        held_back: dict[str, list[dict[str, Any]]] = payload.get("missed_function") or {}
        language = Language(payload.get("language", Language.PYTHON))
        from_tool_calls = payload.get("call_source") == "tool_calls"
        max_steps = int(payload.get("max_steps", MAXIMUM_STEP_LIMIT))

        instances = build_instances(
            payload["initial_config"],
            payload["involved_classes"],
            bool(payload.get("long_context", False)),
        )

        messages: list[dict[str, Any]] = [dict(m) for m in request.messages]
        tools: list[ToolSchema] = list(request.tools or ())
        calls_per_turn: list[list[list[dict[str, Any]]]] = []
        step_budget_exhausted = False
        last_output = LMOutput(text="")

        for turn_index, turn_messages in enumerate(turns):
            if str(turn_index) in held_back:
                offered = held_back[str(turn_index)]
                # A prompted model reads its functions from the conversation, so
                # the ones being offered have to appear in the message itself.
                announcement = ADDITIONAL_FUNCTION_PROMPT
                if from_tool_calls:
                    tools.extend(ToolSchema.from_openai(schema) for schema in offered)
                else:
                    announcement = f"{offered}\n{announcement}"
                turn_messages = [{"role": "user", "content": announcement}]

            messages.extend(dict(m) for m in turn_messages)
            steps: list[list[dict[str, Any]]] = []

            for _ in range(max_steps):
                outputs = await provider.agenerate(
                    [
                        LMRequest(
                            request_type=RequestType.CHAT,
                            messages=tuple(messages),
                            tools=tuple(tools) or None,
                            system_prompt=request.system_prompt,
                        )
                    ],
                    sampling_params,
                )
                last_output = outputs[0][0]

                decoded = self._decode(last_output, from_tool_calls, language, payload)
                if not decoded:
                    self._record_assistant(messages, last_output, from_tool_calls)
                    break

                steps.append(decoded)
                results = execute_calls(calls_from_decoded(decoded), instances)
                self._record_assistant(messages, last_output, from_tool_calls)
                self._record_results(messages, last_output, results, from_tool_calls)
            else:
                step_budget_exhausted = True
                logger.warning(
                    "BFCL multi-turn entry %s stopped after %s steps on turn %s.",
                    (trace_metadata or {}).get("instance_id", "?"),
                    max_steps,
                    turn_index,
                )

            calls_per_turn.append(steps)
            if step_budget_exhausted:
                break

        final = LMOutput(text=last_output.text, metadata=dict(last_output.metadata or {}))
        final.extracted_answer = calls_per_turn
        if step_budget_exhausted:
            final.metadata["bfcl_step_budget_exhausted"] = True

        return HarnessResult(
            final_output=final,
            max_turns_reached=step_budget_exhausted,
            metadata={"turns": len(calls_per_turn)},
        )

    def _decode(
        self,
        output: LMOutput,
        from_tool_calls: bool,
        language: Language,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Read one reply's calls, treating an unreadable reply as none."""
        try:
            if from_tool_calls:
                return decode_tool_calls(output.tool_calls, payload.get("name_map") or {})
            return decode_text(output.text, language)
        except DecodeError:
            return []

    def _record_assistant(
        self, messages: list[dict[str, Any]], output: LMOutput, from_tool_calls: bool
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": output.text or ""}
        if from_tool_calls and output.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in output.tool_calls]
        messages.append(message)

    def _record_results(
        self,
        messages: list[dict[str, Any]],
        output: LMOutput,
        results: list[str],
        from_tool_calls: bool,
    ) -> None:
        """Put the execution results where the model will read them next."""
        if from_tool_calls and output.tool_calls:
            for call, result in zip(output.tool_calls, results, strict=False):
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            return
        # A prompted model has no tool role to read, so the results come back
        # as the user's next message, as the reference implementation does.
        messages.append({"role": "user", "content": repr(results)})

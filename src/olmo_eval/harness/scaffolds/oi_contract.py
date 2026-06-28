"""Scaffold for the oe-sci-litreview / open-instruct `olmo` tool-calling contract.

sftlab's OLMo-3 SFT checkpoints are trained on the open-instruct `olmo` chat template, whose
tool convention differs from the official Olmo-3 (`[func(arg='x')]` pythonic) format that
olmo-eval's `openai_agents` scaffold + `olmo3_tool_parser` target. The contract format is:

  - tools rendered as an OpenAI function-def array inside <functions>...</functions> on the system
    turn (the `functions` field of open-instruct's olmo template);
  - the model calls tools by emitting <function_calls>[{"name":..,"arguments":{..}}]</function_calls>
    (a JSON array) after an optional <think>...</think>;
  - tool results come back as an `environment` role turn;
  - the run ends when the assistant emits <answer>...</answer> OUTSIDE a <think> block.

Rather than bridge this to the OpenAI tools= API + a custom vLLM tool-call parser, this scaffold
drives the loop DIRECTLY using the model's native text format via COMPLETION requests: it builds the
exact `<|im_start|>...` prompt the model trained on, parses <function_calls> itself, dispatches to the
canonical contract tools (search / browse / s2_search) backed by olmo-eval's Serper + Semantic Scholar
helpers, and feeds observations back as `environment` turns. No chat-template field passthrough or
vLLM parser is required.

The 3 canonical tool schemas are exactly those baked into the training data's <functions> block.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.common.types.tools import ToolCall
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)

# Canonical contract tools — schemas verbatim from the training data's <functions> block.
CONTRACT_FUNCTIONS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Web search. Accepts one or more queries; returns ranked results per query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                      "description": "One or more search queries."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "browse",
        "description": "Fetch one or more web pages as clean markdown. Optional goal focuses extraction.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                    "description": "One or more page URLs to fetch."},
            "goal": {"type": "string", "description": "The specific information goal for visiting the page(s)."}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "s2_search",
        "description": "Search Semantic Scholar for scientific papers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "year": {"type": "string"},
            "fieldsOfStudy": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"]}}},
]

DEFAULT_SYSTEM = (
    "You are OLMo, a helpful function-calling AI assistant built by Ai2. You answer research "
    "questions by issuing tool calls to search the web and academic literature, reading the "
    "returned results, and reasoning over them. Use <think>...</think> for private reasoning, "
    "call tools with <function_calls>[{\"name\":..,\"arguments\":..}]</function_calls>, and give "
    "your final grounded answer inside <answer>...</answer> with <cite id=...> tags."
)

_FUNCTION_CALLS_RE = re.compile(r"<function_calls>\s*(.*?)\s*</function_calls>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# An <answer> that appears only inside <think> is scratch reasoning, not terminal (CONTRACT §6).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _render_system(system_message: str) -> str:
    fns = json.dumps(CONTRACT_FUNCTIONS, ensure_ascii=False)
    return f"<|im_start|>system\n{system_message} <functions>{fns}</functions><|im_end|>\n"


def _terminal_answer(text: str) -> str | None:
    """Return the terminal <answer> body if one exists OUTSIDE any <think> block, else None."""
    stripped = _THINK_RE.sub("", text)
    m = _ANSWER_RE.search(stripped)
    return m.group(1).strip() if m else None


def _parse_calls(text: str) -> list[dict[str, Any]] | None:
    """Parse the first <function_calls>[...]</function_calls> JSON array (outside <think>)."""
    stripped = _THINK_RE.sub("", text)
    m = _FUNCTION_CALLS_RE.search(stripped)
    if not m:
        return None
    try:
        calls = json.loads(m.group(1))
        return calls if isinstance(calls, list) else [calls]
    except json.JSONDecodeError:
        return None


@register_scaffold("oi_contract")
class OIContractScaffold(Scaffold):
    """Multi-turn loop for the open-instruct olmo / oe-sci-litreview tool-calling contract."""

    name = "oi_contract"

    async def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Run a contract tool call and return the observation text."""
        from olmo_eval.harness.tools.search import (
            semantic_scholar_search,
            serper_fetch_page,
            serper_web_search,
        )
        try:
            if name == "search":
                queries = args.get("query") or []
                if isinstance(queries, str):
                    queries = [queries]
                parts = [f"### Results for {q!r}\n{await serper_web_search(q)}" for q in queries]
                return "\n\n".join(parts) if parts else "No queries provided."
            if name == "browse":
                urls = args.get("url") or []
                if isinstance(urls, str):
                    urls = [urls]
                parts = [await serper_fetch_page(u) for u in urls]
                return "\n\n".join(parts) if parts else "No urls provided."
            if name == "s2_search":
                return await semantic_scholar_search(args.get("query", ""))
            return f"Error: unknown tool {name!r}. Available: search, browse, s2_search."
        except Exception as e:  # tool failures must not kill the loop
            logger.warning("tool %s failed: %s", name, e)
            return f"Error running {name}: {type(e).__name__}: {e}"

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        max_turns = config.max_turns or 10
        system_message = config.system_prompt or DEFAULT_SYSTEM
        # The question is the last user message (or the request prompt).
        question = request.prompt
        for m in request.messages:
            if m.get("role") == "user":
                question = m.get("content", question)

        sp = sampling_params or SamplingParams()
        # Per-turn generation: stop at the end of the assistant turn so we can parse + dispatch.
        turn_sp = SamplingParams(
            max_tokens=kwargs.get("max_turn_tokens", 4096),
            temperature=sp.temperature,
            top_p=sp.top_p,
            stop_sequences=("<|im_end|>",),
        )

        prompt = _render_system(system_message)
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n"

        turns: list[AgentTurn] = []
        final_text = ""
        max_turns_reached = False

        for _turn in range(max_turns):
            prompt += "<|im_start|>assistant\n"
            req = LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)
            # agenerate (not the blocking sync generate) so the streaming runner's concurrency holds.
            outputs = await provider.agenerate([req], turn_sp)
            gen = outputs[0][0].text if outputs and outputs[0] else ""

            answer = _terminal_answer(gen)
            calls = None if answer is not None else _parse_calls(gen)

            tool_calls = [
                ToolCall.create(
                    call_id=f"call_{_turn}_{i}",
                    name=c.get("name", ""),
                    arguments=c.get("arguments", {}) or {},
                )
                for i, c in enumerate(calls or [])
            ]
            turns.append(AgentTurn.assistant(content=gen, tool_calls=tool_calls or None))

            if answer is not None:
                final_text = answer
                break
            if not calls:
                # No tool call and no terminal answer: take the generation as the final output.
                final_text = _THINK_RE.sub("", gen).strip() or gen.strip()
                break

            # Execute each call, gather observations into one environment turn.
            obs_parts = []
            for c in calls:
                obs_parts.append(await self._dispatch(c.get("name", ""), c.get("arguments", {}) or {}))
            observation = "\n\n".join(obs_parts)

            prompt += gen + "<|im_end|>\n"
            prompt += f"<|im_start|>environment\n{observation}<|im_end|>\n"
        else:
            max_turns_reached = True
            final_text = final_text or "Maximum number of tool-use turns reached without a final answer."

        return HarnessResult(
            final_output=LMOutput(text=final_text),
            trajectory=AgentTrajectory(turns=tuple(turns)) if turns else None,
            max_turns_reached=max_turns_reached,
        )

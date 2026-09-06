"""OpenAI Agents SDK scaffold."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import fields, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.scaffolds import Scaffold, register_scaffold
from olmo_eval.harness.tools import Tool
from olmo_eval.harness.tools.search import search_date_cutoff
from olmo_eval.inference.base import InferenceProvider

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import ExecutorBinding, SandboxManager

logger = logging.getLogger(__name__)

_current_binding: ContextVar[ExecutorBinding | None] = ContextVar("_current_binding", default=None)
FORCED_FINAL_ANSWER_INSTRUCTION = (
    "You have reached the maximum number of steps. Based on the information gathered so far, "
    "provide your final answer now. Do not call any tools."
)

# Chat template defaults applied to self-hosted requests unless configured otherwise.
# Thinking is pinned off because no agentic preset configures a vLLM reasoning parser,
# so a template that defaults thinking on would leak <think> blocks into scored output.
DEFAULT_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

# Sentinel distinguishing "provider does not support chat_template_kwargs" from
# "provider supports it but has none configured".
_UNSUPPORTED: Any = object()


def _resolve_chat_template_kwargs(provider: InferenceProvider) -> dict[str, Any] | None:
    """Resolve the ``chat_template_kwargs`` request field for a provider.

    ``chat_template_kwargs`` is a vLLM extension to the OpenAI chat completions
    body. Managed APIs reject unrecognized body fields with a 400, so it may only
    be sent to self-hosted OpenAI-compatible servers. Providers in that family
    expose a ``chat_template_kwargs`` attribute; every other provider opts out
    simply by not having one.

    Defaults only fill gaps, so an explicitly configured value always wins. The
    result is always a fresh dict, so the provider's own configuration can never
    be mutated through it.

    This runs once per agent creation, and agents are cached per config and
    provider, so applying a default is logged at INFO while the outcomes that
    change nothing stay at DEBUG.

    Args:
        provider: The inference provider backing the agent.

    Returns:
        The kwargs to send in the request body, or None if the provider does not
        support the field.
    """
    configured = getattr(provider, "chat_template_kwargs", _UNSUPPORTED)
    if configured is _UNSUPPORTED:
        logger.debug(
            f"{type(provider).__name__} does not accept chat_template_kwargs; "
            "omitting it from the request body"
        )
        return None

    explicit: dict[str, Any] = configured or {}
    resolved: dict[str, Any] = dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
    resolved.update(explicit)

    defaulted = {k: v for k, v in DEFAULT_CHAT_TEMPLATE_KWARGS.items() if k not in explicit}
    if defaulted:
        # Only spell out the full payload when the provider configured other keys too.
        detail = "" if resolved == defaulted else f"; sending {resolved}"
        logger.info(f"Defaulted chat_template_kwargs {defaulted} for {provider.model_name}{detail}")
    else:
        logger.debug(
            f"Using explicitly configured chat_template_kwargs for "
            f"{provider.model_name}: {resolved}"
        )
    return resolved


def _reasoning_text(raw: Any) -> str:
    """Extract the reasoning text carried by an SDK ``ResponseReasoningItem``.

    Chat Completions ``reasoning_content`` arrives as a single ``summary`` part.
    Providers that return thinking blocks also fill ``content`` with the full
    text, which then supersedes the summary so nothing is stored twice.

    Args:
        raw: The ``raw_item`` of a ``ReasoningItem``.

    Returns:
        The joined reasoning text, or an empty string if the item has none.
    """
    for attr in ("content", "summary"):
        parts = getattr(raw, attr, None) or []
        texts = [text for part in parts if (text := getattr(part, "text", ""))]
        if texts:
            return "\n\n".join(texts)
    return ""


def _mirror_vllm_reasoning(message: Any) -> None:
    """Expose a provider's ``reasoning`` field under the name the SDK converter reads.

    Providers such as vLLM 0.19 return the reasoning parser output as
    ``message.reasoning``, a field outside the OpenAI schema, while openai-agents 0.20
    builds a ``ReasoningItem`` only from ``reasoning_content`` or ``thinking_blocks``.
    Copying the text onto ``reasoning_content`` before conversion keeps the thinking.
    The copy should be removed once the SDK reads ``reasoning`` itself (openai-agents
    0.21.1 and later, which require openai 3.x).

    Nothing changes when the server already returned a non-empty ``reasoning_content``
    or ``thinking_blocks``, or when the message carries no reasoning at all (OpenAI, or
    vLLM without a reasoning parser). The SDK replays ``reasoning_content`` into later
    requests only for model names containing ``deepseek``, so a DeepSeek-named model
    served by vLLM sends the mirrored text back as assistant history; vLLM 0.19.1
    ignores that key.

    Args:
        message: The ``ChatCompletionMessage`` of a choice, mutated in place.
    """
    if getattr(message, "reasoning_content", None) or getattr(message, "thinking_blocks", None):
        return
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        # ChatCompletionMessage allows extra fields, so this lands in model_extra next
        # to ``reasoning`` and is picked up by the converter's getattr.
        message.reasoning_content = reasoning


@lru_cache(maxsize=1)
def _chat_completions_model_class() -> type:
    """Return the SDK model class the scaffold instantiates, importing the SDK lazily.

    The subclass hooks the point where the raw ``ChatCompletion`` is visible before
    conversion: ``OpenAIChatCompletionsModel._fetch_response``. The SDK's ``get_response``
    (the path behind ``Runner.run``) awaits it with ``stream=False`` and hands
    ``choices[0].message`` to ``Converter.message_to_output_items``. ``stream_response``
    calls it with ``stream=True`` and receives a ``(Response, AsyncStream)`` pair, which
    passes through untouched; the SDK's stream handler already reads ``delta.reasoning``
    on that path. The hook is private SDK surface, so the scaffold tests pin it. The
    subclass becomes unnecessary with openai-agents 0.21.1 or later, whose converter
    reads ``reasoning`` itself.
    """
    from agents import OpenAIChatCompletionsModel  # type: ignore[ty:unresolved-import]
    from openai.types.chat import ChatCompletion

    class ReasoningFieldChatCompletionsModel(OpenAIChatCompletionsModel):
        """Chat Completions model that normalizes the vLLM reasoning field."""

        async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
            completion = await super()._fetch_response(*args, **kwargs)
            if isinstance(completion, ChatCompletion):
                for choice in completion.choices:
                    message = getattr(choice, "message", None)
                    if message is not None:
                        _mirror_vllm_reasoning(message)
            return completion

    return ReasoningFieldChatCompletionsModel


#: The managed OpenAI endpoint. Every other base url this harness talks to -- a self-hosted
#: vLLM, DeepSeek, anything behind litellm -- is a chat completions server and stays on that
#: route.
OPENAI_API_HOST = "api.openai.com"
#: The one ``reasoning_effort`` the managed endpoint will answer on chat completions while a
#: request carries ``tools``. Since GPT-5.4 every other value is refused outright, and the 5.x
#: models default to ``medium``, so the refusal fires with the field unset too.
NO_REASONING = "none"
#: The summary verbosity asked for on the Responses route. ``concise`` is rejected by the 5.x
#: series; ``auto`` lets the provider choose. Without it no reasoning comes back at all.
REASONING_SUMMARY = "auto"
#: What a Responses answer's reasoning is. Recorded on the turn beside the text, because the
#: provider writes it and never exposes the model's own chain: a reader of one saved turn must
#: not take it for the thinking the vLLM arms record.
SUMMARY_REASONING_KIND = "summary"


def _client_host(client: Any) -> str:
    """The host an OpenAI client sends to, however its base url is spelled."""

    base_url = getattr(client, "base_url", None)
    if base_url is None:
        return ""
    host = getattr(base_url, "host", None)
    if not host:
        from urllib.parse import urlsplit

        host = urlsplit(str(base_url)).hostname
    return (host or "").lower()


def _configured_effort(model_settings: Any) -> str | None:
    """The ``reasoning_effort`` a run pinned, or None when it left it to the provider."""

    reasoning = getattr(model_settings, "reasoning", None)
    return getattr(reasoning, "effort", None)


def responses_api_required(client: Any, model_settings: Any) -> bool:
    """Whether this agent has to run against ``/v1/responses`` rather than chat completions.

    Only the managed OpenAI endpoint, and only when the run asked for reasoning by name. An
    unpinned effort deliberately does **not** count, even though a 5.x model defaults to
    ``medium`` and would refuse a chat completion carrying tools: the same endpoint serves
    models that have no reasoning at all, and sending them a ``reasoning`` block is its own 400.
    So the run says which it wants. ``reasoning_effort=none`` remains the way to keep tools on
    chat completions, exactly as the preset's docstring describes; naming any other effort is
    the way to keep the model thinking, and moves the agent here.
    """

    if _client_host(client) != OPENAI_API_HOST:
        return False
    effort = _configured_effort(model_settings)
    return bool(effort) and effort != NO_REASONING


def with_reasoning_summary(model_settings: Any) -> Any:
    """``model_settings`` asking the Responses route for a reasoning summary.

    The effort the run pinned is kept; an unpinned one stays unpinned, so the provider's own
    default stands. Without ``summary`` the answer carries a reasoning item with nothing in it
    and the saved trajectory has no reasoning at all.
    """

    from agents import ModelSettings  # type: ignore[ty:unresolved-import]
    from openai.types.shared import Reasoning

    reasoning = getattr(model_settings, "reasoning", None)
    effort = getattr(reasoning, "effort", None)
    summary = Reasoning(effort=effort, summary=REASONING_SUMMARY)
    if model_settings is None:
        return ModelSettings(reasoning=summary)
    return replace(model_settings, reasoning=summary)


def _model_call_usage(result: Any) -> list[dict[str, int]] | None:
    """The token usage of every model call in one run, in the order the calls were made.

    The SDK keeps a ``ModelResponse`` per call with the provider's own usage on it, and this
    scaffold was throwing it away: a finished run recorded what the agent did and nothing about
    what it cost, so a reasoning arm could only be priced from a billing dashboard after the
    fact. The reasoning tokens are the interesting half -- they are billed as output, they are
    the part a reasoning effort actually changes, and they are invisible in the trajectory
    because the provider returns a summary rather than the thinking.

    Args:
        result: The ``RunResult`` from ``Runner.run``.

    Returns:
        One entry per model call, or None when the run made none.
    """
    calls: list[dict[str, int]] = []
    for response in getattr(result, "raw_responses", None) or []:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        entry = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0)
        if reasoning:
            entry["reasoning_tokens"] = int(reasoning)
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0)
        if cached:
            entry["cached_input_tokens"] = int(cached)
        calls.append(entry)
    return calls or None


def _make_tool_error_formatter(valid_tool_names: Sequence[str]) -> Any:
    """Build a ``RunConfig.tool_error_formatter`` that names the tools the model may call.

    The SDK default for a missing tool is ``Tool 'x' not found.``, which tells the model
    nothing about what it should have called. Weak models only recover when the error
    spells out the real names, so the inventory is captured here: ``ToolErrorFormatterArgs``
    carries the *failed* name but no list of valid ones.

    Args:
        valid_tool_names: Names the model is actually allowed to call.

    Returns:
        A formatter suitable for ``RunConfig.tool_error_formatter``.
    """
    if valid_tool_names:
        guidance = f"Available tools: {', '.join(valid_tool_names)}. Call one of these exact names."
    else:
        guidance = "No tools are available."

    def format_tool_error(args: Any) -> str | None:
        # Approval rejections are routed through this same hook; returning None leaves
        # the SDK default in place for every kind we have nothing better to say about.
        if args.kind != "tool_not_found":
            return None
        return f"Error: tool '{args.tool_name}' does not exist. {guidance}"

    return format_tool_error


def build_model_settings(spec: Mapping[str, Any] | None) -> Any | None:
    """Build the agents SDK's ``ModelSettings`` from a plain dict of settings.

    ``reasoning_effort`` is accepted as a flat key because that is the name of
    the request field and what an operator types on a command line. The SDK
    models it as ``ModelSettings.reasoning.effort``, which
    ``OpenAIChatCompletionsModel`` sends as a top-level ``reasoning_effort``
    argument -- the supported route. It is deliberately not routed through
    ``extra_args``: that same call site splats ``extra_args`` into the request
    next to its own explicit ``reasoning_effort=`` keyword, so the two would
    collide on exactly this key.

    Any other key must name a ``ModelSettings`` field, and an unknown one raises
    rather than being dropped. A knob that silently does nothing is the failure
    this function exists to fix: runs carried ``OLMO_EVAL_REASONING_EFFORT`` for
    two weeks while nothing read it, and they looked fine because the server
    default happened to agree.

    Returns None when nothing is configured, which leaves the Agent on the SDK's
    own defaults rather than pinning them to this function's idea of them.
    """
    if not spec:
        return None

    from agents import ModelSettings  # type: ignore[ty:unresolved-import]

    requested = dict(spec)
    kwargs: dict[str, Any] = {}

    effort = requested.pop("reasoning_effort", None)
    if effort is not None:
        from openai.types.shared import Reasoning

        kwargs["reasoning"] = Reasoning(effort=effort)

    known = {field.name for field in fields(ModelSettings)}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(
            f"Unknown model_settings key(s): {', '.join(unknown)}. Valid keys are "
            f"'reasoning_effort' or any ModelSettings field: {', '.join(sorted(known))}."
        )
    kwargs.update(requested)

    return ModelSettings(**kwargs)


@register_scaffold("openai_agents")
class OpenAIAgentsScaffold(Scaffold):
    """Scaffold that delegates execution to OpenAI Agents SDK.

    This scaffold converts Harness tools to the agents SDK format
    and uses the SDK's Runner for execution.
    """

    name = "openai_agents"
    required_extras = ("agents",)

    def __init__(self) -> None:
        self._cached_agent: Any = None  # Agent type from agents SDK
        self._cached_config: HarnessConfig | None = None
        self._cached_provider_id: int | None = None
        self._cached_has_sandbox: bool = False
        self._sandbox_manager: SandboxManager | None = None
        #: What the reasoning the current agent produces is, when it is not the chain:
        #: ``summary`` once :meth:`_create_agent` has chosen the Responses route.
        self._reasoning_kind: str | None = None

    def clear_cache(self) -> None:
        """Clear cached agent to allow recreation with new config/provider."""
        self._cached_agent = None
        self._cached_config = None
        self._cached_provider_id = None
        self._cached_has_sandbox = False

    async def initialize(self, config: HarnessConfig) -> None:
        """Initialize sandbox manager if needed.

        Called during worker startup to create the sandbox before processing.
        """
        needs_sandbox = config.sandboxes and config.has_sandbox_tools

        if needs_sandbox and self._sandbox_manager is None:
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()
            logger.info(
                f"Sandbox manager started with {self._sandbox_manager.executor_count} executor(s)"
            )

    async def cleanup(self) -> None:
        """Clean up resources including sandbox manager."""
        if self._sandbox_manager is not None:
            await self._sandbox_manager.stop()
            self._sandbox_manager = None
        self.clear_cache()

    def _convert_tools(
        self,
        tools: Sequence[Tool],
        function_tool: Any,
        sandbox_manager: SandboxManager | None = None,
    ) -> list[Any]:
        """Convert harness tools to agents SDK format.

        Args:
            tools: Sequence of Tool instances to convert.
            function_tool: The function_tool decorator from agents SDK.
            sandbox_manager: Optional sandbox manager for tools that require it.

        Returns:
            List of agents SDK tool objects.
        """
        agent_tools = []
        for tool in tools:
            execute_fn = tool.execute

            # Wrap sandboxed tools to use the manager
            if tool.sandbox and sandbox_manager is not None:
                execute_fn = self._wrap_sandboxed_tool(tool, sandbox_manager)

            # Use function_tool decorator to wrap the execute function
            wrapped = function_tool(strict_mode=False)(execute_fn)
            # Override name and description
            wrapped.name = tool.name
            if hasattr(wrapped, "description"):
                wrapped.description = tool.description
            agent_tools.append(wrapped)
        return agent_tools

    def _wrap_sandboxed_tool(
        self,
        tool: Tool,
        manager: SandboxManager,
    ) -> Any:
        """Create a wrapper function that executes the tool via sandbox manager.

        Args:
            tool: The tool requiring sandbox execution.
            manager: The sandbox manager to use for routing.

        Returns:
            An async function that executes commands via the sandbox.
        """
        required_caps = tool.sandbox

        if tool.session:

            async def sandboxed_execute(command: str) -> str:
                """Execute command in sandbox session."""
                binding = _current_binding.get()
                if binding is None:
                    raise RuntimeError("No binding set for session tool")
                result = await binding.execute_in_session(command)
                output = result.output
                if result.exit_code != 0:
                    output += f"\n[Exit code: {result.exit_code}]"
                return output
        else:

            async def sandboxed_execute(command: str) -> str:
                """Execute command in sandbox."""
                return await manager.execute(command, capabilities=required_caps)

        return sandboxed_execute

    def _create_agent(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        sandbox_manager: SandboxManager | None = None,
    ) -> Any:
        """Create a new agent with the given configuration.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration.
            sandbox_manager: Optional sandbox manager for sandboxed tools.

        Returns:
            An Agent instance from the agents SDK.
        """
        from agents import (  # type: ignore[ty:unresolved-import]
            Agent,
            ModelSettings,
            function_tool,
            set_tracing_disabled,
        )

        from olmo_eval.inference.utils import patch_openai_agents_for_vllm

        # Disable trace export to OpenAI's backend (we don't have OPENAI_API_KEY set)
        set_tracing_disabled(True)

        patch_openai_agents_for_vllm()

        # Create model using provider's OpenAI client
        client = provider.get_openai_client()
        logger.debug(
            f"Creating agent with client: {type(client).__name__}, "
            f"base_url={getattr(client, 'base_url', 'unknown')}, "
            f"model={provider.model_name}"
        )

        agent_tools = self._convert_tools(config.resolved_tools, function_tool, sandbox_manager)

        # Only set model_settings when something was configured: the Agent's own
        # default is an empty ModelSettings, and passing one built here would
        # replace the SDK's defaults with this scaffold's idea of them.
        model_settings = build_model_settings(config.scaffold_kwargs.get("model_settings"))
        if model_settings is not None:
            logger.debug(f"Applying model_settings from scaffold_kwargs: {model_settings}")

        # Which of the two OpenAI routes this agent runs on. The managed endpoint refuses a
        # chat completion that carries tools at any effort but none, so a thinking run there
        # goes to /v1/responses; the tools, the loop and everything else are unchanged, and the
        # reasoning that comes back is a provider summary rather than the model's own chain.
        # Every other provider, and this one at effort none, keeps the chat completions model.
        self._reasoning_kind = None
        if responses_api_required(client, model_settings):
            from agents import OpenAIResponsesModel  # type: ignore[ty:unresolved-import]

            model_settings = with_reasoning_summary(model_settings)
            self._reasoning_kind = SUMMARY_REASONING_KIND
            logger.info(
                f"Using the Responses API for {provider.model_name}: chat completions refuses "
                f"function tools at reasoning effort "
                f"{_configured_effort(model_settings) or 'the provider default'}"
            )
            model: Any = OpenAIResponsesModel(openai_client=client, model=provider.model_name)
        else:
            # vLLM returns thinking as ``reasoning``, which the SDK does not convert; the
            # subclass mirrors it onto ``reasoning_content`` so it reaches the saved trajectory.
            model = _chat_completions_model_class()(
                openai_client=client,
                model=provider.model_name,
            )

        agent_kwargs: dict[str, Any] = {
            "name": self.name,
            "instructions": config.system_prompt or "",
            "model": model,
            "tools": agent_tools,
        }

        # This scaffold drives the OpenAI client directly instead of the provider's
        # generate path, so the request body built here is the only place a
        # chat_template_kwargs setting can actually reach the server. It rides in
        # extra_body next to whatever the run pinned, and setdefault leaves a
        # chat_template_kwargs the run set itself alone.
        chat_template_kwargs = _resolve_chat_template_kwargs(provider)
        if chat_template_kwargs is not None:
            extra_body = dict(getattr(model_settings, "extra_body", None) or {})
            extra_body.setdefault("chat_template_kwargs", chat_template_kwargs)
            model_settings = (
                ModelSettings(extra_body=extra_body)
                if model_settings is None
                else replace(model_settings, extra_body=extra_body)
            )

        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings

        agent = Agent(**agent_kwargs)

        return agent

    def _get_or_create_agent(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        sandbox_manager: SandboxManager | None = None,
    ) -> Any:
        """Get cached agent or create a new one if config/provider changed.

        Agents are cached based on config, provider, and whether sandbox is used.
        The sandbox manager is stable across runs, so caching works.
        """
        has_sandbox = sandbox_manager is not None
        if (
            self._cached_agent is not None
            and self._cached_config == config
            and self._cached_provider_id == id(provider)
            and self._cached_has_sandbox == has_sandbox
        ):
            return self._cached_agent

        agent = self._create_agent(provider, config, sandbox_manager)

        self._cached_agent = agent
        self._cached_config = config
        self._cached_provider_id = id(provider)
        self._cached_has_sandbox = has_sandbox

        return agent

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        """Execute using OpenAI Agents SDK.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).
            **kwargs: Scaffold-specific options:
                - enable_compaction: Enable context compaction (default: True).
                - model_settings: Request settings applied when the agent is
                  built (see :func:`build_model_settings`); accepted here so
                  that splatting scaffold_kwargs into this call is harmless.

        Returns:
            HarnessResult with trajectory from SDK execution.
        """
        enable_compaction = kwargs.get("enable_compaction", True)
        try:
            from agents import RunConfig, Runner, trace  # type: ignore[ty:unresolved-import]
            from agents.exceptions import (  # type: ignore[ty:unresolved-import]
                MaxTurnsExceeded,
                ModelBehaviorError,
                ModelRefusalError,
            )
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        # Create compaction session if enabled
        session = None
        if enable_compaction:
            try:
                from agents import SQLiteSession  # type: ignore[ty:unresolved-import]
                from agents.memory import (  # type: ignore[ty:unresolved-import]
                    OpenAIResponsesCompactionSession,
                )

                session_id = (trace_metadata or {}).get("task_id", "default")
                # Use an in-memory SQLite session as the underlying storage
                underlying = SQLiteSession(session_id, db_path=":memory:")
                session = OpenAIResponsesCompactionSession(
                    session_id=session_id,
                    underlying_session=underlying,
                )
            except ImportError:
                logger.warning("Context compaction not available - agents.memory not found")

        # Check if we need sandbox execution
        needs_sandbox = config.sandboxes and config.has_sandbox_tools

        # Lazily create and cache the sandbox manager
        if needs_sandbox and self._sandbox_manager is None:
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()
            logger.info(
                f"Sandbox manager started with {self._sandbox_manager.executor_count} executor(s)"
            )

        # Use cached agent (tools read from ContextVar at execution time)
        agent = self._get_or_create_agent(provider, config, self._sandbox_manager)

        # Acquire binding if session tools are used
        has_session_tools = any(t.session for t in config.resolved_tools if t.sandbox)
        binding_token = None

        if has_session_tools and self._sandbox_manager:
            session_caps = frozenset().union(
                *(t.sandbox for t in config.resolved_tools if t.session and t.sandbox)
            )
            binding = await self._sandbox_manager.acquire_binding(session_caps)
            binding_token = _current_binding.set(binding)

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Track if max turns was reached
        max_turns_reached = False
        max_turns = config.max_turns or 10

        # Build trace name from config and metadata
        instance_id = (trace_metadata or {}).get("instance_id", "")
        if instance_id:
            trace_name = f"{config.name}:{instance_id}" if config.name else f"Agent:{instance_id}"
        else:
            trace_name = f"Agent: {config.name}" if config.name else "Agent run"

        # An unknown tool name is a recoverable mistake, not a dead run: let the SDK hand the
        # model an error turn naming the tools it may call instead of aborting the whole
        # instance with ModelBehaviorError. This covers function tools only -- the SDK has no
        # equivalent branch for custom/freeform calls.
        run_config = RunConfig(
            tool_not_found_behavior="return_error_to_model",
            tool_error_formatter=_make_tool_error_formatter(
                [tool.name for tool in config.resolved_tools]
            ),
        )

        # Run agent within trace context for observability
        date_cutoff = (trace_metadata or {}).get("date_cutoff")
        with search_date_cutoff(date_cutoff), trace(trace_name, metadata=trace_metadata):
            try:
                run_kwargs: dict[str, Any] = {
                    "starting_agent": agent,
                    "input": input_text,
                    "max_turns": max_turns,
                    "run_config": run_config,
                }
                if session is not None:
                    run_kwargs["session"] = session

                result = await Runner.run(**run_kwargs)
            except MaxTurnsExceeded as e:
                # Return a result with the error instead of raising
                partial_result = getattr(e, "run_data", None)
                trajectory = self._convert_trajectory(partial_result)
                try:
                    final_text = await self._force_final_answer(
                        Runner=Runner,
                        agent=agent,
                        partial_result=partial_result,
                        original_input=input_text,
                        run_config=run_config,
                    )
                except Exception:
                    logger.warning(
                        "Forced final answer after max_turns failed; using fallback result",
                        exc_info=True,
                    )
                    return HarnessResult(
                        trajectory=AgentTrajectory(turns=()),
                        final_output=LMOutput(text="[Max turns exceeded]"),
                        max_turns_reached=True,
                        error=f"Max turns ({max_turns}) exceeded",
                    )

                return HarnessResult(
                    trajectory=trajectory,
                    final_output=LMOutput(text=final_text),
                    max_turns_reached=True,
                    error=None,
                )
            except ModelBehaviorError as e:
                # Unknown function tools are handled by run_config above, so this is now a
                # backstop for the other ways a model can violate the protocol.
                return HarnessResult(
                    trajectory=AgentTrajectory(turns=()),
                    final_output=LMOutput(text=f"[Tool error: {e}]"),
                    error=str(e),
                )
            except ModelRefusalError as e:
                # A refusal used to arrive as empty final output; the SDK now raises instead.
                # Record it as a scored instance with an error, rather than letting a refusal
                # take down the whole run.
                return HarnessResult(
                    trajectory=AgentTrajectory(turns=()),
                    final_output=LMOutput(text=f"[Model refusal: {e}]"),
                    error=str(e),
                )
            except Exception as e:
                # Log full traceback for debugging connection issues
                import traceback

                logger.error(f"Agent run failed: {e}\n{traceback.format_exc()}")
                raise
            finally:
                # Release binding after run
                binding = _current_binding.get()
                if binding is not None:
                    await binding.release()
                    if binding_token is not None:
                        _current_binding.reset(binding_token)

        # Convert result to HarnessResult
        trajectory = self._convert_trajectory(result)
        final_text = result.final_output if hasattr(result, "final_output") else ""

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text or ""),
            max_turns_reached=max_turns_reached,
            error="Max turns exceeded" if max_turns_reached else None,
        )

    async def _force_final_answer(
        self,
        *,
        Runner: Any,
        agent: Any,
        partial_result: Any,
        original_input: str,
        run_config: Any = None,
    ) -> str:
        """Run one no-tool model call to produce a final answer after max_turns."""
        final_input = self._build_forced_final_input(partial_result, original_input)
        model_settings = replace(agent.model_settings, tool_choice="none")
        final_agent = replace(
            agent,
            tools=[],
            handoffs=[],
            mcp_servers=[],
            model_settings=model_settings,
        )
        final_result = await Runner.run(
            starting_agent=final_agent,
            input=final_input,
            max_turns=1,
            run_config=run_config,
        )
        final_text = getattr(final_result, "final_output", "")
        return str(final_text or "")

    def _build_forced_final_input(
        self,
        partial_result: Any,
        original_input: str,
    ) -> list[Any]:
        """Build model input from the partial run plus a final-answer instruction."""
        input_list: list[Any] = []

        if partial_result is not None and hasattr(partial_result, "to_input_list"):
            try:
                input_list = list(partial_result.to_input_list())
            except Exception:
                input_list = []

        if not input_list and partial_result is not None:
            original = getattr(partial_result, "input", original_input)
            if isinstance(original, list):
                input_list.extend(original)
            elif original:
                input_list.append({"role": "user", "content": str(original)})

            for item in getattr(partial_result, "new_items", None) or []:
                if hasattr(item, "to_input_item"):
                    try:
                        input_list.append(item.to_input_item())
                    except Exception:
                        continue

        if not input_list and original_input:
            input_list.append({"role": "user", "content": original_input})

        input_list.append({"role": "user", "content": FORCED_FINAL_ANSWER_INSTRUCTION})
        return input_list

    def _convert_trajectory(self, result: Any) -> AgentTrajectory:
        """Convert agents SDK result to AgentTrajectory.

        Args:
            result: Result from Runner.run().

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []
        if result is None:
            return AgentTrajectory(turns=tuple(turns))
        # What the run cost, kept beside what it did -- on the Responses route only, which is
        # the route that needed it and the only one whose trajectory shape is new. Every other
        # route keeps the trajectory it has always written: the SDK builds a Usage object for
        # every call whether or not the provider reported one, so attaching this everywhere
        # would put a `model_calls` key (zeros included) on every self-hosted run ever saved
        # from here on.
        usage = _model_call_usage(result) if self._reasoning_kind else None
        metadata = {"model_calls": usage} if usage else {}

        # Get items from new_items (primary source in agents SDK)
        items = getattr(result, "new_items", None) or []
        if not items:
            # Fallback to to_input_list() for full conversation history
            if hasattr(result, "to_input_list"):
                try:
                    input_list = result.to_input_list()
                    if input_list:
                        return self._convert_input_list_to_trajectory(input_list)
                except Exception:
                    pass
            return AgentTrajectory(turns=tuple(turns), metadata=metadata)

        # The SDK emits reasoning as its own item ahead of the message or tool call
        # it belongs to, so hold it until that assistant turn is built.
        pending_reasoning: str | None = None

        for item in items:
            item_class = type(item).__name__

            if item_class == "ReasoningItem":
                text = _reasoning_text(getattr(item, "raw_item", None))
                if text:
                    pending_reasoning = (
                        f"{pending_reasoning}\n\n{text}" if pending_reasoning else text
                    )

            elif item_class == "MessageOutputItem":
                raw = getattr(item, "raw_item", None)
                content = ""
                if raw is not None:
                    raw_content = getattr(raw, "content", None)
                    if raw_content:
                        for part in raw_content:
                            if hasattr(part, "text"):
                                content += part.text
                if content:
                    # One model response yields one reasoning block. When the response has
                    # text it lands here, and any tool-call turns built from the same
                    # response carry none.
                    turns.append(
                        AgentTurn.assistant(
                            content=content,
                            reasoning=pending_reasoning,
                            reasoning_kind=self._reasoning_kind,
                        )
                    )
                    pending_reasoning = None
                elif pending_reasoning:
                    # A message with no text part (e.g. a refusal) still ends the response
                    # that produced this reasoning; keep it here so it cannot slide onto a
                    # later, unrelated turn.
                    turns.append(
                        AgentTurn.assistant(
                            reasoning=pending_reasoning,
                            reasoning_kind=self._reasoning_kind,
                        )
                    )
                    pending_reasoning = None

            elif item_class == "ToolCallItem":
                raw = getattr(item, "raw_item", None)
                if raw is not None:
                    call_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or ""
                    name = getattr(raw, "name", "") or ""
                    arguments = getattr(raw, "arguments", "{}") or "{}"
                    raw_dict = raw.model_dump() if hasattr(raw, "model_dump") else {}
                    tool_call = ToolCall.create(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        metadata=raw_dict,
                    )
                    # Without a text part the block lands on the first tool-call turn of
                    # the response; the remaining tool-call turns carry none.
                    turns.append(
                        AgentTurn.assistant(
                            content="",
                            tool_calls=[tool_call],
                            reasoning=pending_reasoning,
                            reasoning_kind=self._reasoning_kind,
                        )
                    )
                    pending_reasoning = None

            elif item_class == "ToolCallOutputItem":
                if pending_reasoning:
                    # A tool output means the response that produced this reasoning is
                    # over (its own items had no branch above); flush rather than carry it.
                    turns.append(
                        AgentTurn.assistant(
                            reasoning=pending_reasoning,
                            reasoning_kind=self._reasoning_kind,
                        )
                    )
                    pending_reasoning = None
                output = getattr(item, "output", None)
                raw = getattr(item, "raw_item", None)
                # Extract tool_call_id from raw_item
                tool_call_id = ""
                if raw is not None:
                    if isinstance(raw, dict):
                        tool_call_id = (
                            raw.get("call_id", "")
                            or raw.get("tool_call_id", "")
                            or raw.get("id", "")
                            or ""
                        )
                    else:
                        tool_call_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or ""
                content = str(output) if output is not None else ""
                tool_result = ToolResult(
                    tool_call_id=tool_call_id,
                    content=content,
                )
                turns.append(AgentTurn.tool([tool_result]))

        if pending_reasoning:
            # No assistant item followed this reasoning (the run stopped after it);
            # keep it on an otherwise empty assistant turn rather than dropping it.
            turns.append(
                AgentTurn.assistant(
                    reasoning=pending_reasoning, reasoning_kind=self._reasoning_kind
                )
            )

        return AgentTrajectory(turns=tuple(turns), metadata=metadata)

    def _convert_input_list_to_trajectory(self, input_list: list[Any]) -> AgentTrajectory:
        """Convert input list (from to_input_list()) to AgentTrajectory.

        This is a fallback for when new_items is empty but we have the full
        conversation history available via to_input_list().

        Args:
            input_list: List of input items from result.to_input_list().

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        for item in input_list:
            # Items can be dicts or objects
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                tool_calls = item.get("tool_calls", [])

                if role == "assistant":
                    if tool_calls:
                        converted_calls = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                call_id = tc.get("id", "")
                                func = tc.get("function", {})
                                is_dict = isinstance(func, dict)
                                name = func.get("name", "") if is_dict else ""
                                args = func.get("arguments", "{}") if is_dict else "{}"
                            else:
                                call_id = getattr(tc, "id", "")
                                func = getattr(tc, "function", None)
                                name = getattr(func, "name", "") if func else ""
                                args = getattr(func, "arguments", "{}") if func else "{}"
                            converted_calls.append(
                                ToolCall.create(call_id=call_id, name=name, arguments=args)
                            )
                        turns.append(
                            AgentTurn.assistant(content=content, tool_calls=converted_calls)
                        )
                    elif content:
                        turns.append(AgentTurn.assistant(content=content))

                elif role == "tool":
                    tool_call_id = item.get("tool_call_id", "")
                    tool_result = ToolResult(tool_call_id=tool_call_id, content=content)
                    turns.append(AgentTurn.tool([tool_result]))

                elif role == "user":
                    turns.append(AgentTurn.user(content=content))
            else:
                # Handle object-based items
                item_type = type(item).__name__
                role = getattr(item, "role", None) or getattr(item, "type", "")

                is_assistant = item_type in ("ResponseOutputMessage", "MessageOutputItem")
                if is_assistant or role == "assistant":
                    content = ""
                    raw_content = getattr(item, "content", None)
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif raw_content:
                        for part in raw_content:
                            if hasattr(part, "text"):
                                content += part.text
                    if content:
                        turns.append(AgentTurn.assistant(content=content))

        return AgentTrajectory(turns=tuple(turns))

"""Default backend for generation"""

from __future__ import annotations

from typing import TYPE_CHECKING

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types.trajectory import AgentTrajectory, AgentTurn

from ..result import HarnessResult
from . import Backend, register_backend

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

    from ..config import HarnessConfig


@register_backend("default")
class DefaultBackend(Backend):
    """Single-turn generation backend.

    Applies harness config and calls generate. No tool execution.
    """

    name = "default"

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute a single-turn generation.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request to process.
            sampling_params: Optional sampling parameters override.

        Returns:
            HarnessResult with single turn and final output.
        """
        # Apply config to the request (tools, system prompt, etc.)
        transformed = self._apply_config(config, request)

        # Generate using async method
        outputs = await provider.agenerate([transformed], sampling_params)
        output = outputs[0][0] if outputs and outputs[0] else LMOutput(text="")

        # Build trajectory with a single assistant turn
        turn = AgentTurn.assistant(content=output.text)
        trajectory = AgentTrajectory(turns=(turn,))

        return HarnessResult(
            trajectory=trajectory,
            final_output=output,
            max_turns_reached=False,
        )

    def _apply_config(self, config: HarnessConfig, request: LMRequest) -> LMRequest:
        """Inject tool schemas and system prompt from config.

        Args:
            config: Harness configuration.
            request: Original request.

        Returns:
            New request with config applied.
        """
        messages = self._inject_system_prompt(config, request.messages)

        return LMRequest(
            request_type=request.request_type,
            messages=messages,
            prompt=request.prompt,
            continuations=request.continuations,
            tools=config.tool_schemas if config.has_tools else request.tools,
            system_prompt=config.system_prompt or request.system_prompt,
        )

    def _inject_system_prompt(
        self, config: HarnessConfig, messages: tuple[dict[str, object], ...]
    ) -> tuple[dict[str, object], ...]:
        """Add system prompt to messages if configured and not present.

        Args:
            config: Harness configuration.
            messages: Original message tuple.

        Returns:
            Messages with system prompt prepended if needed.
        """
        if not config.system_prompt:
            return messages

        # Check if messages already start with a system message
        if messages and messages[0].get("role") == "system":
            return messages

        # Prepend system message
        system_msg: dict[str, object] = {
            "role": "system",
            "content": config.system_prompt,
        }
        return (system_msg,) + messages

"""Default backend: Single-turn generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from olmo_eval.core.types import SamplingParams
from olmo_eval.core.types.trajectory import AgentTrajectory, AgentTurn

from ..result import HarnessResult
from . import Backend, register_backend

if TYPE_CHECKING:
    from olmo_eval.core.types import LMRequest

    from ..harness import Harness


@register_backend("default")
class DefaultBackend(Backend):
    """Single-turn generation backend.

    Applies harness config and calls generate once. No tool execution.
    """

    async def run(
        self,
        harness: Harness,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute a single generation.

        Args:
            harness: The Harness instance.
            request: The request to process.
            sampling_params: Optional sampling parameters.

        Returns:
            HarnessResult with the generation output.
        """
        turn_request = harness._apply_config(request)
        outputs = harness.provider.generate([turn_request], sampling_params)
        output = outputs[0][0]

        turn = AgentTurn.assistant(
            content=output.text,
            tool_calls=output.tool_calls or [],
        )

        return HarnessResult(
            trajectory=AgentTrajectory(turns=(turn,)),
            final_output=output,
        )

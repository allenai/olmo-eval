"""Scorers for trajectory evaluation.

This module provides scorers that evaluate complete agent trajectories
including tool call sequences, state changes, and efficiency.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response


def _default_state_comparator(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Default state comparator using equality.

    Args:
        expected: Expected state.
        actual: Actual state.

    Returns:
        True if states match, False otherwise.
    """
    return expected == actual


@dataclass(frozen=True, slots=True)
class TrajectoryResponseScorer(Scorer):
    """Score trajectory based on tool call sequence (transcript check).

    Evaluates whether the trajectory contains the expected tool call
    sequence in order.
    """

    name: ClassVar[str] = "trajectory_response"
    require_exact_match: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score trajectory response correctness.

        Note: This scorer requires access to the Response object's trajectory.
        When used through score(), it falls back to checking output.tool_calls.

        Args:
            instance: The evaluation instance.
            output: The model output.

        Returns:
            1.0 if trajectory matches, 0.0 otherwise.
        """
        # Fallback: use tool_calls from output
        if instance.required_trajectory is None:
            return 0.0

        required_names = []
        for step in instance.required_trajectory:
            name = step.get("name", step.get("function", {}).get("name", ""))
            if name:
                required_names.append(name)

        if not required_names:
            return 1.0  # No requirements

        if output.tool_calls is None or len(output.tool_calls) == 0:
            return 0.0

        actual_names = [call.function.name for call in output.tool_calls]

        if self.require_exact_match:
            return 1.0 if required_names == actual_names else 0.0
        else:
            # Subsequence check
            return 1.0 if self._is_subsequence(required_names, actual_names) else 0.0

    def score_response(self, response: Response) -> float:
        """Score a complete response with trajectory.

        Args:
            response: The complete response with trajectory.

        Returns:
            1.0 if trajectory matches, 0.0 otherwise.
        """
        instance = response.instance
        if instance.required_trajectory is None:
            return 0.0

        required_names = []
        for step in instance.required_trajectory:
            name = step.get("name", step.get("function", {}).get("name", ""))
            if name:
                required_names.append(name)

        if not required_names:
            return 1.0

        # Prefer trajectory if available
        if response.trajectory is not None:
            actual_names = response.trajectory.tool_call_names()
        elif response.outputs and response.outputs[0].tool_calls:
            actual_names = [call.function.name for call in response.outputs[0].tool_calls]
        else:
            return 0.0

        if self.require_exact_match:
            return 1.0 if required_names == actual_names else 0.0
        else:
            return 1.0 if self._is_subsequence(required_names, actual_names) else 0.0

    def _is_subsequence(self, required: list[str], actual: list[str]) -> bool:
        """Check if required is a subsequence of actual."""
        req_idx = 0
        for name in actual:
            if req_idx < len(required) and name == required[req_idx]:
                req_idx += 1
        return req_idx == len(required)


@dataclass(frozen=True)
class TrajectoryStateScorer(Scorer):
    """Score trajectory based on final state (outcome check).

    Evaluates whether the final state matches expected state.
    """

    name: ClassVar[str] = "trajectory_state"
    state_comparator: Callable[[dict[str, Any], dict[str, Any]], bool] = field(
        default=_default_state_comparator
    )

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score state correctness.

        Note: This scorer requires trajectory state, which is only available
        via score_response(). This fallback returns 0.0.

        Args:
            instance: The evaluation instance.
            output: The model output.

        Returns:
            0.0 (use score_response for full functionality).
        """
        # Cannot check state without trajectory
        return 0.0

    def score_response(self, response: Response) -> float:
        """Score a complete response with trajectory.

        Args:
            response: The complete response with trajectory.

        Returns:
            1.0 if final state matches expected, 0.0 otherwise.
        """
        instance = response.instance
        if instance.expected_final_state is None:
            return 0.0

        if response.trajectory is None:
            return 0.0

        actual_state = response.trajectory.state_snapshot
        expected_state = instance.expected_final_state

        return 1.0 if self.state_comparator(expected_state, actual_state) else 0.0


@dataclass(frozen=True, slots=True)
class TrajectoryEfficiencyScorer(Scorer):
    """Score trajectory efficiency (minimal_steps / actual_steps).

    Evaluates how efficiently the agent completed the task compared
    to the minimum required steps.
    """

    name: ClassVar[str] = "trajectory_efficiency"
    minimal_steps: int = 1

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score efficiency based on output tool calls.

        Args:
            instance: The evaluation instance.
            output: The model output.

        Returns:
            Efficiency ratio (0.0 to 1.0).
        """
        if output.tool_calls is None or len(output.tool_calls) == 0:
            return 0.0

        actual_steps = len(output.tool_calls)
        if actual_steps == 0:
            return 0.0

        # Use minimal_steps from instance if available
        min_steps = self.minimal_steps
        if instance.required_trajectory is not None:
            min_steps = len(instance.required_trajectory)
        if min_steps <= 0:
            min_steps = 1

        return min(1.0, min_steps / actual_steps)

    def score_response(self, response: Response) -> float:
        """Score a complete response with trajectory.

        Args:
            response: The complete response with trajectory.

        Returns:
            Efficiency ratio (0.0 to 1.0).
        """
        instance = response.instance

        # Get actual steps from trajectory or output
        if response.trajectory is not None:
            actual_steps = response.trajectory.total_tool_calls
        elif response.outputs and response.outputs[0].tool_calls:
            actual_steps = len(response.outputs[0].tool_calls)
        else:
            return 0.0

        if actual_steps == 0:
            return 0.0

        # Get minimal steps
        min_steps = self.minimal_steps
        if instance.required_trajectory is not None:
            min_steps = len(instance.required_trajectory)
        if min_steps <= 0:
            min_steps = 1

        return min(1.0, min_steps / actual_steps)


@dataclass(frozen=True)
class TrajectoryCombinedScorer(Scorer):
    """Score requiring both response AND state to pass.

    This scorer combines trajectory response and state checks,
    returning 1.0 only if both pass.
    """

    name: ClassVar[str] = "trajectory_combined"
    require_exact_sequence: bool = False
    state_comparator: Callable[[dict[str, Any], dict[str, Any]], bool] = field(
        default=_default_state_comparator
    )

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score combined trajectory check.

        Note: Full functionality requires score_response().

        Args:
            instance: The evaluation instance.
            output: The model output.

        Returns:
            0.0 (use score_response for full functionality).
        """
        return 0.0

    def score_response(self, response: Response) -> float:
        """Score a complete response requiring both checks to pass.

        Args:
            response: The complete response with trajectory.

        Returns:
            1.0 if both response and state pass, 0.0 otherwise.
        """
        instance = response.instance

        # Check response (tool sequence)
        response_pass = self._check_response(response)
        if not response_pass:
            return 0.0

        # Check state if expected
        if instance.expected_final_state is not None:
            state_pass = self._check_state(response)
            if not state_pass:
                return 0.0

        return 1.0

    def _check_response(self, response: Response) -> bool:
        """Check if tool sequence matches."""
        instance = response.instance
        if instance.required_trajectory is None:
            return True  # No requirement

        required_names = []
        for step in instance.required_trajectory:
            name = step.get("name", step.get("function", {}).get("name", ""))
            if name:
                required_names.append(name)

        if not required_names:
            return True

        if response.trajectory is not None:
            actual_names = response.trajectory.tool_call_names()
        elif response.outputs and response.outputs[0].tool_calls:
            actual_names = [call.function.name for call in response.outputs[0].tool_calls]
        else:
            return False

        if self.require_exact_sequence:
            return required_names == actual_names
        else:
            return self._is_subsequence(required_names, actual_names)

    def _check_state(self, response: Response) -> bool:
        """Check if final state matches."""
        instance = response.instance
        if instance.expected_final_state is None:
            return True

        if response.trajectory is None:
            return False

        return self.state_comparator(
            instance.expected_final_state,
            response.trajectory.state_snapshot,
        )

    def _is_subsequence(self, required: list[str], actual: list[str]) -> bool:
        """Check if required is a subsequence of actual."""
        req_idx = 0
        for name in actual:
            if req_idx < len(required) and name == required[req_idx]:
                req_idx += 1
        return req_idx == len(required)

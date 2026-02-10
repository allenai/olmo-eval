"""Scorers for tool calling evaluation.

This module provides scorers for evaluating tool call accuracy,
argument matching, and tool sequences.
"""

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


def _parse_arguments(args: str | dict[str, Any]) -> dict[str, Any]:
    """Parse arguments from string or dict.

    Args:
        args: Arguments as JSON string or dict.

    Returns:
        Parsed arguments dict.
    """
    if isinstance(args, dict):
        return args
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}


def _normalize_value(value: Any, case_sensitive: bool = True) -> Any:
    """Normalize a value for comparison.

    Args:
        value: The value to normalize.
        case_sensitive: Whether to preserve case for strings.

    Returns:
        Normalized value.
    """
    if isinstance(value, str):
        return value if case_sensitive else value.lower()
    elif isinstance(value, list):
        return [_normalize_value(v, case_sensitive) for v in value]
    elif isinstance(value, dict):
        return {k: _normalize_value(v, case_sensitive) for k, v in value.items()}
    return value


def _compare_arguments_ast(
    expected: dict[str, Any],
    actual: dict[str, Any],
    case_sensitive: bool = True,
) -> bool:
    """Compare arguments using AST-style matching (BFCL approach).

    This performs structural comparison of argument values,
    handling nested structures and type coercion.

    Args:
        expected: Expected arguments.
        actual: Actual arguments.
        case_sensitive: Whether string comparison is case-sensitive.

    Returns:
        True if arguments match, False otherwise.
    """
    expected_norm = _normalize_value(expected, case_sensitive)
    actual_norm = _normalize_value(actual, case_sensitive)

    # Check all expected keys are present with matching values
    for key, exp_val in expected_norm.items():
        if key not in actual_norm:
            return False
        act_val = actual_norm[key]

        # Handle type coercion for numbers
        if isinstance(exp_val, int | float) and isinstance(act_val, int | float):
            if float(exp_val) != float(act_val):
                return False
        elif exp_val != act_val:
            return False

    return True


@dataclass(frozen=True, slots=True)
class ToolCallScorer(Scorer):
    """Score 1.0 if the correct tool name was called, 0.0 otherwise.

    Checks if the model called the expected tool based on
    instance.expected_tool_calls.
    """

    name: ClassVar[str] = "tool_call"

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score tool call correctness.

        Args:
            instance: The evaluation instance with expected_tool_calls.
            output: The model output with tool_calls.

        Returns:
            1.0 if correct tool called, 0.0 otherwise.
        """
        if instance.expected_tool_calls is None:
            return 0.0

        # Get expected tool names
        expected_names = {
            call.get("name", call.get("function", {}).get("name", ""))
            for call in instance.expected_tool_calls
        }

        if not expected_names:
            return 0.0

        # Get actual tool names
        if output.tool_calls is None or len(output.tool_calls) == 0:
            return 0.0

        actual_names = {call.function.name for call in output.tool_calls}

        # Check if any expected tool was called
        if expected_names & actual_names:
            return 1.0
        return 0.0


@dataclass(frozen=True, slots=True)
class ToolArgumentScorer(Scorer):
    """Score tool argument accuracy using BFCL-style AST matching.

    Compares the arguments of each tool call against expected arguments.
    """

    name: ClassVar[str] = "tool_argument"
    case_sensitive: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score tool argument correctness.

        Args:
            instance: The evaluation instance with expected_tool_calls.
            output: The model output with tool_calls.

        Returns:
            Proportion of correctly matched arguments (0.0 to 1.0).
        """
        if instance.expected_tool_calls is None:
            return 0.0

        if output.tool_calls is None or len(output.tool_calls) == 0:
            return 0.0

        # Build map of expected calls by name
        expected_by_name: dict[str, list[dict[str, Any]]] = {}
        for call in instance.expected_tool_calls:
            name = call.get("name", call.get("function", {}).get("name", ""))
            args = call.get("arguments", call.get("function", {}).get("arguments", {}))
            if name:
                if name not in expected_by_name:
                    expected_by_name[name] = []
                expected_by_name[name].append(_parse_arguments(args))

        if not expected_by_name:
            return 0.0

        # Score each actual call
        total_calls = 0
        correct_calls = 0

        for actual_call in output.tool_calls:
            name = actual_call.function.name
            if name not in expected_by_name:
                total_calls += 1
                continue

            actual_args = actual_call.get_arguments()
            total_calls += 1

            # Check if any expected call matches
            for expected_args in expected_by_name[name]:
                if _compare_arguments_ast(expected_args, actual_args, self.case_sensitive):
                    correct_calls += 1
                    break

        if total_calls == 0:
            return 0.0

        return correct_calls / total_calls


@dataclass(frozen=True, slots=True)
class ToolSequenceScorer(Scorer):
    """Score whether required tool sequence appears as subsequence.

    Checks if the required tools were called in order (but not
    necessarily consecutively).
    """

    name: ClassVar[str] = "tool_sequence"
    strict_order: bool = True

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score tool sequence correctness.

        Args:
            instance: The evaluation instance with required_trajectory.
            output: The model output with tool_calls.

        Returns:
            1.0 if required sequence found, 0.0 otherwise.
        """
        if instance.required_trajectory is None:
            return 0.0

        # Extract required tool names in order
        required_names = []
        for step in instance.required_trajectory:
            name = step.get("name", step.get("function", {}).get("name", ""))
            if name:
                required_names.append(name)

        if not required_names:
            return 0.0

        # Get actual tool names in order
        if output.tool_calls is None or len(output.tool_calls) == 0:
            return 0.0

        actual_names = [call.function.name for call in output.tool_calls]

        if self.strict_order:
            # Check if required is subsequence of actual
            return 1.0 if self._is_subsequence(required_names, actual_names) else 0.0
        else:
            # Just check all required tools were called (any order)
            return 1.0 if set(required_names) <= set(actual_names) else 0.0

    def _is_subsequence(self, required: list[str], actual: list[str]) -> bool:
        """Check if required is a subsequence of actual.

        Args:
            required: The required sequence.
            actual: The actual sequence.

        Returns:
            True if required is a subsequence of actual.
        """
        req_idx = 0
        for name in actual:
            if req_idx < len(required) and name == required[req_idx]:
                req_idx += 1
        return req_idx == len(required)

"""Tests for CLI utility functions."""

import click
import pytest

from olmo_eval.cli.utils import (
    FlaggedArg,
    extract_priority_from_overrides,
    process_ordered_args,
    reconstruct_ordered_args,
)


class TestFlaggedArg:
    """Tests for FlaggedArg dataclass."""

    def test_creation(self):
        """Test creating FlaggedArg."""
        arg = FlaggedArg(flag="m", value="llama3.1-8b")
        assert arg.flag == "m"
        assert arg.value == "llama3.1-8b"


class TestReconstructOrderedArgs:
    """Tests for reconstruct_ordered_args function."""

    def test_empty_args(self):
        """Test with empty arguments."""
        result = reconstruct_ordered_args([])
        assert result == []

    def test_single_task(self):
        """Test single task argument."""
        result = reconstruct_ordered_args(["-t", "mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_single_task_long_form(self):
        """Test single task with --task."""
        result = reconstruct_ordered_args(["--task", "mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_task_with_equals(self):
        """Test task with = syntax."""
        result = reconstruct_ordered_args(["-t=mmlu"])
        assert len(result) == 1
        assert result[0].flag == "t"
        assert result[0].value == "mmlu"

    def test_multiple_tasks(self):
        """Test multiple tasks."""
        result = reconstruct_ordered_args(
            [
                "-t",
                "mmlu",
                "-t",
                "gsm8k",
            ]
        )
        assert len(result) == 2
        assert result[0] == FlaggedArg("t", "mmlu")
        assert result[1] == FlaggedArg("t", "gsm8k")

    def test_tasks_with_overrides(self):
        """Test tasks with overrides in order."""
        result = reconstruct_ordered_args(
            [
                "-t",
                "mmlu",
                "-o",
                "limit=100",
                "-o",
                "batch_size=8",
                "-t",
                "gsm8k",
                "-o",
                "limit=50",
            ]
        )
        assert len(result) == 5
        assert result[0] == FlaggedArg("t", "mmlu")
        assert result[1] == FlaggedArg("o", "limit=100")
        assert result[2] == FlaggedArg("o", "batch_size=8")
        assert result[3] == FlaggedArg("t", "gsm8k")
        assert result[4] == FlaggedArg("o", "limit=50")

    def test_ignores_other_flags(self):
        """Test that non-relevant flags are ignored."""
        result = reconstruct_ordered_args(
            [
                "-m",
                "model1",
                "--cluster",
                "h100",
                "-t",
                "task1",
                "--gpus",
                "4",
            ]
        )
        # Only -t is tracked, -m is not handled by reconstruct_ordered_args
        assert len(result) == 1
        assert result[0] == FlaggedArg("t", "task1")


class TestProcessOrderedArgs:
    """Tests for process_ordered_args function."""

    def test_empty_list(self):
        """Test with empty list."""
        task_overrides, harness_overrides = process_ordered_args([])
        assert task_overrides == {}
        assert harness_overrides == []

    def test_single_task_no_overrides(self):
        """Test single task without overrides."""
        ordered = [FlaggedArg("t", "mmlu")]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {"mmlu": []}
        assert harness_overrides == []

    def test_single_task_with_overrides(self):
        """Test single task with overrides."""
        ordered = [
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {"mmlu": ["limit=100"]}
        assert harness_overrides == []

    def test_multiple_tasks_with_overrides(self):
        """Test multiple tasks with overrides."""
        ordered = [
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
            FlaggedArg("t", "gsm8k"),  # No overrides
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }
        assert harness_overrides == []

    def test_harness_with_overrides(self):
        """Test harness with overrides."""
        ordered = [
            FlaggedArg("h", "default"),
            FlaggedArg("o", "max_turns=10"),
        ]
        task_overrides, harness_overrides = process_ordered_args(ordered)
        assert task_overrides == {}
        assert harness_overrides == ["max_turns=10"]

    def test_override_without_preceding_flag_raises(self):
        """Test that -o without preceding -t or --harness raises error."""
        ordered = [FlaggedArg("o", "gpus=4")]
        with pytest.raises(click.UsageError, match="-o/--override must follow"):
            process_ordered_args(ordered)


class TestExtractPriorityFromOverrides:
    """Tests for extract_priority_from_overrides function."""

    def test_empty_overrides(self):
        """Test with empty overrides dict."""
        priority, filtered = extract_priority_from_overrides({})
        assert priority is None
        assert filtered == {}

    def test_no_priority_override(self):
        """Test with no priority in overrides."""
        task_overrides = {"mmlu": ["limit=100", "batch_size=8"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority is None
        assert filtered == {"mmlu": ["limit=100", "batch_size=8"]}

    def test_priority_extracted(self):
        """Test priority is extracted from overrides."""
        task_overrides = {"mmlu": ["priority=urgent", "limit=100"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "urgent"
        assert filtered == {"mmlu": ["limit=100"]}

    def test_priority_only_override(self):
        """Test when only priority is in overrides - task not in filtered result."""
        task_overrides = {"mmlu": ["priority=high"]}
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "high"
        assert filtered == {}  # No other overrides left

    def test_multiple_tasks_with_priority(self):
        """Test priority from multiple tasks - last one wins."""
        task_overrides = {
            "mmlu": ["priority=normal", "limit=100"],
            "gsm8k": ["priority=urgent"],
        }
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "urgent"  # Last one wins
        assert filtered == {"mmlu": ["limit=100"]}

    def test_preserves_non_priority_overrides(self):
        """Test that non-priority overrides are preserved."""
        task_overrides = {
            "mmlu": ["limit=100", "priority=high", "batch_size=8"],
            "gsm8k": ["limit=50"],
        }
        priority, filtered = extract_priority_from_overrides(task_overrides)
        assert priority == "high"
        assert filtered == {
            "mmlu": ["limit=100", "batch_size=8"],
            "gsm8k": ["limit=50"],
        }


class TestEndToEndOrdering:
    """End-to-end tests for CLI arg ordering."""

    def test_full_command_line(self):
        """Test reconstructing and processing a full command line."""
        args = [
            "run",
            "-t",
            "mmlu",
            "-o",
            "limit=100",
            "-t",
            "gsm8k",
            "--output-dir",
            "/tmp/results",
        ]
        ordered = reconstruct_ordered_args(args)
        task_overrides, harness_overrides = process_ordered_args(ordered)

        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }
        assert harness_overrides == []

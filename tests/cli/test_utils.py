"""Tests for CLI utility functions."""

import click
import pytest

from olmo_eval.cli.utils import FlaggedArg, process_ordered_args, reconstruct_ordered_args


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

    def test_single_model(self):
        """Test single model argument."""
        result = reconstruct_ordered_args(["-m", "llama3.1-8b"])
        assert len(result) == 1
        assert result[0].flag == "m"
        assert result[0].value == "llama3.1-8b"

    def test_single_model_long_form(self):
        """Test single model with --model."""
        result = reconstruct_ordered_args(["--model", "llama3.1-8b"])
        assert len(result) == 1
        assert result[0].flag == "m"
        assert result[0].value == "llama3.1-8b"

    def test_model_with_equals(self):
        """Test model with = syntax."""
        result = reconstruct_ordered_args(["-m=llama3.1-8b"])
        assert len(result) == 1
        assert result[0].flag == "m"
        assert result[0].value == "llama3.1-8b"

    def test_multiple_models_and_tasks(self):
        """Test multiple models and tasks."""
        result = reconstruct_ordered_args(
            [
                "-m",
                "model1",
                "-t",
                "task1",
                "-m",
                "model2",
            ]
        )
        assert len(result) == 3
        assert result[0] == FlaggedArg("m", "model1")
        assert result[1] == FlaggedArg("t", "task1")
        assert result[2] == FlaggedArg("m", "model2")

    def test_models_with_overrides(self):
        """Test models with overrides in order."""
        result = reconstruct_ordered_args(
            [
                "-m",
                "llama3.1-8b",
                "-o",
                "provider.name=vllm",
                "-o",
                "gpus=4",
                "-m",
                "other-model",
                "-t",
                "mmlu",
                "-o",
                "limit=100",
            ]
        )
        assert len(result) == 6
        assert result[0] == FlaggedArg("m", "llama3.1-8b")
        assert result[1] == FlaggedArg("o", "provider.name=vllm")
        assert result[2] == FlaggedArg("o", "gpus=4")
        assert result[3] == FlaggedArg("m", "other-model")
        assert result[4] == FlaggedArg("t", "mmlu")
        assert result[5] == FlaggedArg("o", "limit=100")

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
        assert len(result) == 2
        assert result[0] == FlaggedArg("m", "model1")
        assert result[1] == FlaggedArg("t", "task1")


class TestProcessOrderedArgs:
    """Tests for process_ordered_args function."""

    def test_empty_list(self):
        """Test with empty list."""
        model_overrides, task_overrides = process_ordered_args([])
        assert model_overrides == {}
        assert task_overrides == {}

    def test_single_model_no_overrides(self):
        """Test single model without overrides."""
        ordered = [FlaggedArg("m", "llama3.1-8b")]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {"llama3.1-8b": []}
        assert task_overrides == {}

    def test_single_model_with_overrides(self):
        """Test single model with overrides."""
        ordered = [
            FlaggedArg("m", "llama3.1-8b"),
            FlaggedArg("o", "provider.name=vllm"),
            FlaggedArg("o", "gpus=4"),
        ]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {"llama3.1-8b": ["provider.name=vllm", "gpus=4"]}
        assert task_overrides == {}

    def test_multiple_models_with_overrides(self):
        """Test multiple models with different overrides."""
        ordered = [
            FlaggedArg("m", "llama3.1-8b"),
            FlaggedArg("o", "provider.name=vllm"),
            FlaggedArg("m", "gpt-4o"),
            FlaggedArg("o", "provider.name=litellm"),
        ]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {
            "llama3.1-8b": ["provider.name=vllm"],
            "gpt-4o": ["provider.name=litellm"],
        }
        assert task_overrides == {}

    def test_model_without_overrides_between_others(self):
        """Test model without overrides between models with overrides."""
        ordered = [
            FlaggedArg("m", "model1"),
            FlaggedArg("o", "gpus=2"),
            FlaggedArg("m", "model2"),  # No overrides
            FlaggedArg("m", "model3"),
            FlaggedArg("o", "gpus=4"),
        ]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {
            "model1": ["gpus=2"],
            "model2": [],
            "model3": ["gpus=4"],
        }

    def test_single_task_with_overrides(self):
        """Test single task with overrides."""
        ordered = [
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
        ]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {}
        assert task_overrides == {"mmlu": ["limit=100"]}

    def test_mixed_models_and_tasks(self):
        """Test mixed models and tasks with overrides."""
        ordered = [
            FlaggedArg("m", "llama3.1-8b"),
            FlaggedArg("o", "provider.name=vllm"),
            FlaggedArg("o", "provider.package=vllm==0.14.0"),
            FlaggedArg("m", "other-model"),
            FlaggedArg("t", "mmlu"),
            FlaggedArg("o", "limit=100"),
            FlaggedArg("t", "gsm8k"),  # No overrides
        ]
        model_overrides, task_overrides = process_ordered_args(ordered)
        assert model_overrides == {
            "llama3.1-8b": ["provider.name=vllm", "provider.package=vllm==0.14.0"],
            "other-model": [],
        }
        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }

    def test_override_without_preceding_flag_raises(self):
        """Test that -o without preceding -m or -t raises error."""
        ordered = [FlaggedArg("o", "gpus=4")]
        with pytest.raises(click.UsageError, match="-o/--override must follow"):
            process_ordered_args(ordered)


class TestEndToEndOrdering:
    """End-to-end tests for CLI arg ordering."""

    def test_full_command_line(self):
        """Test reconstructing and processing a full command line."""
        args = [
            "beaker",
            "launch",
            "-m",
            "llama3.1-8b",
            "-o",
            "provider.name=vllm",
            "-o",
            "provider.package=vllm==0.14.0",
            "-m",
            "gpt-4o",
            "-o",
            "provider.name=litellm",
            "-t",
            "mmlu",
            "-o",
            "limit=100",
            "-t",
            "gsm8k",
            "--cluster",
            "h100",
        ]
        ordered = reconstruct_ordered_args(args)
        model_overrides, task_overrides = process_ordered_args(ordered)

        assert model_overrides == {
            "llama3.1-8b": ["provider.name=vllm", "provider.package=vllm==0.14.0"],
            "gpt-4o": ["provider.name=litellm"],
        }
        assert task_overrides == {
            "mmlu": ["limit=100"],
            "gsm8k": [],
        }

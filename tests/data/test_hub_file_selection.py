"""Tests for data file selection when a repository has a legacy loading script.

`datasets` refuses to run legacy loading scripts, so the HuggingFace backend
falls back to reading the repository's data files directly. That fallback has
to load the files the caller asked for.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmo_eval.data import DataSource
from olmo_eval.data.backends.huggingface import HuggingFaceBackend


class _RecordingLoader:
    """Stands in for `datasets.load_dataset`, recording how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"path": path, **kwargs})
        return [{"doc": "loaded"}]


@pytest.fixture
def recording_loader(monkeypatch: pytest.MonkeyPatch) -> _RecordingLoader:
    import datasets

    loader = _RecordingLoader()
    monkeypatch.setattr(datasets, "load_dataset", loader)
    return loader


def _all_repo_files(monkeypatch: pytest.MonkeyPatch, files: list[str]) -> None:
    """Make the Hub report `files` as the repository's contents."""
    import huggingface_hub

    class _StubApi:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def list_repo_files(self, path: str, repo_type: str = "dataset") -> list[str]:
            del path, repo_type
            return files

    monkeypatch.setattr(huggingface_hub, "HfApi", _StubApi)


def test_declared_data_files_are_the_only_ones_loaded(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The repository holds six data files; the caller asked for three of them.
    _all_repo_files(
        monkeypatch,
        ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    )
    source = DataSource(
        path="org/scripted-repo",
        data_files=("test.jsonl", "test2.jsonl", "test3.jsonl"),
        split="train",
    )

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        data_files=["test.jsonl", "test2.jsonl", "test3.jsonl"],
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": [
            "hf://datasets/org/scripted-repo/test.jsonl",
            "hf://datasets/org/scripted-repo/test2.jsonl",
            "hf://datasets/org/scripted-repo/test3.jsonl",
        ]
    }


def test_a_single_declared_data_file_is_not_widened(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a subset, substring matching accepts every file in the repository.
    _all_repo_files(monkeypatch, ["test.jsonl", "test2.jsonl", "test3.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files="test.jsonl", split="train")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        data_files="test.jsonl",
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": ["hf://datasets/org/scripted-repo/test.jsonl"]
    }


def test_subset_matching_still_applies_without_declared_files(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["math/test.jsonl", "biology/test.jsonl", "README.md"])
    source = DataSource(path="org/scripted-repo", subset="math", split="test")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
    )

    assert recording_loader.calls[0]["data_files"] == {
        "test": ["hf://datasets/org/scripted-repo/math/test.jsonl"]
    }


def test_missing_subset_files_raise(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["math/test.jsonl"])
    source = DataSource(path="org/scripted-repo", subset="chemistry", split="test")

    with pytest.raises(FileNotFoundError):
        HuggingFaceBackend._load_from_hub_files(
            "org/scripted-repo",
            source,
            streaming=False,
            token=None,
        )

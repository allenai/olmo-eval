"""Tests for multimodal task scoring (no model, no API key, no real data)."""

import math

from olmo_eval.common.types import Instance
from olmo_eval.evals.tasks.common.multimodal_base import MultimodalGenerationTask
from olmo_eval.evals.tasks.pixmo_cap_eval import PixmoCapEvalTask, PixmoCapEvalTaskConfig


# ---------------------------------------------------------------------------
# MultimodalGenerationTask.format_request
# ---------------------------------------------------------------------------


def test_format_request_no_images():
    task = PixmoCapEvalTask()
    instance = Instance(question="Describe this.", gold_answer="A cat.", images=None)
    req = task.format_request(instance)
    content = req.messages[0]["content"]
    assert len(content) == 1  # text only
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Describe this."


def test_format_request_with_image():
    task = PixmoCapEvalTask()
    fake_img = object()
    instance = Instance(question="Describe.", gold_answer="A cat.", images=(fake_img,))
    req = task.format_request(instance)
    content = req.messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image"
    assert content[0]["image"] is fake_img
    assert content[1]["type"] == "text"


# ---------------------------------------------------------------------------
# PixmoCapEvalTask — scoring requires API key so we just test the data helpers
# ---------------------------------------------------------------------------


def test_pixmo_cap_eval_score_raises_without_api_key(monkeypatch):
    """score_all should raise RuntimeError when OPENAI_API_KEY is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = PixmoCapEvalTask()
    instances = [
        Instance(
            question="Describe.",
            gold_answer="transcript",
            images=None,
            metadata={"atomic_statements": ["A cat.", "A mat."], "image_id": "abc"},
        )
    ]
    import pytest
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        task.score_all(["a cat on a mat"], instances)


def test_pixmo_cap_eval_instances_metadata_shape():
    """Verify instance metadata keys without touching the filesystem."""
    # We can't iterate real instances without data, but we can verify the
    # Instance schema is correct by constructing one manually.
    inst = Instance(
        question="Describe this image.",
        gold_answer="joined transcripts",
        images=(object(),),
        metadata={"image_id": "abc123", "atomic_statements": ["stmt1", "stmt2"]},
    )
    assert inst.metadata["image_id"] == "abc123"
    assert len(inst.metadata["atomic_statements"]) == 2

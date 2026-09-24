"""Tests for the per-instance JSONL writers."""

import functools
import json

from olmo_eval.runners.io.writers import write_requests_jsonl


def _lazy_image(index: int) -> str:
    return f"image-{index}"


def test_write_requests_renders_non_json_metadata_by_repr(tmp_path):
    """A vision instance may carry a lazy image callable in its metadata; the request
    dump must still be written rather than crash on serialization."""
    image = functools.partial(_lazy_image, 3)
    requests = [{"doc": {"query": "q", "image": image}, "request": {"context": "q"}}]

    write_requests_jsonl(str(tmp_path), "my_task", requests, "model", task_hash="abcdef")

    (path,) = tmp_path.glob("requests/model/*.jsonl")
    row = json.loads(path.read_text())
    assert row["doc"]["query"] == "q"
    assert row["doc"]["image"] == repr(image)

"""JSONL writers: lazy image loaders in request records, strictness elsewhere."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from olmo_eval.common.images import (
    CappedImageList,
    HfDatasetImage,
    HfDatasetImageList,
    ImageList,
    PathImage,
)
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.runners.io.builders import build_requests
from olmo_eval.runners.io.writers import write_predictions_jsonl, write_requests_jsonl


def _load_image(path: str) -> None:
    return None


def _written(tmp_path: Path) -> list[dict]:
    (path,) = tmp_path.rglob("*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_request_records_keep_a_placeholder_for_lazy_loaders(tmp_path: Path) -> None:
    loader = functools.partial(_load_image, "cat.png")
    write_requests_jsonl(str(tmp_path), "task", [{"doc": {"images": loader, "q": "?"}}], "m")
    assert _written(tmp_path) == [{"doc": {"images": "<partial>", "q": "?"}}]


def test_request_records_still_reject_other_non_json_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="set"):
        write_requests_jsonl(str(tmp_path), "task", [{"doc": {"tags": {"a"}}}], "m")


def test_prediction_records_stay_strict(tmp_path: Path) -> None:
    loader = functools.partial(_load_image, "cat.png")
    with pytest.raises(TypeError):
        write_predictions_jsonl(str(tmp_path), "task", [{"metrics": loader}], "m")


def test_request_records_serialize_image_references_as_json(tmp_path: Path) -> None:
    inst = Instance(
        question="What is this?",
        metadata={
            "image": HfDatasetImage(dataset=object(), index=3, column="img"),
            "gallery": ImageList((PathImage("/a.png"), HfDatasetImageList(object(), 4, "imgs"))),
            "capped": CappedImageList([PathImage("/b.png")], max_images=2),
        },
    )
    req = LMRequest(request_type=RequestType.CHAT, messages=({"role": "user", "content": "q"},))
    records = build_requests([inst], [req], "vision_task")
    write_requests_jsonl(str(tmp_path), "vision_task", records, "m")

    written = _written(tmp_path)
    assert len(written) == 1
    doc = written[0]["doc"]
    assert doc["image"] == {"type": "hf_dataset_image", "index": 3, "column": "img"}
    assert doc["gallery"] == {
        "type": "image_list",
        "items": [
            {"type": "path", "path": "/a.png"},
            {"type": "hf_dataset_image_list", "index": 4, "column": "imgs"},
        ],
    }
    assert doc["capped"] == {
        "type": "capped_image_list",
        "max_images": 2,
        "images": [{"type": "path", "path": "/b.png"}],
    }


def test_prediction_records_serialize_image_references(tmp_path: Path) -> None:
    write_predictions_jsonl(
        str(tmp_path),
        "vision_task",
        [{"doc_id": 0, "image_ref": PathImage("/tmp/example.png")}],
        "m",
    )
    assert _written(tmp_path) == [
        {"doc_id": 0, "image_ref": {"type": "path", "path": "/tmp/example.png"}}
    ]

"""CI-safe tests for the ``image_mode`` perception-vs-knowledge ablations.

No data files, model, GPU, or network. These guard the three request shapes the ablations
depend on -- real image, no image, caption-instead-of-image -- and the failure modes that
would otherwise produce a plausible-looking but meaningless score.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from olmo_eval.common.types import Instance, RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.image_qa_base import ImageQATask
from olmo_eval.evals.tasks.common.registry import get_task

Image = pytest.importorskip("PIL.Image")


class _DummyImageQATask(ImageQATask):
    def _build_instances(self) -> Iterator[Instance]:
        return iter(())


def _tiny_image():
    return Image.new("RGB", (8, 8))


def _instance(example_id: str = "ex1"):
    return Instance(
        question="What is the value?",
        metadata={"image": _tiny_image(), "example_id": example_id},
    )


def _caption_file(tmp_path, records: dict[str, str]):
    path = tmp_path / "captions.jsonl"
    with open(path, "w") as f:
        for example_id, caption in records.items():
            f.write(json.dumps({"example_id": example_id, "caption": caption}) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# The three modes
# ---------------------------------------------------------------------------


def test_default_mode_still_attaches_the_image():
    # Regression guard: the ablation machinery must not perturb the published benchmarks.
    task = _DummyImageQATask(TaskConfig(name="dummy"))
    instance = _instance()

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.images == (instance.metadata["image"],)
    assert request.messages[0]["content"] == "What is the value?"


def test_text_only_mode_drops_the_image_and_keeps_the_question():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="none"))

    request = task.format_request(_instance())

    assert request.images is None
    assert request.messages[0]["content"] == "What is the value?"


def test_caption_mode_substitutes_the_caption_for_the_image(tmp_path):
    source = _caption_file(tmp_path, {"ex1": "A bar chart peaking at 60."})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    request = task.format_request(_instance())

    assert request.images is None
    content = request.messages[0]["content"]
    assert "A bar chart peaking at 60." in content
    assert content.endswith("What is the value?")


# ---------------------------------------------------------------------------
# Failure modes that would otherwise score silently
# ---------------------------------------------------------------------------


def test_missing_caption_raises_rather_than_falling_back_to_text_only(tmp_path):
    # Silently dropping to text-only would collapse the two conditions the experiment is
    # built to distinguish, and still report a number.
    source = _caption_file(tmp_path, {"other": "irrelevant"})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    with pytest.raises(ValueError, match="No caption for example_id"):
        task.format_request(_instance("ex1"))


def test_caption_mode_without_a_source_raises():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption"))

    with pytest.raises(ValueError, match="requires caption_source"):
        task.format_request(_instance())


def test_unknown_image_mode_raises():
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="oracle"))

    with pytest.raises(ValueError, match="Unknown image_mode"):
        task.format_request(_instance())


def test_empty_caption_file_raises(tmp_path):
    source = _caption_file(tmp_path, {})
    task = _DummyImageQATask(TaskConfig(name="dummy", image_mode="caption", caption_source=source))

    with pytest.raises(ValueError, match="is empty"):
        task.format_request(_instance())


# ---------------------------------------------------------------------------
# MMMU-Pro: the vision setting cannot be run imageless
# ---------------------------------------------------------------------------


def test_mmmu_pro_vision_setting_rejects_imageless_modes():
    # In the vision setting the question is rendered into the screenshot, so
    # instance.question is only the answer-format instruction. Dropping the image leaves
    # nothing to answer -- and the official parser falls back to a seeded random choice,
    # so it would report ~25% rather than failing.
    from olmo_eval.evals.tasks.mmmu_pro import MmmuProTask

    task = MmmuProTask(TaskConfig(name="mmmu_pro", image_mode="none"))
    instance = Instance(
        question="Answer with the option letter from the given choices directly.",
        metadata={
            "mmmu_pro_setting": "vision",
            "example_id": "v1",
            "image": _tiny_image(),
        },
    )

    with pytest.raises(ValueError, match="vision setting"):
        task.format_request(instance)


def test_mmmu_pro_standard_setting_drops_all_interleaved_images():
    from olmo_eval.evals.tasks.mmmu_pro import MmmuProTask

    task = MmmuProTask(TaskConfig(name="mmmu_pro", image_mode="none"))
    instance = Instance(
        question="Which is larger?\nA. x\nB. y",
        metadata={
            "mmmu_pro_setting": "standard10",
            "example_id": "s1",
            "images": [_tiny_image(), _tiny_image(), _tiny_image()],
        },
    )

    request = task.format_request(instance)

    assert request.images is None
    assert request.messages[0]["content"] == "Which is larger?\nA. x\nB. y"


def test_mmmu_pro_standard_setting_keeps_images_by_default():
    from olmo_eval.evals.tasks.mmmu_pro import MmmuProTask

    task = MmmuProTask(TaskConfig(name="mmmu_pro"))
    instance = Instance(
        question="Which is larger?",
        metadata={
            "mmmu_pro_setting": "standard10",
            "example_id": "s1",
            "images": [_tiny_image(), _tiny_image()],
        },
    )

    request = task.format_request(instance)

    assert request.images is not None
    assert len(request.images) == 2


# ---------------------------------------------------------------------------
# Variant registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("mmmu", "real"),
        ("mmmu:text_only", "none"),
        ("mmmu:oracle_caption", "caption"),
        ("charxiv_descriptive:text_only", "none"),
        ("charxiv_descriptive:oracle_caption", "caption"),
        ("charxiv_reasoning:text_only", "none"),
        ("charxiv_reasoning:oracle_caption", "caption"),
    ],
)
def test_registered_variants_resolve_to_expected_image_mode(spec: str, expected: str):
    assert get_task(spec).config.image_mode == expected

"""Tests for the ScreenSpot GUI grounding tasks and their point-in-box scoring.

Everything here is offline: records are synthetic and no image is ever opened.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.vision.benchmarks.screenspot import (
    DATA_TYPE_KEY,
    PLATFORM_KEY,
    PLATFORMS,
    SCREENSPOT_REVISION,
    SCREENSPOT_V2_REVISION,
    SOURCE_TO_PLATFORM,
    GroupAccuracyMetric,
)
from olmo_eval.evals.vision.scoring.grounding import (
    BBOX_KEY,
    PointInBoxScorer,
    parse_point,
    point_in_box,
)
from olmo_eval.evals.vision.scoring.prompts import POINTING_TEMPLATES
from olmo_eval.runners.processing.utils import compute_task_hash

# ---------------------------------------------------------------------------
# Point parsing
# ---------------------------------------------------------------------------


class TestParsePoint:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Molmo2: "<frame> <idx> <x> <y>", scaled by 1000 (verbatim Molmo2-4B outputs)
            ('<points coords="1 1 208 108">check the weather</points>', (20.8, 10.8)),
            (
                '<points coords="1 1 362 467">set the second photo as wallpaper</points>',
                (36.2, 46.7),
            ),
            ('<points coords="1 1 894 106">check my profile</points>', (89.4, 10.6)),
            # several points in one frame: the first is the answer
            ('<points coords="1 1 050 100 2 900 900">tabs</points>', (5.0, 10.0)),
            # several frames, separated as the model card decodes them
            ('<points coords="1 1 050 100; 2 1 900 900">tabs</points>', (5.0, 10.0)),
            ('<tracks coords="1 1 250 750"/>', (25.0, 75.0)),
            # Molmo (v1): x/y attributes in percent
            ('<point x="12.3" y="45.6" alt="close">close</point>', (12.3, 45.6)),
            ('<point x="7" y="93">x</point>', (7.0, 93.0)),
            # <points> markup: the first point is the answer
            (
                '<points x1="10.0" y1="20.0" x2="30.0" y2="40.0" alt="tabs">tabs</points>',
                (10.0, 20.0),
            ),
            ('Sure. <point x=" 50.5 " y=" 60.5 " alt="a">a</point> done', (50.5, 60.5)),
            ('{"name": "click", "x": 70.2, "y": 3.4, "button": "left"}', (70.2, 3.4)),
            ('{"action": {"name": "click", "x": 70.2, "y": 3.4}, "thought": "t"}', (70.2, 3.4)),
            ('{"x": "12", "y": "34"}', (12.0, 34.0)),
        ],
    )
    def test_recognized_forms(self, text, expected):
        assert parse_point(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "There are none.",
            "The button is in the top right corner.",
            '{"name": "click"}',
            '{"x": 12}',
            '<point x="1.0" alt="no y">a</point>',
            # a mismatched index pair is not a point
            '<points x1="1.0" y2="2.0">a</points>',
            # Molmo2 markup without a complete point
            '<points coords="1 1 208">a</points>',
            '<points coords="">a</points>',
        ],
    )
    def test_no_point(self, text):
        assert parse_point(text) is None

    def test_markup_wins_over_json(self):
        text = '{"x": 1, "y": 2} <point x="3" y="4">a</point>'
        assert parse_point(text) == (3.0, 4.0)

    def test_out_of_range_molmo2_point_is_skipped(self):
        # x=1200 is off the image, so the next point is the answer
        assert parse_point('<points coords="1 1 1200 100 2 300 400">a</points>') == (30.0, 40.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _instance(bbox=None, **metadata) -> Instance:
    if bbox is not None:
        metadata[BBOX_KEY] = bbox
    return Instance(question="q", gold_answer=None, metadata=metadata)


class TestPointInBox:
    BOX = [10.0, 20.0, 30.0, 40.0]

    @pytest.mark.parametrize("point", [(20.0, 30.0), (10.0, 20.0), (30.0, 40.0), (10.0, 40.0)])
    def test_inside_and_on_edges(self, point):
        assert point_in_box(point, self.BOX)

    @pytest.mark.parametrize("point", [(9.9, 30.0), (30.1, 30.0), (20.0, 19.9), (20.0, 40.1)])
    def test_outside(self, point):
        assert not point_in_box(point, self.BOX)

    def test_scorer_reads_extracted_answer(self):
        scorer = PointInBoxScorer()
        inside = LMOutput(text="", extracted_answer=(15.0, 25.0))
        outside = LMOutput(text="", extracted_answer=(50.0, 25.0))
        assert scorer.score(_instance(self.BOX), inside) == 1.0
        assert scorer.score(_instance(self.BOX), outside) == 0.0

    def test_scorer_zero_without_point_or_box(self):
        scorer = PointInBoxScorer()
        # the raw text is never parsed here; extraction is the task's job
        no_point = LMOutput(text='<point x="15" y="25">a</point>', extracted_answer=None)
        assert scorer.score(_instance(self.BOX), no_point) == 0.0
        assert scorer.score(_instance(), LMOutput(text="", extracted_answer=(15.0, 25.0))) == 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _response(score: float, **metadata) -> Response:
    return Response(
        instance=_instance(**metadata),
        request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
        outputs=[LMOutput(text="")],
        scores={PointInBoxScorer().name: score},
    )


class TestGroupAccuracyMetric:
    def test_averages_only_the_group(self):
        responses = [
            _response(1.0, data_type="text"),
            _response(0.0, data_type="text"),
            _response(0.0, data_type="icon"),
        ]
        text = GroupAccuracyMetric(name="text_accuracy", group_key=DATA_TYPE_KEY, group="text")
        icon = GroupAccuracyMetric(name="icon_accuracy", group_key=DATA_TYPE_KEY, group="icon")
        assert text.compute(responses) == pytest.approx(0.5)
        assert icon.compute(responses) == pytest.approx(0.0)

    def test_empty_group_reports_zero(self, caplog):
        metric = GroupAccuracyMetric(name="web_accuracy", group_key=PLATFORM_KEY, group="web")
        with caplog.at_level("WARNING"):
            assert metric.compute([_response(1.0, platform="mobile")]) == 0.0
        assert "web_accuracy" in caplog.text

    def test_per_instance_value_only_inside_group(self):
        metric = GroupAccuracyMetric(name="icon_accuracy", group_key=DATA_TYPE_KEY, group="icon")
        assert metric.compute_instance(_response(1.0, data_type="icon")) == 1.0
        assert metric.compute_instance(_response(1.0, data_type="text")) is None
        assert metric.supports_pairwise_scorer_fallback() is False

    def test_serialization_distinguishes_groups(self):
        text = GroupAccuracyMetric(name="text_accuracy", group_key=DATA_TYPE_KEY, group="text")
        icon = GroupAccuracyMetric(name="icon_accuracy", group_key=DATA_TYPE_KEY, group="icon")
        assert text.to_dict() != icon.to_dict()
        assert text.to_dict()["group"] == "text"
        assert text.to_dict()["group_key"] == DATA_TYPE_KEY
        assert text.pairwise_display_format() == "percentage"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

V1_RECORD = {
    "file_name": "pc_ede36f9b.png",
    "bbox": [0.9479, 0.1444, 0.99375, 0.2074],
    "instruction": "close",
    "data_type": "icon",
    "data_source": "windows",
}

V2_RECORD = {
    "img_filename": "web_213f816e.png",
    "bbox": [2321, 129, 2529, 199],
    "instruction": "click the button to create a new project",
    "application": "gitlab",
    "id": "web_0",
    "action": 'Click the "New project" button.',
    "ui_type": "text",
    "platform": "web",
    "img_size": [2560, 1440],
}


class TestRegistration:
    @pytest.mark.parametrize("name", ["screenspot", "screenspot_v2"])
    def test_registered_with_greedy_short_generation(self, name):
        assert name in list_tasks()
        task = get_task(name)
        assert task.config.sampling_params.temperature == 0.0
        assert task.config.sampling_params.max_tokens == 256

    @pytest.mark.parametrize("name", ["screenspot", "screenspot_v2"])
    def test_reports_overall_type_and_platform_accuracy(self, name):
        task = get_task(name)
        assert [m.name for m in task.config.metrics] == [
            "accuracy",
            "text_accuracy",
            "icon_accuracy",
            "mobile_accuracy",
            "desktop_accuracy",
            "web_accuracy",
        ]
        assert all(m.scorer().name == "point_in_box" for m in task.config.metrics)
        assert task.config.primary_metric.name == "accuracy"

    def test_dataset_revisions_are_pinned(self):
        v1 = get_task("screenspot").config.to_dict()["data_source"]
        v2 = get_task("screenspot_v2").config.to_dict()["data_source"]
        assert v1["revision"] == SCREENSPOT_REVISION
        assert v2["revision"] == SCREENSPOT_V2_REVISION

    def test_every_source_maps_to_a_platform(self):
        assert set(SOURCE_TO_PLATFORM.values()) == set(PLATFORMS)


class TestProcessRecord:
    def test_v1_scales_normalized_box_to_percent(self):
        instance = get_task("screenspot").process_record(V1_RECORD, 0, "lazy-image")
        assert instance.metadata[BBOX_KEY] == [94.8, 14.4, 99.4, 20.7]
        assert instance.gold_answer == "[94.8, 14.4, 99.4, 20.7]"
        assert instance.metadata[DATA_TYPE_KEY] == "icon"
        assert instance.metadata["data_source"] == "windows"
        assert instance.metadata[PLATFORM_KEY] == "desktop"
        assert instance.metadata["id"] == "pc_ede36f9b.png"
        assert instance.metadata["instruction"] == "close"
        assert instance.metadata["image"] == "lazy-image"

    def test_v2_scales_pixel_box_by_image_size(self):
        instance = get_task("screenspot_v2").process_record(V2_RECORD, 3, "/data/web_213f816e.png")
        assert instance.metadata[BBOX_KEY] == [90.7, 9.0, 98.8, 13.8]
        assert instance.metadata[DATA_TYPE_KEY] == "text"
        assert instance.metadata["data_source"] == "gitlab"
        assert instance.metadata[PLATFORM_KEY] == "web"
        assert instance.metadata["id"] == "web_0"
        assert instance.metadata["index"] == 3
        assert instance.metadata["image_path"] == "/data/web_213f816e.png"
        # only the original instruction is used, never the rephrased variants
        assert instance.metadata["instruction"] == V2_RECORD["instruction"]

    def test_gold_point_scores_one_end_to_end(self):
        task = get_task("screenspot_v2")
        instance = task.process_record(V2_RECORD, 0, "/data/img.png")
        output = LMOutput(text='<points coords="1 1 947 114">New project</points>')
        output.extracted_answer = task.extract_answer(output)
        assert output.extracted_answer == (94.7, 11.4)
        assert PointInBoxScorer().score(instance, output) == 1.0


class TestPrompt:
    def test_default_uses_a_pointing_template(self):
        task = get_task("screenspot_v2")
        question = task.process_record(V2_RECORD, 0, "/data/img.png").question
        label = V2_RECORD["instruction"]
        assert label in question or label.lower() in question
        assert any(
            question == template.replace("{label}", text, 1)
            for template in POINTING_TEMPLATES
            for text in (label, label.lower())
        )

    def test_template_choice_is_seeded_by_index(self):
        task = get_task("screenspot_v2")
        first = task.process_record(V2_RECORD, 0, "/data/img.png").question
        again = task.process_record(V2_RECORD, 0, "/data/img.png").question
        assert first == again
        others = {task.process_record(V2_RECORD, i, "/data/img.png").question for i in range(20)}
        assert len(others) > 1

    def test_pretrain_family_gives_prefixed_bare_instruction(self):
        task = get_task("screenspot")
        task.config = replace(
            task.config, prompt_templates="none", system_prompt_style="style_and_length_v2"
        )
        instance = task.process_record({**V1_RECORD, "instruction": "Close Window"}, 5, "img")
        assert instance.question == "pointing: close window"

    def test_unknown_prompt_family_raises(self):
        task = get_task("screenspot")
        task.config = replace(task.config, prompt_templates="uber_modle_v2")
        with pytest.raises(ValueError, match="prompt_templates"):
            task.process_record(V1_RECORD, 0, "img")

    def test_prompt_family_changes_task_hash(self):
        task = get_task("screenspot")
        base = compute_task_hash(task.config.to_dict())
        styled = replace(task.config, prompt_templates="none")
        assert compute_task_hash(styled.to_dict()) != base


class TestRequests:
    def test_v2_request_carries_the_path_unopened(self):
        task = get_task("screenspot_v2")
        instance = task.process_record(V2_RECORD, 0, "/nonexistent/img.png")
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert request.images == ("/nonexistent/img.png",)
        assert request.messages == ({"role": "user", "content": instance.question},)

    def test_v1_request_carries_the_lazy_image(self):
        task = get_task("screenspot")
        sentinel = object()
        instance = task.process_record(V1_RECORD, 0, sentinel)
        assert task.format_request(instance).images == (sentinel,)

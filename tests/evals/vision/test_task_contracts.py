"""Cross-cutting contracts for the vision tasks: per-instance metric storage,
task-identity hashing of the prompt-family fields, config validation, and the
lazy image path through requests."""

from __future__ import annotations

from dataclasses import replace

import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.images import resolve_images
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks.dense_caption import _DEFAULT_METRICS, DenseCaptionAvgMetric
from olmo_eval.runners.asynq.processing import _result_request
from olmo_eval.runners.processing.utils import compute_task_hash


def _metric(name: str):
    return next(m for m in _DEFAULT_METRICS if m.name == name)


def _response(result: dict | None, scores: dict | None = None) -> Response:
    instance = Instance(question="q", gold_answer=None, metadata={})
    output = LMOutput(text="caption")
    if result is not None:
        output.metadata = {"dense_caption_result": result}
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, prompt="q"),
        outputs=[output],
        scores=scores or {},
    )


class TestDenseCaptionPerInstanceMetrics:
    """Persisted per-instance values must be each metric's own field, never the
    scorer channel (which is recall)."""

    RESULT = {
        "recall": 0.4,
        "recall_at_10": 0.5,
        "num_statements": 10,
        "num_covered": 4,
        "recall_valid": True,
        "consistency": 0.75,
        "num_consistent": 3,
        "consistency_num_statements": 4,
        "consistency_valid": True,
    }

    def test_each_metric_stores_its_own_field(self):
        # poison the scorer channel: nothing may fall back to it
        response = _response(self.RESULT, scores={"dense_caption_judge": 0.123})
        assert _metric("recall").compute_instance(response) == pytest.approx(40.0)
        assert _metric("consistency").compute_instance(response) == pytest.approx(75.0)
        assert _metric("recall_at_10").compute_instance(response) == pytest.approx(50.0)
        assert _metric("num_statements").compute_instance(response) == pytest.approx(4.0)

    def test_invalid_side_stores_none(self):
        result = dict(self.RESULT, recall_valid=False)
        response = _response(result, scores={"dense_caption_judge": 0.123})
        assert _metric("recall").compute_instance(response) is None
        assert _metric("recall_at_10").compute_instance(response) is None
        assert _metric("consistency").compute_instance(response) == pytest.approx(75.0)

    def test_avg_has_no_per_instance_value(self):
        response = _response(self.RESULT, scores={"dense_caption_judge": 0.123})
        assert DenseCaptionAvgMetric().compute_instance(response) is None
        assert DenseCaptionAvgMetric().supports_pairwise_scorer_fallback() is False

    def test_scorer_fallback_disabled(self):
        # no judge result at all: the poisoned scorer channel must NOT leak through
        response = _response(None, scores={"dense_caption_judge": 0.123})
        for name in ("recall", "consistency", "recall_at_10", "num_statements"):
            assert _metric(name).compute_instance(response) is None

    def test_aggregate_matches_mean_of_instances(self):
        responses = [
            _response(self.RESULT),
            _response(dict(self.RESULT, recall=0.6, recall_valid=True)),
            _response(dict(self.RESULT, recall_valid=False)),
        ]
        vals = [_metric("recall").compute_instance(r) for r in responses]
        expected = sum(v for v in vals if v is not None) / 2
        assert _metric("recall").compute(responses) == pytest.approx(expected)

    def test_legacy_scale_displays_as_raw(self):
        # 0-100 legacy values must not be re-scaled by the percentage renderer
        assert _metric("recall").pairwise_display_format() == "raw"
        assert DenseCaptionAvgMetric().pairwise_display_format() == "raw"


class TestPromptFamilyTaskIdentity:
    """The prompt-family fields change the request, so they must change the task hash."""

    def test_system_prompt_style_changes_hash(self):
        task = get_task("dense_caption")
        base = compute_task_hash(task.config.to_dict())
        styled = replace(task.config, system_prompt_style="style_and_length_v2")
        assert compute_task_hash(styled.to_dict()) != base

    def test_prompt_templates_changes_hash(self):
        task = get_task("dense_caption")
        base = compute_task_hash(task.config.to_dict())
        templated = replace(task.config, prompt_templates="none")
        assert compute_task_hash(templated.to_dict()) != base


class TestDenseCaptionConfigValidation:
    def test_unknown_system_prompt_style_raises(self):
        task = get_task("dense_caption")
        task.config = replace(task.config, system_prompt_style="style_and_lenght_v2")
        with pytest.raises(ValueError, match="Unsupported system_prompt_style"):
            task._question(0)

    def test_multi_sample_runs_rejected(self):
        task = get_task("dense_caption")
        params = replace(task.config.sampling_params, num_samples=2)
        task.config = replace(task.config, sampling_params=params)
        with pytest.raises(ValueError, match="num_samples"):
            list(task.instances)


class TestLazyImageRequests:
    """Requests carry lazy image references; providers resolve them."""

    def test_format_request_attaches_path_not_pixels(self):
        task = get_task("dense_caption")
        instance = Instance(
            question="q", gold_answer=None, metadata={"image_path": "/nonexistent/img.png"}
        )
        request = task.format_request(instance)
        # a path string — building the request must not open the file
        assert request.images == ("/nonexistent/img.png",)

    def test_resolve_images_opens_paths_and_calls_callables(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")
        path = tmp_path / "img.png"
        PIL_Image.new("RGB", (3, 2)).save(path)
        resolved = resolve_images((str(path), lambda: PIL_Image.new("RGB", (5, 4))))
        assert resolved is not None
        assert resolved[0].size == (3, 2)
        assert resolved[1].size == (5, 4)
        assert resolve_images(None) is None

    def test_result_request_strips_image_payload(self):
        request = LMRequest(request_type=RequestType.CHAT, prompt="q", images=("x.png",))
        stripped = _result_request(request)
        assert stripped is not None
        assert stripped.images is None
        assert stripped.prompt == "q"
        # no-image requests pass through untouched
        bare = LMRequest(request_type=RequestType.CHAT, prompt="q")
        assert _result_request(bare) is bare


class TestLazyMultiImageRequests:
    def test_list_entries_flatten_and_cap(self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")
        from olmo_eval.evals.vision.data.images import capped_image_list

        paths = []
        for i in range(3):
            path = tmp_path / f"{i}.png"
            PIL_Image.new("RGB", (2 + i, 2)).save(path)
            paths.append(str(path))
        entry = capped_image_list(paths, max_images=2)
        resolved = resolve_images((entry,))
        assert resolved is not None
        assert [img.size for img in resolved] == [(2, 2), (3, 2)]

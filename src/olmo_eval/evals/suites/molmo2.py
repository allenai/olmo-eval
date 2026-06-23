"""Molmo2 image-QA + PixMo-Cap suites (run with ``-m molmo2-4b``)."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

# The 11 single-image QA / counting benchmarks Molmo2 reports.
MOLMO2_IMAGE_QA_TASKS = (
    "chart_qa",
    "vqa2",
    "doc_qa",
    "info_qa",
    "text_vqa",
    "real_world_qa",
    "mmmu",
    "math_vista",
    "countbench_qa",
    "pixmo_count",
    "ai2d",
)

make_suite(
    "molmo2_imageqa",
    MOLMO2_IMAGE_QA_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's 11 image-QA benchmarks (primary metrics are all 0-1).",
)

make_suite(
    "molmo2_imageqa_caption",
    (*MOLMO2_IMAGE_QA_TASKS, "dense_caption"),
    # dense_caption's primary metric is 0-100, so no cross-task average is computed.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Molmo2's 11 image-QA benchmarks plus PixMo-Cap dense caption (GPT judge).",
)

# Image-pointing benchmarks — a separate family from the image-QA tasks above
# (point-in-mask precision/recall/f1, not VQA-style answers).
MOLMO2_POINTING_TASKS = (
    "pixmo_points_eval",
    "sa_co_gold_subset",
)

make_suite(
    "molmo2_pointing",
    MOLMO2_POINTING_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's image-pointing benchmarks (primary metric is f1, 0-1).",
)

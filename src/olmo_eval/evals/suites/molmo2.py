"""Molmo2 multimodal benchmark suites.

Grows with the vision families as they land: captioning now, pointing/counting,
image QA, and multi-image next.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

make_suite(
    "molmo2_imageqa_caption",
    ("dense_caption",),
    # dense_caption's primary metric is 0-100, so no cross-task average is computed.
    # The image-QA tasks join this suite when their family lands.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="Molmo2's PixMo-Cap dense caption (GPT judge); image-QA tasks forthcoming.",
)

# Image-pointing benchmarks — point-in-mask precision/recall/f1, not VQA-style answers.
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

# The mm_olmo `_mp` variants: same scoring, but the prompt is built from a bare label by
# the model's own formatter, so it follows the checkpoint (see `vision.scoring.prompts`).
MOLMO2_POINTING_MP_TASKS = (
    "pixmo_points_eval_mp",
    "sa_co_gold_subset_mp",
    "sa_co_gold_point_4k_mp",
)

# `sa_co_gold_point_mp` (the unsampled 166,766-example gold set) is registered but kept out
# of the suite: it is ~6x the 4k variant and measures the same thing.

make_suite(
    "molmo2_pointing_mp",
    MOLMO2_POINTING_MP_TASKS,
    aggregation=AggregationStrategy.AVERAGE,
    description="Molmo2's image-pointing benchmarks with mm_olmo's model-prompt (_mp) inputs.",
)

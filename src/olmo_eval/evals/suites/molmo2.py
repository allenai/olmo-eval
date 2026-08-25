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

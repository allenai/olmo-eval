"""Document-OCR suites: page images in, markdown (or JSON) out."""

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite

OCR_TASKS = (
    "olmocr_bench",
    "cc_ocr_multi_scene",
    "omnidocbench",
)

make_suite(
    "ocr",
    OCR_TASKS,
    # olmocr_bench and cc_ocr_multi_scene report 0-1 while omnidocbench's Overall is 0-100, so no
    # cross-task average is computed.
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description=(
        "Document OCR and parsing benchmarks (olmOCR-bench, CC-OCR multi-scene, OmniDocBench v1.6)."
    ),
)


OCR_EN_TASKS = (
    "olmocr_bench",
    "cc_ocr_multi_scene_en",
    "omnidocbench_en",
)

make_suite(
    "ocr_en",
    OCR_EN_TASKS,
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description=(
        "English-only document OCR: olmOCR-bench, CC-OCR's 8 English multi-scene sub-datasets and "
        "OmniDocBench v1.6's English pages."
    ),
)

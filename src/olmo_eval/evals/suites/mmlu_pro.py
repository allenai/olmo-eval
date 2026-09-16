from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.mmlu_pro import MMLU_PRO_CATEGORIES, category_slug

_SLUGS = tuple(category_slug(category) for category in MMLU_PRO_CATEGORIES)

MMLU_PRO = make_suite(
    "mmlu_pro",
    tuple(f"mmlu_pro_{s}" for s in _SLUGS),
    aggregation=AggregationStrategy.AVERAGE,
    description="5-shot multiple-choice MMLU-Pro, averaged over the 14 categories",
)

MMLU_PRO_BPB = make_suite(
    "mmlu_pro:bpb",
    tuple(f"mmlu_pro_{s}:rc:bpb" for s in _SLUGS),
    aggregation=AggregationStrategy.AVERAGE,
    description="5-shot cloze MMLU-Pro bits per byte, averaged over the 14 categories",
)

MMLU_PRO_COT = make_suite(
    "mmlu_pro:cot",
    tuple(f"mmlu_pro_{s}:cot" for s in _SLUGS),
    aggregation=AggregationStrategy.AVERAGE,
    description="0-shot chain-of-thought MMLU-Pro, averaged over the 14 categories",
)

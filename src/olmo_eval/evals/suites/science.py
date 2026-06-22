"""Science evaluation suites.

This module keeps the existing GPQA convenience suites while also defining a
non-overlapping science hierarchy for aggregate reporting.

Design rules for the ``science:*`` hierarchy:
1. ``science:all`` should contain each underlying task spec exactly once.
2. Reuse existing suites where they are already coherent and non-overlapping.
3. Prefer subject-sliced GPQA tasks inside the hierarchy so biology, chemistry,
   and physics live in separate domain suites without also including the
   full-subset GPQA tasks.
4. Provide an execution-oriented split between judge-free and judge-dependent
   tasks so large science runs can be staged in two passes.

The legacy ``gpqa`` / ``gpqa:mc`` / ``gpqa:bpb`` suites are retained as
convenience entry points, but they are intentionally not nested under
``science:all`` because they would duplicate the GPQA questions already
allocated to domain-specific suites.

Execution guidance:
- Use ``science:nojudge`` for the main science stack when you want to avoid
  external LLM-as-judge dependencies.
- Use ``science:judge`` for the judge-dependent science tasks.
- Use ``science:all`` only when you want both together as a single umbrella.
"""

import olmo_eval.evals.suites.astabench  # noqa: F401 - ensure suite registration
import olmo_eval.evals.suites.biology  # noqa: F401 - ensure suite registration
import olmo_eval.evals.suites.math  # noqa: F401 - ensure suite registration
import olmo_eval.evals.suites.mmlu  # noqa: F401 - ensure suite registration
from olmo_eval.evals.suites.registry import AggregationStrategy, get_suite, make_suite

# =============================================================================
# GPQA Suite
# =============================================================================

_GPQA_TASKS = (
    "gpqa_diamond",
    "gpqa_main",
    "gpqa_extended",
)

_GPQA_BIOLOGY_TASKS = tuple(f"{t}_biology" for t in _GPQA_TASKS)
_GPQA_CHEMISTRY_TASKS = tuple(f"{t}_chemistry" for t in _GPQA_TASKS)
_GPQA_PHYSICS_TASKS = tuple(f"{t}_physics" for t in _GPQA_TASKS)

_MMLU_MEDICINE_TASKS = (
    "mmlu_anatomy",
    "mmlu_clinical_knowledge",
    "mmlu_college_medicine",
    "mmlu_human_aging",
    "mmlu_medical_genetics",
    "mmlu_nutrition",
    "mmlu_professional_medicine",
    "mmlu_virology",
)

GPQA = make_suite(
    "gpqa",
    _GPQA_TASKS,
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA graduate-level science Q&A (diamond/main/extended)",
)

GPQA_MC = make_suite(
    "gpqa:mc",
    tuple(f"{t}:mc" for t in _GPQA_TASKS),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA with logprob-based MC scoring",
)

GPQA_BPB = make_suite(
    "gpqa:bpb",
    tuple(f"{t}:bpb" for t in _GPQA_TASKS),
    aggregation=AggregationStrategy.DISPLAY_ONLY,
    description="GPQA with bits-per-byte evaluation",
)

# =============================================================================
# Non-overlapping science hierarchy
# =============================================================================
#
# Notes on composition:
# - science:core is the broad STEM / school-science layer.
# - science:biology owns the biology slice of GPQA plus the dedicated biology
#   benchmarks (LAB-Bench + GeneTuring via the biology suite).
# - science:physical owns only chemistry + physics GPQA slices, avoiding
#   duplication with science:core's broader STEM exams.
# - science:medicine uses med benchmarks plus medicine-heavy MMLU subjects, but
#   does not include ``medqa`` because it points at the same benchmark family as
#   ``medqa_en`` and would double-weight that content.
# - science:research groups scientific literature / evidence-use tasks.
# - science:nojudge / science:judge are execution helpers for running the full
#   science stack in two stages.
# - science:math groups mathematical reasoning tasks used in science-adjacent
#   evaluation.

SCIENCE_CORE = make_suite(
    "science:core",
    (
        "arc_easy",
        "arc_challenge",
        "sciq",
        get_suite("mmlu:stem"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Broad STEM knowledge and science exam QA.",
)

SCIENCE_BIOLOGY = make_suite(
    "science:biology",
    (
        get_suite("biology"),
        *_GPQA_BIOLOGY_TASKS,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Biology, genomics, and wet-lab science evaluation, including GPQA biology.",
)

SCIENCE_MEDICINE = make_suite(
    "science:medicine",
    (
        "medmcqa",
        "medqa_en",
        *_MMLU_MEDICINE_TASKS,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description=(
        "Medical QA and medicine-focused knowledge tasks without duplicate MedQA weighting."
    ),
)

SCIENCE_PHYSICAL = make_suite(
    "science:physical",
    (
        *_GPQA_CHEMISTRY_TASKS,
        *_GPQA_PHYSICS_TASKS,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Chemistry and physics tasks without duplicating broad STEM core coverage.",
)

SCIENCE_RESEARCH = make_suite(
    "science:research",
    (
        "qasper_yesno",
        "sciriff_yesno",
        "expertqa",
        "litsearch",
        "litsearch_rerank",
        get_suite("astabench"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Scientific literature understanding, evidence use, and scholarly synthesis.",
)

# What the members of science:research actually measure (read the metric tiers,
# not a single aggregate):
# - expertqa: attribution + on-topic precision of cited long-form answers
#   (citation precision/recall + answer precision). NOT factual correctness vs a
#   reference; a high score means well-cited, on-topic prose, not verified truth.
# - litsearch: an AGENTIC retrieval smoke test (does a gold paper surface in live
#   Semantic Scholar results), not the published fixed-corpus Recall@k.
# - litsearch_rerank: fixed-corpus reranking. The model reranks a frozen pool of
#   BM25-retrieved candidates per query; scored Recall@5/@20 over the model's own
#   selection. Reproducible, judge-free, tool-free (the opposite trade-offs to
#   agentic litsearch). The BM25 retriever Recall@k baseline is reported by the
#   offline build script, not this task.
#
# Note: litsearch (agentic) is intentionally only in science:research, not
# science:judge / science:nojudge / science:all. It needs an agentic
# tool-providing harness (semantic_scholar_snippet_search) rather than a judge,
# so it does not fit the judge/nojudge execution split and would score zero in a
# routine science:all run. litsearch_rerank has no such dependency and so does
# sit in science:nojudge (and thus science:all).

SCIENCE_MATH = make_suite(
    "science:math",
    (
        "gsm8k",
        "gsm_symbolic",
        get_suite("minerva_math"),
        "math500",
        "aime_2024",
        "aime_2025",
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Mathematical reasoning for science-oriented model evaluation.",
)

SCIENCE_NOJUDGE = make_suite(
    "science:nojudge",
    (
        SCIENCE_CORE,
        SCIENCE_BIOLOGY,
        SCIENCE_MEDICINE,
        SCIENCE_PHYSICAL,
        "qasper_yesno",
        "sciriff_yesno",
        "litsearch_rerank",
        SCIENCE_MATH,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="All current science tasks that do not require external LLM judges.",
)

SCIENCE_JUDGE = make_suite(
    "science:judge",
    (
        "expertqa",
        get_suite("astabench"),
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Current science tasks that require external LLM-as-judge scoring.",
)

SCIENCE_ALL = make_suite(
    "science:all",
    (
        SCIENCE_NOJUDGE,
        SCIENCE_JUDGE,
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    description="Non-overlapping umbrella suite covering all current science tasks exactly once.",
)

"""Berkeley Function Calling Leaderboard suites.

The suites reproduce the leaderboard's own summaries for the single-turn
categories. Two kinds of averaging are involved, and BFCL uses them in
different places:

- Its non-live summaries average the categories as equals, which the suites
  below express with ``AVERAGE`` and ``AVERAGE_OF_AVERAGES``.
- Its live summaries weight each category by how many instances it holds. A
  suite cannot express that, so the weighted summaries are tasks that pool
  their categories' instances — ``bfcl_live_ast`` and ``bfcl_live`` — which
  gives the same number.

The executable and REST categories are not implemented, so ``bfcl:non_live``
here is BFCL's non-live overall with those terms left out, and ``bfcl`` is its
single-turn overall on the same footing. Both differ from a published non-live
number, which averages the executable categories in.
"""

from olmo_eval.evals.suites.registry import AggregationStrategy, Suite, register

NON_LIVE_SIMPLE = register(
    Suite(
        name="bfcl:non_live_simple",
        tasks=("bfcl_simple", "bfcl_java", "bfcl_javascript"),
        aggregation=AggregationStrategy.AVERAGE,
        description="BFCL non-live simple AST, across languages",
    )
)

_AST_TERMS: tuple[str | Suite, ...] = (
    NON_LIVE_SIMPLE,
    "bfcl_multiple",
    "bfcl_parallel",
    "bfcl_parallel_multiple",
)

register(
    Suite(
        name="bfcl:non_live_ast",
        tasks=_AST_TERMS,
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="BFCL non-live AST summary",
    )
)

NON_LIVE = register(
    Suite(
        name="bfcl:non_live",
        tasks=(*_AST_TERMS, "bfcl_irrelevance"),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="BFCL non-live AST summary with irrelevance",
    )
)

register(
    Suite(
        name="bfcl",
        tasks=(NON_LIVE, "bfcl_live"),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="BFCL v3 single-turn overall",
    )
)

register(
    Suite(
        name="bfcl:categories",
        tasks=tuple(
            f"bfcl_{category}"
            for category in (
                "simple",
                "multiple",
                "parallel",
                "parallel_multiple",
                "java",
                "javascript",
                "irrelevance",
                "live_simple",
                "live_multiple",
                "live_parallel",
                "live_parallel_multiple",
                "live_irrelevance",
                "live_relevance",
            )
        ),
        aggregation=AggregationStrategy.DISPLAY_ONLY,
        description="Every BFCL v3 single-turn category, reported separately",
    )
)

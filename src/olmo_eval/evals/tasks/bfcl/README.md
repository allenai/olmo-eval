# BFCL (Berkeley Function Calling Leaderboard)

The BFCL v3 single-turn categories, reformulated as a completion task so a
pretrained model with no chat template can be measured on them.

A prediction is graded by BFCL's own checker: the predicted calls are compared
against a list of accepted values per parameter, so an answer is right when it
names the right function and supplies values the benchmark accepts — not when
it matches one reference string.

Paper: <https://arxiv.org/abs/2502.17858> ·
Dataset: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

## Running it

Each instance is one block of text — the instruction, the function documents,
the question — that the model continues with its calls.

```bash
uv run olmo-eval run -m my-base-model -t bfcl
uv run olmo-eval run -m my-base-model -t bfcl_simple
```

## Exemplars

Hand-written exemplars carry the answer format; `:0shot` through `:5shot`
change how many are shown, and the default is five:

```bash
uv run olmo-eval run -m my-base-model -t bfcl_simple:2shot
uv run olmo-eval run -m my-base-model -t bfcl:0shot
```

They are hand-written rather than sampled from the data, so no exemplar is
drawn from the evaluated set, and they cover the shapes a category can take —
one call, one of several candidate functions, several calls, and declining
when no function fits. The declining exemplar matters: without it a base model
calls something on nearly every instance and the irrelevance categories
measure nothing.

Each language has its own set, since the Java and JavaScript categories expect
calls written in those languages.

The expected answer format is BFCL's `[func(arg=value)]` for every model, so
numbers are comparable across models and checkpoints. Because a base model has
never been taught that format, the decoder also accepts the JSON tool-call
shapes models pick up during pretraining — a bare list of
`{"name": ..., "arguments": {...}}` objects, or one wrapped in `<tool_call>`
tags — so the score reflects which function was chosen rather than which
surface form the model reached for.

## Categories and suites

```bash
uv run olmo-eval suite inspect bfcl
```

| Suite | What it reports |
|-------|-----------------|
| `bfcl` | BFCL's single-turn overall, without the executable categories |
| `bfcl:non_live_ast` | BFCL's non-live AST summary |
| `bfcl:non_live` | Non-live AST summary with irrelevance |
| `bfcl:non_live_simple` | Simple AST across Python, Java and JavaScript |
| `bfcl:categories` | Every category reported separately |

BFCL weights its live summaries by how many instances each category holds,
which a suite average cannot express, so those summaries are tasks that pool
their categories' instances: `bfcl_live_ast` and `bfcl_live`. The per-category
tasks are `bfcl_simple`, `bfcl_multiple`, `bfcl_parallel`,
`bfcl_parallel_multiple`, `bfcl_java`, `bfcl_javascript`, `bfcl_irrelevance`,
and the six `bfcl_live_*` categories.

`bfcl_java` and `bfcl_javascript` read calls written in those languages, and
declare the `tree-sitter-java` and `tree-sitter-javascript` grammars as
task-specific dependencies (see [Task-Specific
Dependencies](../../../../../README.md#task-specific-dependencies)). No other
category needs them.

## The checker

The checker is vendored under `olmo_eval/common/scorers/bfcl/` rather than
depended on: the `bfcl-eval` package pins `numpy==1.26.4` and pulls faiss,
sentence-transformers, anthropic and cohere, which does not coexist with the
vLLM environment here.

Two deliberate departures from the reference, both commented where they
happen:

- An arithmetic argument is folded rather than passed to `eval`, which would
  run whatever a model wrote.
- The v3 prompt is reproduced verbatim, including its wording slips and its
  rendering of function documents as a Python repr under a sentence calling it
  JSON, because changing either moves the scores.

Replaying BFCL's own possible answers back through the checker reproduces
999/1000 non-live Python, 1344/1351 live, 92/100 Java and 49/50 JavaScript.
Every remaining case is a dataset quirk the reference checker fails
identically, such as `simple_363`, whose possible answer names `find_closest`
while its function document names `restaurant_search.find_closest`.

## What is not implemented

- **Multi-turn** (`multi_turn_base`, `multi_turn_miss_func`,
  `multi_turn_miss_param`, `multi_turn_long_context`, `multi_turn_composite`):
  scoring them needs BFCL's stateful API backend.
- **Executable and REST** (`exec_*`, `rest`): these grade by running the
  predicted calls against live third-party APIs.

Because the executable categories are missing, `bfcl:non_live` is BFCL's
non-live overall with those terms left out and will not match a published
non-live number, which averages them in. `bfcl:non_live_ast` and the live
summaries are directly comparable.

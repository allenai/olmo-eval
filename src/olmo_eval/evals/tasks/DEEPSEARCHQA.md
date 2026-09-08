# DeepSearchQA

`deepsearchqa` implements [DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa)
(Google DeepMind, [arXiv:2601.20975](https://arxiv.org/abs/2601.20975); Kaggle
starter code: [deepsearchqa-starter-code](https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code),
[leaderboard](https://www.kaggle.com/benchmarks/google/dsqa)) inside olmo-eval.

900 hand-crafted, multi-step information-seeking questions across 17 domains
(Politics, Finance, Science, Health, History, Geography, Media, ...). Each
question is either a **Single Answer** (one entity/value) or a **Set Answer**
(an enumeration or composite answer requiring multiple items); the goal is to
test whether an agent can plan and execute a search process that returns a
*complete and precise* answer set, not just a single plausible fact.

## Files

- [`deepsearchqa.py`](deepsearchqa.py) — the task implementation (this file's
  companion module).
- [`../../../../tests/evals/tasks/test_deepsearchqa.py`](../../../../tests/evals/tasks/test_deepsearchqa.py)
  — unit tests for parsing, scoring, and the registered task/variant.

No other files were modified; the task is picked up automatically by the
package's task auto-discovery (`src/olmo_eval/evals/tasks/__init__.py`), the
same mechanism every other task in this directory relies on.

## Data

- HuggingFace dataset: `google/deepsearchqa`, config `deepsearchqa`, split
  `eval` (900 rows).
- Columns: `problem` (question), `problem_category` (domain), `answer`
  (ground truth, comma-joined for multi-item answers), `answer_type`
  (`"Single Answer"` or `"Set Answer"`).
- Both answer types are parsed the same way: the `answer` column is split on
  `,` into a gold item set (a `"Single Answer"` becomes a one-item set), so
  scoring treats every instance uniformly as set comparison.
- 4 `Set Answer` rows encode "no items satisfy every constraint" as the
  literal text `None` in the source CSV; HuggingFace's CSV loader coerces
  that to a null value. These are treated as an empty gold answer set (not
  dropped) — a correctly-empty model prediction scores full marks, and any
  predicted item scores zero. A missing `answer` on a `Single Answer` row is
  still treated as malformed and dropped, since that combination shouldn't
  occur in valid data.

## How grading works

The model is prompted to end its response with:

```
FINAL ANSWER: <item 1>, <item 2>, ...
```

Only the text after the last `FINAL ANSWER` marker is graded (the rest of the
response — search steps, reasoning, etc. — is not scored directly). That line
is split into a predicted item set `S` and compared against the gold item set
`G` with an LLM judge that decides semantic equivalence per item (so
"NZ" ≟ "New Zealand" counts as a match). From the judge's matched indices the
task computes, per instance:

| Metric                    | Formula                    |
|----------------------------|----------------------------|
| `deepsearchqa_precision`   | `\|S ∩ G\| / \|S\|`        |
| `deepsearchqa_recall`      | `\|S ∩ G\| / \|G\|`        |
| `deepsearchqa_f1`          | harmonic mean of precision/recall (**primary metric**) |
| `deepsearchqa_exact_match` | `1.0` iff the item sets match exactly, else `0.0` |

**Deviation from the official benchmark**: the paper/starter notebook grade
with a Gemini 2.5 Flash autorater using an unpublished prompt. olmo-eval's
judge infrastructure (`build_openai_judge_fn`) is OpenAI-only, so this task
uses an independently written judge prompt against an OpenAI model instead
(default `gpt-5.5:medium`, overridable — see below). The metric *definitions*
(precision/recall/F1 over item sets) match the paper; absolute numbers are
not directly comparable to the public Kaggle leaderboard.

This benchmark is about live information-seeking rather than parametric
knowledge, so it is intended to be evaluated with search tools attached via
olmo-eval's Harness abstraction — running it without tools mostly measures
how much of the answer the model already knows.

## Running it

```bash
# Requires an OpenAI key for the LLM judge
export OPENAI_API_KEY=...

# Full 900-instance run with search tools (the intended way to run this eval)
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa --harness dr_tulu

# Baseline, no search tools (parametric-knowledge reference point only)
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa

# Quick iteration on a 50-instance subset
uv run olmo-eval run -m llama3.1-8b -t deepsearchqa:mini --harness dr_tulu

# Preview instances/prompts without calling a model or the judge
uv run olmo-eval task inspect deepsearchqa -n 3 --request

# Preview the run config without executing anything
uv run olmo-eval run -m mock -t deepsearchqa --dry-run
```

Use a different judge model with `OLMO_EVAL_JUDGE`, e.g.:

```bash
OLMO_EVAL_JUDGE=gpt-5-mini uv run olmo-eval run -m llama3.1-8b -t deepsearchqa --harness dr_tulu
```

See the main [README's Harness section](../../../../README.md#harness) for
configuring which search tools `--harness` attaches, and [Adding New
Tasks](../../../../README.md#adding-new-tasks) for how this task fits the
general task framework.

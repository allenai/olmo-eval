# Serialized Tasks

A serialized task lets you add an evaluation to olmo-eval without
implementing a new Task class. You provide a JSONL file containing
pre-formatted instances and model requests; the built-in `SerializedTask`
class handles loading and plumbing. All you need to do is register a variant
that points at your data and specifies the metrics you want.

The goal is to keep this path as simple as possible for common task shapes
(completion, loglikelihood, chat). More exotic task types that need custom
runtime behavior (per-instance sampling params, dynamic prompt construction,
multi-request-per-instance patterns) may still require a full Task subclass,
but we aim to grow `SerializedTask` over time to cover the common cases.

## JSONL schema at a glance

Each line is a JSON object with two groups of fields -- one that maps to an
`Instance` (used for scoring) and one that maps to an `LMRequest` (sent to
the model):

**Loglikelihood / BPB example** (single continuation scored against the prompt):

```json
{
  "doc_id": 0,
  "question": "Write a function to find the maximum element in a list.",
  "gold_answers": ["def max_element(lst):\n    return max(lst)"],
  "choices": null,
  "metadata": {"task_name": "mbpp:3shot:bpb", "id": 601, "test": "assert max_element([1,2,3]) == 3"},
  "request_type": "loglikelihood",
  "prompt": "You are an expert Python programmer...\n\n",
  "messages": null,
  "continuations": ["def max_element(lst):\n    return max(lst)"]
}
```

**Multiple-choice example** (one continuation per choice, scored by logprob):

```json
{
  "doc_id": 0,
  "question": "What is the powerhouse of the cell?",
  "gold_answers": ["Mitochondria"],
  "choices": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
  "metadata": {"task_name": "sciq", "gold_idx": 1},
  "request_type": "loglikelihood",
  "prompt": "What is the powerhouse of the cell?\n  A. Nucleus\n  B. Mitochondria\n  C. Ribosome\n  D. Golgi apparatus\nAnswer:",
  "messages": null,
  "continuations": [" A", " B", " C", " D"]
}
```

**Chat example** (generation request with message list):

```json
{
  "doc_id": 0,
  "question": "What is the capital of France?",
  "gold_answers": ["Paris"],
  "choices": null,
  "metadata": {"task_name": "trivia_chat"},
  "request_type": "chat",
  "prompt": null,
  "messages": [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "continuations": null
}
```

**Instance-level fields** (for scoring):

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `int` | Unique numeric ID within the file |
| `question` | `str` | Raw question text (best-effort; may not be a clean question for all task types) |
| `gold_answers` | `list[str]` | Valid gold answers; first element maps to `Instance.gold_answer` |
| `choices` | `list[str] \| null` | Answer choices for MC tasks, null otherwise |
| `metadata` | `dict` | Arbitrary per-instance metadata; always includes `task_name`, may include `gold_idx` for MC, plus any scorer-specific fields |

**LMRequest-level fields** (for inference):

| Field | Type | Description |
|-------|------|-------------|
| `request_type` | `str` | `"completion"`, `"loglikelihood"`, or `"chat"` |
| `prompt` | `str \| null` | Formatted prompt string (completion / loglikelihood) |
| `messages` | `list[dict] \| null` | Chat messages list (chat requests) |
| `continuations` | `list[str] \| null` | Continuation strings for loglikelihood scoring |

### What gets serialized and what does not

The JSONL captures **what goes into the model**: the fully formatted prompt
or chat messages, including baked-in few-shot examples, chat templates, and
continuation strings. It also captures Instance fields needed by scorers
(question, gold answers, choices, metadata).

The JSONL does **not** capture:

- **SamplingParams** (max_tokens, temperature, stop_sequences, etc.) --
  these are runtime concerns configured via `TaskConfig` or CLI overrides.
- **Formatter, Scorer, and Metric definitions** -- these live in olmo-eval
  task registrations, not in the data.
- **Answer extraction logic** -- implemented by scorers in olmo-eval.
- **system_prompt** -- supported by `LMRequest` but not currently read from
  metadata by the default `SerializedTask` (easy to add by subclassing).

---

## Example 1: Migrating a task from oe-eval-internal

This walkthrough covers how the `olmo3:base_easy:code:bpb` suite was
migrated from oe-eval-internal using serialized data. Use it as a reference
if you have a task that already exists in oe-eval.

### Step 1: Serialize the data

The script
[`oe_eval/serialize_benchmark.py`](https://github.com/allenai/oe-eval-internal/blob/main/oe_eval/serialize_benchmark.py)
converts oe-eval tasks into JSONL files. It loads a task through oe-eval's
own `load_task` / `build_all_requests` pipeline, then extracts the formatted
prompt and instance fields into `SerializedRecord` objects
(see lines 43-81 of that script for the schema dataclass).

```bash
python -m oe_eval.serialize_benchmark \
    --task-suite olmo3:base_easy:code_bpb \
    --output-dir /tmp/serialized_benchmarks
```

This produces one JSONL file per leaf task plus a `manifest.json`. The
script handles collapsing multiple-choice request instances into a single
record with `choices` and `continuations` lists, and extracts metadata
fields like `test` and `entry_point` from the underlying dataset document
(see lines 221-256 for the extraction logic).

Note that this script does not necessarily cover every task type in oe-eval
out of the box. It was written to migrate the code BPB suite and serves as
an example of how to approach serialization for other tasks. Tasks with
unusual request types (e.g. `generate_until_and_loglikelihood`) or
non-standard `doc_to_target()` implementations may need adjustments -- see
the [Limitations](#limitations-and-gotchas) section below.

### Step 2: Upload the data

Upload the JSONL files to a location accessible by `DataLoader` (S3, GCS,
or a local path):

```bash
aws s3 cp /tmp/serialized_benchmarks/ s3://my-bucket/serialized/ --recursive
```

### Step 3: Register variants in olmo-eval

Each JSONL file becomes a variant of the `serialized` base task. Add
registrations in
[`src/olmo_eval/evals/tasks/serialized.py`](../src/olmo_eval/evals/tasks/serialized.py).
Here is what the code BPB migration looks like (simplified):

```python
from olmo_eval.common.metrics import BPBMetricInstanceAvg
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register_variant

_S3_BASE = "s3://ai2-llm/ianm/oe-eval-serialized/olmo3_base_easy_code_bpb"
_BPB_METRICS = (BPBMetricInstanceAvg(),)

register_variant(
    "serialized",
    "codex_humaneval_3shot_bpb",
    data_source=DataSource(path=f"{_S3_BASE}/codex_humaneval_3shot_bpb__none.jsonl"),
    metrics=_BPB_METRICS,
)

register_variant(
    "serialized",
    "mbpp_3shot_bpb",
    data_source=DataSource(path=f"{_S3_BASE}/mbpp_3shot_bpb__none.jsonl"),
    metrics=_BPB_METRICS,
    limit=500,
)
```

Each call overrides `TaskConfig` fields on the `serialized` base task.
The kwargs you can pass are any `TaskConfig` field: `data_source`,
`metrics`, `limit`, `primary_metric`, `sampling_params`, etc.

For the full set of registrations including the per-language multilingual
MBPP variants, see
[`serialized.py` lines 96-149](../src/olmo_eval/evals/tasks/serialized.py).

### Step 4: Define a suite

Group related serialized tasks into a suite in
[`src/olmo_eval/evals/suites/code.py`](../src/olmo_eval/evals/suites/code.py):

```python
OLMO3_BASE_EASY_CODE_BPB_SERIALIZED = register(
    Suite(
        name="olmo3:base_easy:code:bpb:serialized",
        tasks=(
            "serialized:codex_humaneval_3shot_bpb",
            "serialized:mbpp_3shot_bpb",
            _SERIALIZED_MT_MBPP_V2FIX_BPB,
        ),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        description="OLMo3 base_easy code BPB suite from serialized data",
    )
)
```

### Step 5: Run it

```bash
olmo-eval run -t serialized:codex_humaneval_3shot_bpb -m my-model
# or the whole suite:
olmo-eval run -s olmo3:base_easy:code:bpb:serialized -m my-model
```

---

## Example 2: Creating a new serialized task from scratch

This walkthrough covers creating a serialized task that does not come from
oe-eval -- you write a standalone script to produce the JSONL.

### Step 1: Write a serialization script

See
[`examples/serialize_task_example.py`](../examples/serialize_task_example.py)
for a complete, runnable example. It loads the SciQ dataset from
HuggingFace (a simple 4-choice science QA benchmark) and writes a JSONL
file. The key function is `serialize_doc`, which builds one record per
example:

```python
def serialize_doc(doc_id, doc):
    # ... shuffle choices, find gold_idx ...
    return {
        "doc_id": doc_id,
        "question": doc["question"],
        "gold_answers": [doc["correct_answer"]],
        "choices": choices,
        "metadata": {"task_name": "sciq_serialized", "gold_idx": gold_idx},
        "request_type": "loglikelihood",
        "prompt": make_prompt(doc, choices),
        "messages": None,
        "continuations": [f" {label}" for label in "ABCD"],
    }
```

Run it:

```bash
pip install datasets
python examples/serialize_task_example.py --output /tmp/sciq_serialized.jsonl
```

### Step 2: Register the variant

Add a registration pointing at your JSONL. For a multiple-choice task
scored by logprob:

```python
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register_variant

register_variant(
    "serialized",
    "sciq_mc",
    data_source=DataSource(path="/tmp/sciq_serialized.jsonl"),
    metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
)
```

### Step 3: Run it

```bash
olmo-eval run -t serialized:sciq_mc -m my-model
```

---

## Passing task-specific metadata

Scorers and metrics sometimes need per-instance data beyond the basic
Instance fields. For example, code execution scoring needs `test` and
`entry_point`, and IFEval constraint checking needs `instruction_id_list`.

Put these fields in the `metadata` dict of each JSONL record:

```json
{"metadata": {"test": "assert foo(1) == 2", "entry_point": "foo"}, ...}
```

The `SerializedTask` passes `metadata` through to `Instance.metadata`
unchanged, so they are available at scoring time via
`instance.metadata["test"]`.

If you need behavior the default `SerializedTask` does not provide -- for
example, a custom `extract_answer` method, per-instance `SamplingParams`,
or reading `system_prompt` from metadata into `LMRequest` -- subclass
`SerializedTask` and register your subclass as a new base task.

---

## Limitations and gotchas

Serialization works well for straightforward completion, loglikelihood, and
chat tasks. More complex task types can introduce complications:

- **Baked-in formatting is model-specific.** The serialized prompt includes
  any chat template, FIM tokens, or context-window truncation applied during
  serialization. A JSONL file produced for one model family may not be valid
  for another. This is usually fine since evaluation suites tend to be
  model-specific already.

- **Dual-mode tasks (generation + BPB).** Some oe-eval tasks issue both a
  `generate_until` and a `loglikelihood` request per instance. These need
  two separate JSONL files -- one per request type -- registered as separate
  variants with different metrics.

- **Multiple-choice PMI normalization.** Unconditioned-prompt requests (used
  for `acc_norm`) are not serialized. If needed, the olmo-eval consumer
  would construct them at runtime from the `continuations` list and a
  task-level unconditioned prompt string.

- **Rolling / corpus perplexity.** Short-document perplexity can be
  serialized as loglikelihood with `prompt=""` and a single continuation.
  True sliding-window rolling for long documents requires olmo-eval
  infrastructure that does not exist yet.

- **Gold answer reliability.** When serializing from oe-eval,
  `doc_to_target()` extracts `gold_answers`, but this method is not always
  the canonical answer source for every task. Verify the serialized answers
  match what scoring expects before running at scale.

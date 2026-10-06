# Working on olmo-eval

olmo-eval supports base-model evaluation, post-training checks, and tool-augmented or
agentic evaluation. The central design requirement is that a result can be traced back to
the data, configuration, code, and environment that produced it. A new benchmark should
fit that model and be understandable to the next person who runs it.

This guide explains the design and takes a contribution from local inspection to a trial
run. [README.md](README.md) is the installation and API reference;
[AGENTS.md](AGENTS.md) contains the implementation rules and verification commands for
coding agents. Humans reviewing code should know those rules too.

## Start with one evaluation

Follow the [quick start](README.md#quick-start) to install with `uv sync --frozen` and
set up pre-commit hooks. From the repository root:

```bash
uv run olmo-eval tasks -f gsm8k
uv run olmo-eval suite inspect olmobase:code
uv run olmo-eval run -m mock -t gsm8k --dry-run
```

These commands show the registered names, expand a suite, and resolve a run's
configuration without loading a model. Follow the definition in
[`gsm8k.py`](src/olmo_eval/evals/tasks/gsm8k.py) through `process_doc`, `format_request`,
and `extract_answer`. For a new contribution, also read a nearby task and its tests that
use the same request and scoring types. Existing tasks include legacy conventions; use
the rules below when deciding which parts to copy. The
[task authoring reference](README.md#adding-new-tasks) has the class and variant APIs.

## How an evaluation is put together

The normal path is:

```text
DataSource -> Task.process_doc -> Instance -> Task.format_request -> LMRequest
                                                                    |
                                                       Harness / Provider
                                                                    |
                   saved results <- Metrics <- extraction / scoring / responses
```

A **task** defines the measurement: data, split, few-shot examples, prompts, sampling
parameters, answer extraction, scoring, and aggregation. A **formatter** turns instances
into requests. A **scorer** evaluates outputs; a **metric** aggregates scores and exposes
per-instance values where meaningful. Requests use `CHAT` or `COMPLETION` for generation,
and `LOGLIKELIHOOD` for likelihood-based multiple choice and bits-per-byte evaluation.

A **harness** defines the execution setup: provider, tools, scaffold, sandboxes, auxiliary
providers, concurrency, and failure tolerance. A **provider** adapts requests to a model
backend. A **scaffold** controls a multi-turn interaction. The task should not need to know
whether it is running locally or on Beaker.

This separation lets one task run with different models and agent setups. Tools, system
prompts, and scaffolds can change performance, so record the harness as part of the
experimental condition. For judge tasks, the grading protocol and judge settings belong
to the measurement; the harness supplies any auxiliary provider needed to execute it.

An **external eval** wraps a benchmark with its own runner, such as tau2 or SciCode. It
uses `run-external` and the external-eval result contract instead of forcing its execution
loop into the normal task pipeline. Use this route when the benchmark needs that runner.

### Where changes belong

Paths below are relative to `src/olmo_eval/` unless stated otherwise.

| Change | Start here |
| --- | --- |
| Text benchmark or variant | `evals/tasks/`; base types and registry in `evals/tasks/common/` |
| Frozen subsets, exemplars, long constant tables | `evals/tasks/constants/` |
| Group of benchmarks and its aggregation | `evals/suites/` |
| Shared answer extraction | `evals/extract/` |
| Reusable formatting, scoring, metrics | `common/formatters.py`, `common/scorers/`, `common/metrics/` |
| Multimodal benchmark | `evals/vision/benchmarks/`, with shared tasks, data, and scoring under `evals/vision/` |
| Benchmark with its own runner | `evals/external/benchmarks/` |
| Data access | `data/` |
| Model backend | `inference/providers/`; shared request/response types in `common/types/` |
| Tools, scaffolds, sandboxes, execution presets | `harness/` |
| Scheduling, aggregation, saved artifacts | `runners/` |
| CLI and remote submission | `cli/`, `launch/` |
| Results storage and analysis | `storage/`, `analysis/`; migrations at repository-root `alembic/` |
| Tests and maintenance scripts | Repository-root `tests/` and `scripts/` |

Compose a new benchmark from existing parts before extending shared interfaces. Keep
benchmark-specific behavior in its module; move a helper into shared code when its
consumer needs a reusable interface. Avoid adding hooks for hypothetical future tasks.

Registration happens on import. Text task modules, suite modules, and external benchmark
packages are discovered automatically. Vision benchmarks have an explicit import list in
[`evals/vision/benchmarks/__init__.py`](src/olmo_eval/evals/vision/benchmarks/__init__.py).
Check that the public CLI discovers the new benchmark; a direct test import can hide a
missing registration step.

## Keep the meaning of a result stable

A **task spec** such as `humaneval:3shot:bpb` selects a registered task and applies
variants from left to right. Some base names contain colons, so the registry resolves the
longest registered task prefix first. A **suite** lists tasks or nested suites and declares
how their scores combine. Equal task weighting and instance weighting answer different
questions; match the reference's aggregation explicitly.

Treat a published spec's default measurement as fixed. A prompt, extraction, data, metric,
or sampling-budget change gets a new variant or versioned name. This includes fixing a
reference implementation's scoring bug: retain the parity behavior under its old name
and make the correction explicit in the new spec. Shared helper changes can affect many
specs, so audit their callers as part of review.

The spec is a convenient name; the **resolved configuration** records the actual task,
including overrides. `TaskConfig.to_dict()` supplies its stored representation and hash.
A runtime override such as `-o limit=5` is useful for trials, but that result describes a
limited run. Save the overrides when reporting it, and register a descriptive variant for
a setting others should routinely reuse. The task hash does not fingerprint the source
code or capture the entire model and harness configuration.

Reproducing the experiment also requires the code revision, model/checkpoint and tokenizer,
harness, data revisions, and dependency environment. Pin new datasets and judge models;
use the checked-in lockfile and record the image used for remote runs. This makes the
procedure recoverable; stochastic generation and hosted services can still vary between
runs.

### Match the reference before improving it

For a port, identify the reference config and revision before coding. Match the split,
few-shot selection, exact prompt and suffix, stop sequences, token budget, extraction,
normalization, and aggregation. Cite the reference in the task module and PR.

Compare requests and per-instance scores as well as headline numbers. A similar aggregate
can hide two incompatible implementations. Report our result beside the reference using
the same model and regime where possible; state when only code or prompt parity was
checked. Improvements belong in separately named variants.

### Make failures distinguishable from wrong answers

A model's wrong or unparseable answer may legitimately score zero under the benchmark's
rules. A provider outage, exhausted judge retry, or failed scoring process must remain an
explicit failure. Preserve the raw response and diagnostic information needed to tell
them apart, and check that it reaches saved predictions.

The runner tracks hard failures and applies `max_hard_failure_rate`; a task with no scored
responses fails. Before trusting a score, check completion status, failure counts, the
number of scored examples, and representative predictions. Do not raise failure tolerance
just to make a trial appear successful.

## Validate a new eval in stages

### 1. Inspect locally and exercise the pipeline

Replace `gsm8k` below with the new task spec after it is registered:

```bash
# Loads task data and shows the requests before model inference.
uv run olmo-eval task inspect gsm8k --request -n 2

# Resolves configuration; does not exercise generation or scoring.
uv run olmo-eval run -m mock -t gsm8k -o limit=5 --dry-run

# Exercises data loading, formatting, mock inference, scoring, and artifact writing.
uv run olmo-eval run -m mock -t gsm8k -o limit=5 \
    --inspect-request --inspect-response --output-dir /tmp/olmo-eval-gsm8k-smoke
```

Task inspection and mock runs may download data. `mock` replaces the model provider;
judge calls, tools, and code-scoring sandboxes may still need credentials or services.
Unit tests should use local fixtures and mocked services. A mock score says nothing about
model quality or reference parity.

Requests and predictions are saved by default. Inspect the output directory's `requests/`,
`predictions/`, and `metrics.json`: verify prompt text, gold labels, extracted answers,
per-instance metrics, and failure accounting. Use a fresh directory for each trial so
artifacts remain attributable. Add focused tests and run the
[local checks](AGENTS.md#tooling-and-verification) before a remote trial.

### 2. Run a small real-model trial

GPU trials use `olmo-eval beaker launch`, which submits through gantry. Gantry clones the
repository at a pushed commit: uncommitted edits and unpushed commits are not a way to test
new code remotely. Commit and push the intended branch before submission. Agents do this
only when the human has asked them to commit and push.

Choose a model and harness appropriate to the benchmark's regime. Confirm Beaker access,
an appropriate workspace/budget and cluster allocation, and any model, data, judge, or tool
credentials. Select a GPU count that fits the model. Set `--min-runtime` to cover setup
and the expected run, and set `--timeout` to bound it. Check current allocation and queue
state when choosing a cluster; a copied workspace or high priority does not establish
that the job can schedule there.

This template previews one limited trial. Replace every angle-bracketed value, including
the resource estimates:

```bash
uv run olmo-eval beaker launch \
    --name '<trial-name>' \
    --model '<model-preset-or-checkpoint>' \
    --harness '<harness-preset>' \
    --task '<task-spec>' -o limit=10 \
    --cluster '<cluster>' --workspace '<workspace>' --budget '<budget>' \
    --gpus '<gpu-count>' --min-runtime '<duration>' --timeout '<duration>' \
    --inspect-request --inspect-response --no-follow --dry-run
```

Check the resolved task, provider, image, resources, and command in the preview. Beaker
preview can require authentication and perform credential setup; it does not submit the
evaluation. Remove `--dry-run` to submit the reviewed command. `--no-follow` returns after
submission. See the [Beaker reference](README.md#launching-on-beaker) for YAML configs,
provider overrides, secrets, storage, and multi-model runs. Task and harness `-o` overrides
attach to the preceding `--task` or `--harness`.

For an external eval, use `--external-eval` / `-E` and its `--eval-arg` / `-A` options
instead of `--task`. Consult that benchmark's arguments for limiting the trial; external
evals do not share the normal task `limit` override.

### 3. Inspect the trial before expanding it

Keep the experiment URL/ID and submitted command. To attach to its logs:

```bash
uv run olmo-eval beaker watch -e '<experiment-id>'
```

With the standalone Beaker CLI, bounded reads and result retrieval are also available:

```bash
beaker experiment logs '<experiment-id>' --tail 200
beaker experiment tasks '<experiment-id>' --format=json
beaker experiment results '<experiment-id>' -o '/tmp/olmo-eval-<trial-name>'
```

Inspect the task/job list and result files in Beaker before downloading. The experiment
download selects the latest execution of each task; for an earlier retry, fetch that job's
result dataset explicitly. If a job is queued or fails, inspect its events and logs before
relaunching. A launch succeeding only confirms submission.

Check real requests and predictions for prompt mismatch, truncation, extraction errors,
judge failures, and missing scores. Only then expand to the intended dataset and compare
with the reference. Keep trial results distinguishable from full benchmark results.

## Make the contribution easy to review

A task PR normally contains the task, its tests, and any suite entry it needs. Keep unrelated
refactors, dependency upgrades, and image changes separate. Split a larger feature into
independently reviewable, working steps. Check open PRs for overlapping work first.

The PR should explain the measurement, reference and intentional deviations, which existing
specs are affected, and what was actually verified. Include exact commands, code revision,
Beaker experiment IDs, reference comparisons, and unresolved gaps as applicable. An
unrun check belongs under remaining work, not validation. Use an imperative, sentence-case
title suitable for the squash commit.

Review owners are listed in [.github/CODEOWNERS](.github/CODEOWNERS).

For code review, apply the [agent guide](AGENTS.md), especially the task-identity, failure,
and scoring rules. Keep docstrings about stable interfaces and comments about why the code
works that way. When a workflow changes, update the relevant guide with the code; use
links to the README for detailed options rather than maintaining another CLI reference.

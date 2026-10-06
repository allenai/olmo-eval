# Agent guide for olmo-eval

[HUMANS.md](HUMANS.md) explains the design, code organization, and contributor workflow;
[README.md](README.md) is the installation and API reference.

Protect the meaning of stored results: changes to a measurement must be identifiable,
and its configuration, code, data, and environment must be recoverable. The task hash is
only one part of that record.

## Working sequence

1. Read the relevant module and tests, check the working tree, and check open PRs for
   overlapping changes. Preserve existing user edits.
2. Identify which task specs or execution paths the change affects. For a port, identify
   the reference config and revision before implementing it.
3. Make one focused change, reuse existing components, and add tests for new behavior.
4. Run focused tests while iterating, then the applicable verification below. For a new
   eval, follow the [trial workflow](HUMANS.md#validate-a-new-eval-in-stages).
5. Report what changed, the checks actually run, and remaining gaps. A submitted Beaker
   job is not a verified result; inspect its completion and artifacts.

Do not commit or push unless the human asks. When a requested remote trial needs a pushed
commit, finish the local code, tests, and concrete launch command before explaining that
dependency.

## Tooling and verification

- Use `uv run` for Python commands. Install from the lockfile with `uv sync --frozen`;
  use `uv add` for new dependencies, never `pip install`.
- Do not hand-edit `uv.lock`. Use `uv lock` / `uv add` only when dependency changes are
  in scope. Keep incidental lockfile changes out of other PRs.
- CI's unit tests run on Python 3.12 and 3.13. CI uses `uv run --frozen --no-group vllm`;
  use those flags to reproduce its environment.
- vLLM/CUDA dependencies have Linux platform markers. Unit tests must run on macOS and
  Linux without a GPU, network, or Docker. GPU tests use `--gpu`; Docker-backed tests
  live in `tests/integration/`.

For code changes, run:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/ alembic/
uv run pytest tests/ --ignore=tests/integration -v
```

`./scripts/fix.sh` formats and applies lint fixes across `src/` and `tests/`; avoid
unrelated edits. `./scripts/verify.sh` syncs dependencies and runs checks including
Docker-backed integration tests (`--no-docker` skips those; `--gpu` enables GPU tests).
`make type-check` checks only `src/`, so use the command above to include `alembic/`.
Run relevant integration tests for storage changes. For docs-only changes, check links,
paths, and documented CLI syntax; Python tests are needed only if executable code changed.

Useful discovery commands:

```bash
uv run olmo-eval tasks -f gsm8k
uv run olmo-eval suites
uv run olmo-eval suite inspect olmobase:code
uv run olmo-eval harnesses
uv run olmo-eval models
uv run olmo-eval external-evals
uv run olmo-eval run -m mock -t gsm8k --dry-run
```

## Code organization and registration

Use the [repo map](HUMANS.md#where-changes-belong) to choose the owning package.

- Tasks live in `src/olmo_eval/evals/tasks/`; variants are `TaskConfig` overrides.
  Specs apply variants left to right, resolving the longest registered task prefix first.
  Task specs and suite names must not collide.
- Variant names describe the measurement or regime (`mc`, `rc`, `bpb`, `gen`, `cot`,
  `judge`, `3shot`). Prefer composing descriptive variants to naming them after a model.
- Text tasks, suites, and external benchmark packages are discovered by import. Vision
  benchmarks require an explicit import in `evals/vision/benchmarks/__init__.py`.
  Verify discovery through the public registry/CLI.
- Shared extraction goes in `evals/extract/` and its exports; reusable scorers go in
  `common/scorers/`. Check `MultipleChoiceScorer`, `MCQAChatFormatter`,
  `LogprobMCAccuracyMetric`, and existing extractors before adding another implementation.
- Put frozen subsets, static few-shot examples, and long tables in
  `evals/tasks/constants/`. Put specialized data access in `data/` or the existing
  vision/external data layer.
- Tasks do not import suite modules or depend on Beaker. Harnesses own execution policy;
  tasks own the measurement. External evals with their own runners use
  `evals/external/benchmarks/` and its result contract.
- Keep Beaker-specific launch logic out of the runner; runners must work locally too.
- CLI `-o` overrides attach to the preceding `-t` or `--harness`; task configuration
  overrides must be `TaskConfig` fields.

## Task identity and reference parity

`TaskConfig.to_dict()` in `evals/tasks/common/base.py` supplies the serialized task
configuration used for hashing and storage. Code changes are not automatically reflected
in that hash.

- Do not change an existing spec's prompt, data, extraction, metric, or sampling budget
  in place. Add a descriptive variant or versioned spec (`:v2`), and state in the PR
  which results become incomparable. Audit callers of shared helpers too.
- Every task setting that changes requests or scores must reach `to_dict()`, including
  nested formatter/scorer settings, few-shot sources, and judge settings. Execution
  settings must be recorded in the model/harness configuration. Do not read unrecorded
  output-affecting settings from environment variables at scoring time.
- New optional fields must preserve existing hashes when unused. Follow the conditional
  serialization patterns for `prompt_templates`, `strip_thinking`, and judge settings.
  Test both that changing the setting changes the hash and that an unrelated existing
  task retains its previous hash.
- Pin a new dataset's `revision=` at introduction, and pin judge models to dated
  snapshots. Adding a pin later changes the task identity; handle that deliberately.
- Cite the reference config, paper, or repo revision in the task module and PR. Match
  sampling parameters, few-shot count/selection, prompt text/suffixes, stop sequences,
  extraction, and normalization. Keep reference bugs under the parity name; put a
  correction in a new variant.
- Put prompt suffixes in formatter configuration when supported, rather than appending
  them in `process_doc`. Preserve the reference's exact resulting prompt.
- Suites declare an explicit aggregation strategy. Check whether the reference weights
  tasks or instances; instance weighting uses `WEIGHTED_AVERAGE`.
- Show reference results beside ours when run under comparable conditions. Label prompt,
  code, or numerical parity precisely; a mock run or a similar score is insufficient to
  claim numerical parity.

## Failure visibility and scoring

An infrastructure failure must remain distinguishable from a benchmark score.

- Do not catch broad exceptions around provider, judge, or scoring calls and substitute a
  score. Exhausted retries and judge parse failures need explicit failure accounting and
  available raw responses. Catch narrowly when recovery is defined.
- Never treat missing logprobs as logprob zero. Preserve the provider's missing-value
  handling (often `-inf`) and propagate unusable outputs through failure accounting;
  do not let them produce a plausible metric.
- Do not use sentinel scores such as `-1` or NaN without an explicit error. A model's
  invalid answer can score zero if the benchmark defines that outcome; a broken grader
  cannot. A task with no scored responses must fail the run.
- Diagnostics must survive serialization. Check `runners/io/builders.py` and saved
  predictions for `answer_format_correct`, `judge_result`, scoring errors, and relevant
  raw output; arbitrary metadata is not automatically copied.
- Telemetry/progress errors are caught and logged so they do not kill the evaluation.
  Do not apply that exception policy to inference or scoring.
- `Metric.compute_instance()` defaults to the scorer value. Override it when the metric
  differs from that value, and return `None` for out-of-scope instances. Check saved
  per-instance values as well as the aggregate.
- Accuracy, F1, and other bounded quality metrics use 0–1, not percentages. Preserve the
  native units of metrics such as bits per byte and perplexity.
- For extraction, test negative numbers, thousands separators, malformed answers, and
  answers inside `<think>` traces as applicable. New parsing/scoring logic needs direct
  unit tests.
- Treat `num_fewshot=0` as an explicit value (`is None` means unset). Check that an
  explicit token budget respects the context window and a requested split is not
  silently replaced by a class default.

## Shared interfaces, environments, and storage

- Changes to `SamplingParams` or `LMRequest` must work across affected providers
  (`vllm`, `vllm_server`, `hf`, `litellm`, `olmo_core`, and vision providers), or have an
  explicit supported scope and validation.
- Add a provider kind to `ProviderKind` and audit its allowlists. Search for an existing
  kind, including `_GPU_PROVIDERS` in `inference/providers/config.py` and
  `_SEQUENTIAL_ONLY` in `runners/asynq/batching/config.py`.
- Do not import CUDA-only or optional packages at module import time in shared code.
  Keep heavy/side-effectful imports, including downloads, inside the code that needs them.
- Deep-copy mutable preset/config contents before applying overrides. Cached harnesses
  must not change across runs. Keep multiprocessing payloads small and picklable; pass
  paths instead of decoded images.
- Schema changes need a migration in the correct chain (`alembic/results` or
  `alembic/metrics`), with one head per chain. Tell users to run `make db-upgrade`.
- Do not introduce Ai2-specific paths, workspaces, or buckets as source defaults. Accept
  deployment-specific values through configuration; this repository is public.
- Explain dependency pins in `pyproject.toml`. A core dependency upgrade (torch, vLLM,
  transformers, OpenAI client) needs a representative real-model canary. An image bump
  also updates the applicable `BEAKER_DEFAULT_IMAGE` / `BEAKER_SANDBOX_IMAGE` constant
  in `common/constants/infrastructure.py`.

## Code style and tests

- Keep interfaces stable and avoid coupling general methods to benchmark-specific config
  fields. Add fields and hooks with their first consumer.
- Keep docstrings general and comments short and about why. Temporary project status and
  PR-description material belong in the PR, not in code comments. Avoid hard-coded counts
  of tasks or examples in docstrings.
- Type dictionaries that cross interfaces with `TypedDict` when they have a defined
  schema. Use `Literal` or validation for fixed string choices, and tuples for immutable
  collections. Ruff owns formatting (line length 100).
- Do not add `ty` ignores. `allowed-unresolved-imports` in `pyproject.toml` is for optional
  dependencies only.
- Add unit tests for new code. Use pytest functions and fixtures, not `unittest.TestCase`.
  Update tests to reflect intended behavior; do not distort source code to satisfy stale
  tests. A regression test must fail against the old implementation; verify that when
  practical without disturbing the user's changes.
- Exercise public paths (`load()`, `get_task(spec)`, CLI), not just private helpers. For a
  new task, test registration and every variant, representative documents, formatted
  requests, extraction edge cases, and any new config/hash behavior.
- Use local data fixtures and mock providers/services, never a real model in unit tests.
  Scripts under `scripts/` that generate checked-in data need tests in `tests/scripts/`.

## Trial runs and handoff

Follow the [local-to-Beaker workflow](HUMANS.md#validate-a-new-eval-in-stages). Use
`olmo-eval beaker launch` for GPU trials. Before submission, verify the pushed code
revision, model/harness, task limit, workspace/budget/cluster, GPU count, minimum runtime,
timeout, and required secrets. Credentials go through secret mappings, not plaintext
`--env` values. Keep resource choices explicit instead of copying a teammate's defaults.

A local `--dry-run` resolves configuration. A mock run exercises the pipeline but can
still invoke real judges, tools, or sandboxes. Beaker `--dry-run` previews submission and
may perform credential setup. Know which check is needed before running it.

For a requested trial, inspect logs, completion status, scored/failure counts, and saved
requests/predictions. Record experiment IDs and results; diagnose failures before
relaunching or increasing the limit. If credentials or infrastructure prevent a check,
state the exact gap and leave a runnable command for the human.

## PR scope and description

- One behavior change per PR: a task port plus tests and a needed suite entry, or a bug
  fix plus its regression test. Keep unrelated infrastructure, prompt edits, dependency
  upgrades, image bumps, and optional refactors separate.
- Split large features into a numbered series of independently working, reviewable PRs.
  Report unrelated bugs separately rather than fixing them in the same patch.
- Use a sentence-case imperative title without a conventional-commit prefix; it becomes
  the squash commit on `main`.
- Explain what changed and why, the reference and intentional deviations, affected specs
  and stored-result comparability, and exact validation commands. Include Beaker IDs and
  parity comparisons when available. Do not claim checks that did not run.
- List unresolved gaps and deferred work, linking existing follow-up issues. Update the
  description when the code changes during review.

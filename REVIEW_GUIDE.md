# Reviewing pull requests in olmo-eval

This guide tells a reviewing agent how to review a pull request against this
repository. It codifies the standards the original maintainer applied across
every review from the project's founding through August 2026. Follow it to keep
the repo well-structured, free of the bug classes that have recurred here, and
internally consistent.

Read the whole guide once. Then, for each PR, work through the procedure in
section 2 and write the review using section 6.

---

## 1. What this repository is and what matters most

olmo-eval is a workbench for evaluating language models. A **Task** turns a
dataset into **Instances**, a **Formatter** turns an instance into an
**LMRequest**, an **InferenceProvider** answers it, a **Scorer** scores one
output, and **Metrics** aggregate scores across responses. **Variants** compose
onto a task name (`humaneval:3shot:bpb`). **Suites** group tasks. A **Harness**
wraps execution with tools, scaffolds, and sandboxes. Results and instance-level
predictions are persisted to files and to a PostgreSQL store.

The single most important property of this system is that **a stored result is
reproducible from its stored configuration**. Every review decision flows from
that. In order of weight, a reviewer protects:

1. **Persisted-result correctness and task identity.** Aggregate numbers,
   per-instance predictions, and the task hash must all be right and must all
   agree. Findings here block the merge.
2. **Failure visibility.** Infrastructure failures must never become scores.
   Findings here block the merge.
3. **Parity with the reference implementation** (oe-eval, a paper, an official
   repo). Divergence must be intentional, named, and documented.
4. **Portability.** No Ai2-specific paths or infrastructure assumptions in
   source.
5. **Structure that makes the next contributor's mistake impossible.**
6. **Typing, tests, naming, and documentation hygiene.**

Key files a reviewer will need repeatedly:

| Concern | Where |
|---|---|
| Task config, serialization, task hash | `src/olmo_eval/evals/tasks/common/base.py` (`TaskConfig.to_dict`, `get_primary_metric`, `Task._get_scorers`) |
| Metric base, `compute_instance` fallback, display format | `src/olmo_eval/common/metrics/base.py` |
| Scorer base classes | `src/olmo_eval/common/scorers/` |
| Prediction row construction | `src/olmo_eval/runners/io/builders.py` (`build_predictions`) |
| Shared request/sampling types | `src/olmo_eval/common/types/base.py` |
| Answer extraction | `src/olmo_eval/evals/extract/` |
| Provider kind allowlists | `src/olmo_eval/inference/providers/config.py` (`_GPU_PROVIDERS`), `src/olmo_eval/runners/asynq/batching/config.py` (`_SEQUENTIAL_ONLY`) |
| Infrastructure constants | `src/olmo_eval/common/constants/infrastructure.py` |
| Sandbox execution | `src/olmo_eval/harness/sandbox/` |
| Project conventions | `CLAUDE.md`, `DEVELOPMENT.md` |

---

## 2. Review procedure

Do these steps in order. Do not skip to writing findings.

### Step 1. Read the PR description and classify the change

Identify which of these the PR is, since each has its own checklist in
section 3:

- **A new task or benchmark port** (most common).
- **A scorer, metric, or extraction change.**
- **A provider or inference change.**
- **A runner, storage, or CLI change.**
- **An external or sandboxed evaluation.**
- **A dependency or image bump.**
- **Documentation or cleanup.**

Note what the author claims about testing and parity. You will verify those
claims, not accept them.

### Step 2. Run the verification the project expects

```
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/ alembic/
uv run pytest tests/ --ignore=tests/integration -v
```

`./scripts/verify.sh` runs the same set. If CI is red, say so first and check
whether `main` is green before attributing it to the PR.

### Step 3. Read the whole diff, then read the code the diff touches

Read every changed file completely. Then open the callers and callees of
anything new. Many of the worst bugs found in this repo were not in the diff
but in how the diff interacts with an existing default: a metric inheriting a
fallback, a config field never reaching `to_dict`, a provider kind missing from
an allowlist. Reading only the diff will miss them.

### Step 4. Reproduce before asserting

When you believe you have found a bug, confirm it. Run the function twice,
compute the two hashes, instantiate the metric with the mock provider, count
the collisions in the dataset. State what you did in the review ("I reproduced
this with two consecutive calls", "I confirmed both configurations produce the
same hash"). A reviewer who verifies is trusted; one who speculates is not.

If a claim cannot be verified without a GPU or credentials, say so and ask the
author for the specific evidence (a side-by-side runtime, a parity table, a
run against the test workload).

### Step 5. Walk the checklists

Work through section 3 for the PR's category and section 4 for every PR. Write
down each finding with the file and line, the concrete failing input or state,
and the observed wrong output.

### Step 6. Rank, then write

Sort findings: blocking (correctness of persisted results, task identity,
failure visibility, crashes) first; then parity and behavior; then structure;
then hygiene and nits. Write the review as described in section 6.

---

## 3. Checklists by change type

### 3.1 New task or benchmark port

**Task identity and reproducibility**

- Every setting that affects the request or the score is a field on
  `TaskConfig` and is included in `TaskConfig.to_dict()`. Prompt style, prompt
  templates, judge model, judge reasoning effort, judge token budget, system
  prompt, and few-shot source all count. If a field is output-affecting but
  absent from serialization, two different runs collide on the same task hash
  and artifact name. This is a blocking finding. Ask for a test asserting that
  changing the field changes the hash.
- Nothing output-affecting is read from an environment variable at scoring
  time without also being recorded in the config.
- Judge and model references use a dated snapshot, not a mutable alias.
- Dataset references pin a revision where the loader supports it.
- Arguments that look like they control something but do not (for example a
  `split=` passed to a `DataSource` that the class-level split always
  overwrites) are removed.
- Task names do not collide with existing suite names. Check the registry.
- Variant names describe a format or metric modifier, not a model or a
  research regime. Model-specific presets are composed from variants, not
  registered as a single opaque variant.

**Parity**

- The PR states what reference it matches (oe-eval config name, paper, official
  repo) and shows numbers or a comparison. If it does not, ask.
- Sampling parameters, few-shot counts, prompt suffixes, and stop sequences
  match the named reference. If the PR borrows settings from a differently
  named reference config (for example a `_deepseek` variant's temperature),
  ask it to be renamed to reflect that regime.
- Where the reference has a bug, the port reproduces the bug-compatible
  behavior under the existing name and lands the fix as a versioned variant
  (`:v2`). Do not accept silent divergence or a backport to oe-eval as the
  plan.
- Prompt-affecting knobs the reference exposes (temperature, stop sequences,
  prompt template, regex set) are overridable via `register_variant`. Prompt
  suffixes belong in the formatter (`ChatFormatter(user_template=...)`), not
  concatenated inside `process_doc`, so future variants are a config change
  and not a refactor.

**Data handling**

- Static few-shot examples and long constant tables live in an adjacent
  constants module, not inline in the task module.
- Loaders stream and apply `limit` while reading. A loader that decodes an
  entire multi-gigabyte file and then slices is a blocking finding for
  long-context tasks. Ask whether the full suite has been run without
  exhausting memory.
- Downloads fetch only the shard the task needs, not a whole repository.
- Heavy or side-effectful imports (NLTK setup, model downloads) are lazy and
  live inside the function that needs them, so importing the scorers package
  for an unrelated task does not trigger them.
- Instance identifiers are unique. If a fallback id is used, check the real
  dataset for collisions.

**Scoring inside the task**

- Custom extraction and parsing logic has a lightweight unit test.
- Every regex-based extractor is checked for sign stripping, comma handling,
  and thinking-trace leakage (`<think>...</think>` content matched as an
  answer).
- Diagnostics the reference reports (`format_correct`, raw judge response,
  parse status, rubric points) are written to `output.metadata` so a parse
  failure can be distinguished from a genuine zero after the run. Verify they
  survive into saved predictions via `build_predictions`.
- Unexpected branches are handled explicitly: an `else` that returns `None` or
  raises; a `ValueError` for an unknown style string; a sentinel when a subset
  is empty so a small `limit` run does not divide by zero.

**Portability**

- No `/weka/...` paths, Beaker workspace names, or Ai2 bucket names as
  defaults in task or runner code. Defaults are generic; Ai2 users override at
  launch. If a task only works on internal data, either name it as internal
  or make the data reproducible, and file a follow-up issue.

### 3.2 Scorer, metric, or extraction change

- **`compute_instance` fallback.** `Metric.compute_instance()` falls back to
  the shared scorer channel when no metric-specific value exists. Any metric
  whose per-instance meaning differs from the scorer's return value (precision
  vs F1, consistency vs recall, a bucketed or subset metric) must override
  `compute_instance`, returning `None` when the instance is outside the
  metric's scope. Otherwise persisted per-instance values are wrong while
  aggregates look right. This has recurred in three PRs and is always
  blocking. Ask for a regression test that inspects persisted instance
  metrics.
- **Scale.** Metrics report on the framework's 0 to 1 scale.
  `Metric.pairwise_display_format()` classifies by name; a metric named
  `recall` returning 0 to 100 renders as thousands of percent in the viewer.
  Either keep 0 to 1 or override the display format explicitly.
- **Multi-sample aggregation.** `TaskConfig.output_score_aggregation`
  (default MAX) must be applied before averaging across responses. A metric
  that flattens every output ignores it.
- **Scorer identity.** `Task._get_scorers` keys scorers by `name` and drops
  duplicates silently. A scorer with two fields that must agree (a `name` and
  a `flex` flag, for example) is a foot gun; derive one from the other in
  `__post_init__`.
- **Scorer serialization.** Scorer instances with configuration serialize that
  configuration in `Metric.to_dict`, not only the class name, so differently
  configured judges hash differently.
- **Reuse.** Scoring logic lives in a `Scorer`, not reimplemented inside a
  `Metric`. Generic scorers (set recall, Jaccard, substring) belong in
  `common/scorers`, not in a task module. Check for an existing implementation
  before accepting a new one.
- **Logprobs.** Missing or incomplete logprobs are treated as `-inf`, not 0.
  Check that a change does not quietly revert this.
- **Two BPB formulations exist by agreement** (byte-weighted and
  instance-averaged). Do not accept a PR that removes one; do accept one that
  makes the choice explicit.
- **Metadata assumptions.** A metric that reads
  `instance.metadata["some_flag"]` must tolerate its absence (`.get`) or the
  scorer must always initialize it before any path that can raise.
- **Judge parsing.** Parse failures set an explicit parse-error flag, record
  the raw response, and return 0.0. `KeyError` on an unexpected label must not
  escape a block that only catches `ValueError`. Partial judge responses (fewer
  labels than statements) are invalid, not valid.
- **Location.** Extraction code lives in `src/olmo_eval/evals/extract/` and is
  exported from its `__all__`.
- **Dead parameters.** Parameters and branches with no callers are removed or
  justified with a comment naming the planned caller.

### 3.3 Provider or inference change

- **Shared contracts apply to all providers.** A change to `SamplingParams`
  or `LMRequest` (for example allowing `max_tokens=None`) is either handled by
  every provider or explicitly scoped to one. Check `vllm`, `vllm_server`,
  `hf`, `litellm`, and `olmo_core`. A crash in a provider the author did not
  test is the author's to fix before merge.
- **Allowlists.** A new provider kind is added to every allowlist:
  `_GPU_PROVIDERS`, `_SEQUENTIAL_ONLY`, provider extra normalization, package
  override normalization. Search the codebase for the existing kinds and
  expect the new one beside each.
- **Fail fast on terminal errors.** An unrecoverable provider state (engine
  death, dead worker) propagates as a terminal error and cancels siblings. It
  is never handled per request as an ordinary failure.
- **Boundary handling for logprob requests.** Empty continuation, continuation
  exactly at the limit, continuation over the limit, and a per-request
  `max_length` are each handled and tested. Truncation happens after those
  checks. Mirror the existing text provider's behavior.
- **Memory.** No cached full-table copies (embedding tables, decoded image
  tensors) without a clear need. Caches are released in `close()`.
- **Idempotence.** Module shims and import guards work on the second call in
  the same process (multiple provider instances, replica sets). Ask for a test
  that calls the helper twice.
- **Correctness-sensitive logic is isolated and unit-tested.** Attention mask
  construction, chat template application, and tokenization boundaries are
  module-level helpers with direct tests that do not need a model.
- **Optional dependencies** stay optional: no import of a CUDA-only or
  provider-only package at module import time in shared code.

### 3.4 Runner, storage, or CLI change

- **A run with no scored responses is a failure.** A gate that can be
  configured to pass an all-failed task (for example `max_hard_failure_rate=
  1.0`) is a bug. Validate configured ranges.
- **Error information is not dropped on persistence.** If two error sources
  exist (a gate error and an instance-failure summary), both reach storage.
- **Progress and telemetry never terminate a run.** Anything that raises in a
  progress callback or metrics reporter is caught and logged. This is the one
  place where swallowing exceptions is correct.
- **Progress reports stay below 100% for incomplete runs.**
- **Alembic.** Schema changes ship with a migration, and the PR reminds users
  to run it.
- **Queue payloads are small and picklable.** Do not put decoded images or
  large tensors into multiprocessing queue items; pass a path or lazy
  reference and resolve in the worker.
- **Dependency direction.** Task modules do not import from suite modules.
  Runner code does not know about Beaker.
- **Presets are not mutated** by harness or CLI overrides.

### 3.5 External or sandboxed evaluation

- Upstream repos and agent frameworks are pinned to a commit, not a branch.
- No `/weka` or workspace-specific defaults for data files (see 3.1).
- Provider type checks unwrap `InstrumentedProvider` before `isinstance`.
- Sandbox capabilities advertised match what `execute_*` calls need.
- The sandbox image's Python version is compatible with `requires-python`.
- Time-budget constants carry a unit suffix (`_SEC`).
- Fallback URLs do not make API-backed providers look local.
- Wrappers around upstream configuration do not silently override budgets the
  user set (for example replacing a user's `max_tokens` with a stage default).
- Network-heavy tools (crawling, external search) note concurrency limits so
  parallel jobs from shared infrastructure do not hammer one domain.
- Date or version comparisons preserve "unknown" rather than padding to a
  value that defeats a conservative cutoff.

### 3.6 Dependency or image bump

- The lockfile is updated and the PR states how it was regenerated.
- A core bump (torch, vLLM, transformers, the OpenAI client) is accompanied by
  a canary run: build and push the image, run the test workload or a small
  parity subset, and paste the result.
- Transitive upgrades that touch the provider layer (the OpenAI client, for
  example) are called out and checked.
- `BEAKER_DEFAULT_IMAGE` and `BEAKER_SANDBOX_IMAGE` in
  `common/constants/infrastructure.py` are updated when a new image is the new
  default. A description claiming "no source changes needed" for an image
  bump is wrong if those constants still point at the old image.
- Do not accept an SDK upgrade on the strength of its changelog alone; confirm
  the behavior the repo relies on still holds (defaults can change).

### 3.7 Documentation or cleanup

- Docstrings stay general. No exact counts ("14 tasks") or implementation
  details that go stale.
- No comments describing temporary or in-progress state.
- README examples still run.

---

## 4. Checks for every PR

**Structure**

- Repeated adapter logic across several task modules is factored into a shared
  base class or helper, so a future fix lands once.
- Helpers live where their consumers can import them without inverting the
  dependency direction.
- Dynamic class generation (`type(...)`, `globals()[...]` injection) is
  replaced by `__init_subclass__` or a single class plus `partial`
  configuration.
- Fields with no consumer in this PR are not added "for later". They land with
  their consumer, or the PR describes them as forward plumbing and serializes
  them.

**Typing**

- Dict literals that cross function boundaries are typed (`TypedDict` or an
  explicit annotation).
- Fixed string sets are `Literal[...]` or validated in `__post_init__`.
- Collections that should not mutate are tuples.
- Generic names (`GroupResult`, `Config`, `Helper`) are made specific.
- `ty` passes without new ignores. An `unresolved-import = "ignore"` override
  is tolerated only for optional dependencies and should not grow.

**Tests**

- New extraction, parsing, and scoring logic has a direct unit test, not only
  an indirect one through a task.
- Every bug the reviewer finds gets a regression test as part of the fix.
- Tests are adapted to match source, not the reverse. Do not ask an author to
  rewrite tests to preserve behavior that the source change intentionally
  drops. Do flag a test that explicitly preserves a misclassification the PR
  claims to fix.
- Do not add or modify tests yourself as part of a review; describe the test
  you want.

**Naming**

- Task names describe the benchmark; variant names describe the modifier.
- Names encode units (`_SEC`) and sources (hosting service as a suffix on
  model presets so prefix filtering is predictable).
- A rename is proposed when a name implies something false (a "long-context"
  suffix on a benchmark that is already long-context).

**Comments and docs**

- Non-obvious tricks (`lstrip("0") or "0"`) get a one-line comment saying why.
- No hard-coded totals in docstrings.

**Process**

- The PR description says what was tested and how. If it lists `make verify`
  or a parity run, spot-check that claim.
- Deferred work is filed as an issue and linked from the PR before merge.
- Potential duplication with another open PR is checked before review effort
  is spent (search open PRs for the same files).

---

## 5. Things a reviewer should not do

- Do not comment on formatting, import order, or line length. Ruff owns those.
- Do not block on nits. Label them as nits and approve if nothing else blocks.
- Do not request Ai2-specific conveniences in source. The repo is public.
- Do not ask for a preset or harness config when CLI overrides on the default
  harness are sufficient.
- Do not reintroduce regimes or any other bundled, opaque configuration layer.
  Everything composes from variants.
- Do not accept "the numbers look reasonable" as parity evidence. Ask for the
  reference number beside the new one.
- Do not merge on the author's behalf unless asked. After approval, tell the
  author they may merge once CI passes and any rebase is done.
- Do not speculate. If you have not reproduced or read the code path, phrase
  it as a question and say what you were unable to check.

---

## 6. How to write the review

**Choose the delivery format for the findings.** Three options are available,
and the reviewing agent decides which fits the PR:

- **One summary comment only.** Best when findings are architectural, span
  several files, or the author is likely to paste the review into their own
  agent without a GitHub integration.
- **Inline review comments only.** Best for a handful of line-specific defects
  where the diff context is most of the explanation.
- **Both.** Usually the strongest choice for a substantial PR: a summary
  comment carrying the verdict, the ranking, and the merge instruction, plus
  inline comments on the exact lines for findings that benefit from context.

Whichever format is chosen, the summary comment (when present) must stand on
its own: every finding, including ones also left inline, appears in it with a
`path/to/file.py:LINE` reference, so a person can copy one comment and hand the
complete review to an agent.

**Disclose that the review is agentic, on every surface it appears.** The
review must never be mistaken for a person's. Use exactly this line as the
first line of the review body or summary comment, followed by a blank line:

```
> This review was generated by an agent following `REVIEW_GUIDE.md`. It is not from a human reviewer. Reply in the PR thread and a maintainer will follow up.
```

When inline comments are posted as part of a GitHub review that has a body,
the disclosure in the body covers them. When inline comments are posted
without a summary body, begin each inline comment with the short form:

```
> Agent review per `REVIEW_GUIDE.md`.
```

The rest of the review can be as friendly as a human review. The disclosure
does not change the tone; it changes who the reader understands is speaking.

**Open with the overall verdict in one line**, warmly and honestly. The house
style is brief: "Looks good overall! A few things we should address before
merging." or "lgtm! Two non-blocking questions." A change request still opens
with what is good.

**State the ranking explicitly** when there are several findings: "I found four
issues. The first two affect persisted result correctness and task identity,
so they should be fixed before merge. The other two affect prompt fidelity and
documented failure behavior."

**Each finding**, whether inline or in the summary, contains:

1. A `path/to/file.py:LINE` reference on its first line (in the summary
   comment; inline comments already carry their location).
2. The concrete failure: input or state, then the wrong output. ("With
   `max_hard_failure_rate=1.0`, an all-failed task exits 0 with no metrics.")
3. Why it matters in this repo's terms: persisted results, task hash, parity,
   silent zeros.
4. The requested change, phrased as a question when it is a judgment call
   ("Could we...", "Should we...") and as a request when it is a defect
   ("Please serialize both fields and add a test asserting that changing
   either one changes the task hash.").
5. A code suggestion in a fenced block when the fix is short and you are
   confident in it.

In the summary comment, findings are numbered under `Blocking` and
`Non-blocking` headings, and nits go in a final `Nits` section, one line each,
prefixed "Nit:".

**Own the framework's faults.** If the bug is caused by a default or interface
in the existing code rather than the PR, say so ("This is a deficiency in the
current `compute_instance` default") and propose the framework fix alongside
the local one.

**Ask for evidence, not reassurance.** Runtime before and after, a parity
table, a hash comparison, a memory figure for the full suite.

**Ask for a follow-up issue** when you accept something imperfect for
expediency, and name what the issue should say.

**Close** with the merge instruction: what must change before merge, and that
the author may merge after that once CI is green.

Template for the summary comment:

````
> This review was generated by an agent following `REVIEW_GUIDE.md`. It is not from a human reviewer. Reply in the PR thread and a maintainer will follow up.

<one-line verdict>

<optional: ranked summary of findings>

### Blocking

1. `src/olmo_eval/path/file.py:123`
   <failure> <why it matters> <request>
   ```python
   <optional suggestion>
   ```

2. ...

### Non-blocking

1. `src/olmo_eval/path/file.py:456`
   <question or suggestion>

### Nits

- Nit: ...

### Before merge

<what must change> <merge instruction> <follow-up issue asks>
````

If the review finds nothing blocking, keep the disclosure line, the verdict,
and any non-blocking items or nits, and say the author may merge once CI is
green.

---

## 7. Decisions already made

Do not reopen these in review. Point authors to them.

- **Regimes are removed.** Model- or paper-specific configurations compose from
  variants.
- **Bug-compatible ports, versioned fixes.** Match the reference, then ship
  `:v2`. No backporting to oe-eval.
- **Two BPB formulations coexist** (byte-weighted, instance-averaged) until a
  deliberate canonicalization.
- **Progress and telemetry swallow exceptions; everything else propagates.**
- **Dependabot is disabled** as too noisy. Dependency bumps are deliberate PRs
  with canary runs.
- **`backend` is called `scaffold`.**
- **Missing logprobs are `-inf`.**
- **Default harness plus CLI overrides** beats per-task harness presets.
- **Task spec direction** is `<dataset>:<format>:<metric>`; new variants should
  fit that shape.

---

## 8. Known pitfalls to check for by name

These have each caused a real bug or near miss in this repository. Check for
them explicitly when the change touches the relevant area.

| Pitfall | Where it hides |
|---|---|
| Output-affecting field not in `TaskConfig.to_dict()` | Any new `TaskConfig` field or env-var read |
| Metric inheriting `compute_instance` fallback | Any new `Metric` subclass |
| Metric on 0 to 100 scale named like a 0 to 1 metric | Any new metric named `recall`, `accuracy`, `f1` |
| Multi-sample outputs averaged instead of aggregated per config | Any metric iterating `r.outputs` |
| Scorer dedup by `name` dropping a differently configured instance | Any scorer with a mode flag |
| Blanket `except Exception` inside a judge or scorer | Any judge call |
| Judge parse `KeyError` escaping a `ValueError` handler | Label mapping dicts |
| Division by zero on an empty subset under `limit` | Subset metrics |
| Whole-file load before `limit` | Any custom loader |
| `snapshot_download` of a whole repo per task | HF-backed loaders |
| Import-time heavy dependency in a shared package | New scorer modules |
| `/weka` or workspace default paths | External evals, vision, long-context |
| Provider kind missing from an allowlist | New provider kinds |
| `SamplingParams` change handled by one provider only | `common/types/base.py` |
| `InstrumentedProvider` failing an `isinstance` check | External evals |
| Module shim with `__spec__ = None` breaking `find_spec` on second call | Provider checkpoint loaders |
| Concatenated embedding cache copying the whole table | VLM providers |
| Decoded images in multiprocessing queue items | Vision tasks |
| Unpinned git dependency or branch reference | `pyproject.toml`, external eval clones |
| Mutable model alias as a judge default | Judge configuration |
| Task name colliding with a suite name | New registrations |
| `<think>` content leaking into answer extraction | Reasoning-regime tasks |
| Sign stripped from negative integers by a number regex | Math extractors |
| Prompt suffix hard-coded in `process_doc` | New generative tasks |
| Fixed few-shot builder shared between MC and generative variants leaking labels | Few-shot sources |
| All-failed run reported as Success with 0.0 | Gates, hard-failure thresholds |
| Gate error dropped when an instance summary is present | Storage writers |
| Default image constants not updated on an image bump | `common/constants/infrastructure.py` |
| `split=` argument overwritten by class-level split | `DataSource` construction |
| `answer_prefix=""` and other defaults restated explicitly | Formatter construction (nit) |

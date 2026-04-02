# No-GPU Oracle-vs-Corrupted Task Tests (Real Dataset Instances)

## Summary
Add deterministic no-GPU task-pipeline tests for:
- `humaneval:bpb`
- `lab_bench_litqa2:mc`
- `minerva_math_algebra` (default end-task metric)

Each test follows the same pattern:
1. Build real requests from loaded task instances.
2. Build perfect synthetic responses.
3. Build corrupted synthetic responses.
4. Score both and assert corrupted performance is lower.

Execution mode:
- Tests are **opt-in** and non-blocking by default.
- They run only when `RUN_REAL_DATASET_TESTS=1` is set.
- Without that env var, tests are collected and skipped.

## Placement
- Test file: `tests/evals/tasks/test_oracle_no_gpu_pipeline.py`
- Rationale: validates concrete task formatting/extraction/scoring contracts.

## Data source policy
- Use **real dataset instances** loaded from `task.instances` for all three tasks.
- Do **not** use synthetic fallback instances if dataset loading fails.
- Keep these checks non-blocking for normal CI via skip-by-default gating.

## Task-specific behavior

### humaneval:bpb
- Request type: `LOGLIKELIHOOD` with one continuation.
- Perfect: continuation text with mild token logprobs.
- Corrupted: same continuation text with degraded logprobs on selected tokens.
- Metric comparison: BPB quality uses `-bits_per_byte` (higher is better).

### lab_bench_litqa2:mc
- Request type: `LOGLIKELIHOOD` with one continuation per choice.
- Perfect: highest `total_logprob` at gold choice index.
- Corrupted: highest `total_logprob` at deterministic non-gold index.
- Metric comparison: accuracy (`accuracy.multiple_choice`).

### minerva_math_algebra
- Request type: `COMPLETION` (no `request.continuations`).
- Perfect: generated text uses the real instance `solution_text` when available.
- Corrupted: generated text is parseable but intentionally wrong.
- Metric comparison: end-task accuracy (`accuracy.minerva_math_flex`).

## Explicit keyword arguments
All relevant method calls should use explicit keyword arguments, e.g.:
- `get_task(spec=..., config_overrides=...)`
- `task.process_doc(doc=..., index=...)`
- `task.format_request(instance=...)`
- `task.score_responses(responses=...)`
- `task.compute_metrics(responses=...)`

## Acceptance criteria
- Plan file exists in `plans/`.
- New test file contains 3 async tests (humaneval, labbench mc, minerva).
- Each test checks request shape and `corrupted_quality < perfect_quality`.
- Tests are skipped by default and do not block merges:
  - `uv run pytest tests/evals/tasks/test_oracle_no_gpu_pipeline.py -v`
- Real-dataset run passes when explicitly enabled:
  - `RUN_REAL_DATASET_TESTS=1 uv run pytest tests/evals/tasks/test_oracle_no_gpu_pipeline.py -v`

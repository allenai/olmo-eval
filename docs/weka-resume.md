# Resume Beaker evaluations from WEKA

## Scope and reference

Replace the transfer of active evaluation state through Beaker result datasets with
direct writes to a claimed WEKA volume. Preserve the per-instance recording, identity
checks, failure accounting, and dashboard lifecycle introduced by PR #519
(`1b9fe6f3`). No task spec, prompt, metric, or task hash changes. External runners are
outside this change.

The pilot uses `ai2/olmo-eval` (500 GB, volume ID `v_X2K0I2O2N3`). The debug workspace
has a read-write claim. Claims authorize a workspace; each job must still mount with
`source.volume` and `mode: readwrite`. The parent workspace does not automatically
have a claim. Never create claims implicitly during launch.

## Implementation sequence

1. Qualify native volume mounts with a bounded CPU probe. Check file and directory
   fsync, rename visibility, SHA-256 integrity, advisory locking, and release after a
   killed process. Repeat across jobs and record their actual nodes.
2. Add opt-in `--resume-volume` / YAML `resume_volume` to the task launcher. Gantry
   prepares the existing environment and saves its spec. Add the native volume mount
   to that spec and submit with the volume-capable Beaker CLI. Keep credentials in
   secret mappings. This bridge is necessary until SDK/Gantry accept native mounts.
3. Wrap the task command in a shared-run lifecycle. Allocate a unique directory per
   submission, stable across replacement jobs, and put the complete output directory
   there. Take an advisory lock before touching manifests or progress. Reclaim the
   predecessor's progress locks only while holding this outer lock. Leave only a
   small receipt in Beaker `/results`, with no symlink to the volume.
4. Refresh recovery files during active runs. Use one `tmp-7d` ancestor, never nested
   `tmp-*` directories. Refresh every file's mtime, not just a heartbeat: prune uses
   per-file mtime and reading files does not extend their lifetime. Preserve completed
   output for the inactive retention window; dashboard upload remains the durable
   result destination. Verify actual scan/prune registration with storage engineering.
5. Detect incomplete recovery state, publish completion atomically, and avoid
   repeating completed runs after a restart. Retain recovery data on failed upload.
   Stop rather than silently mixing expired, corrupt, or incompatible state.
6. Add public-path regression tests and document access, retrieval, retention, and
   recovery. Keep dependency and image upgrades out of this feature branch.

## Validation gates

- Local: existing resume suite plus lifecycle subprocess tests (SIGTERM/SIGKILL,
  simultaneous writer, stale lock recovery, expiry, completion, failed child), native
  spec serialization/submission tests, CLI/YAML wiring, unchanged task hashes.
- Required checks: `uv run ruff check src/ tests/`,
  `uv run ruff format --check src/ tests/`, `uv run ty check src/`,
  `uv run pytest tests/ --ignore=tests/integration -v`.
- CPU Beaker: explicit debug workspace, bound budget, zero GPUs, short min-runtime
  where allocated, bounded timeout. First qualify storage. Then run the actual
  wrapper with a local-fixture deterministic eval and real preemption/retry. Verify
  run ID, completed-instance reuse, failure counts, exactly one final row per instance,
  intact finalized tasks, and small Beaker receipts.
- Scale: compare 1 GB and 10 GB WEKA state across replacements, checksum it, and
  separate scheduler/finalization delay, Gantry setup, and restore time. Cap all
  scratch data well below the volume's 500 GB capacity.
- GPU Beaker: pushed revision, one H100, pinned small model, limited task, bounded
  timeout, and test-only preemption. Use `--no-upload` for development validation.
  Check restored output exactly; newly generated text may differ with batch composition.
- Retention: unit-test age handling immediately. Actual `tmp-1d` pruning requires
  observation over a scan/deleter cycle; never claim this verified from naming alone.

## Pilot incident handling

Record every volume failure, including failures followed by a successful retry: volume
ID, experiment/job ID, node, operation, events, and retry outcome. Report in the team's
storage channel. If AI Infra investigation or a change is needed, file feedback and
assign Sarah Tattersall. Do not hide mount failures by falling back to local storage.

## Rollout

Start with explicit `--resume-volume ai2/olmo-eval` in the debug workspace. Promote the
default only after storage, recovery, upload, and retention gates pass and production
workspace claims are verified. Existing submissions keep their original storage.

## Execution evidence

Implementation and experiment receipts are added here as each gate completes. A
submitted experiment is not a verified result.

- Base revision: `1810320a` on `main`. Beaker CLI upgraded to v1.5.346; SDK and
  Gantry dependencies remain unchanged.
- Native-volume POSIX probe: experiment `01M6MK1WTY0GCKCX980Q4N63NY`, job
  `01M6MK1WV9D0XT3JSA6X6PS195`, Ceres node `ceres-cs-aus-450`. Completed with exit
  zero. File integrity, atomic rename, file/directory fsync, lock exclusion, and
  release after SIGKILL all passed. The probe requested zero GPUs; Ceres reserved
  a GPU slot as its scheduling unit. No model ran.
- Shared wrapper plus local offline fixture: 80/80 scored, accuracy 1.0, exit zero.
  The scoring delay is intentional so a real Beaker preemption can interrupt it.
- Focused baseline regressions: 401 passed. New lifecycle tests cover independent
  UUIDs, simultaneous resumes, SIGTERM forwarding, retained child ownership after
  supervisor SIGKILL, stale PID-lock cleanup, command changes, retention refresh,
  expiration, segment integrity, completed-artifact integrity, and YAML wiring.
- Full-suite attempt initially found one test-only monkeypatch resolution error;
  after correction, final verification passed: 5,165 passed, 40 skipped, 20 subtests.
  Ruff lint, format checking, and `ty check src/` also passed. Commands used the
  CI-compatible `uv run --frozen --no-group vllm` prefix.

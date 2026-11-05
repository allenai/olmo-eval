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

### Beaker results on `1190cef0`

| Gate | Experiment | Observed result |
| --- | --- | --- |
| Second-host POSIX probe | `01M6MMD629REA4BG95QKBRHQG9` | Passed on Ceres 447, reading the payload originally written on Ceres 450 |
| Actual preemption, 10 GiB state | `01M6MMB6GF0EXY8TVW8NJXFYNA` | Exit 143 after `beaker job preempt`; replacement restored 49/80 instances, ran 31, and finished with exit zero |
| Uninterrupted fixture control | `01M6MMMK1ZZESXVCNVBD9YHVK4` | 80/80 scored, exit zero |
| Artifact comparison | `01M6MMT8AJ6PNARVGE6G0HPAER` | All 80 request rows and all 80 prediction rows byte-identical to control; scores equal; no duplicates; completion checksums valid |
| Cross-node advisory lock | `01M6MMX4MDCBGCFN602461GBMD` | Writer on Ceres 451 excluded reader on Ceres 442; reader acquired after release; payload identical |
| Pinned-model interruption/retry | `01M6MMCM0H7B64VMGY4DNAHRTH` | One H100, Qwen2.5-0.5B-Instruct; intentional SIGTERM at 16/32; retry restored 16, generated 16, finished 32/32 |
| Model artifact audit | `01M6MN43XT53P0A2JZ44Y6FS5Q` | 32 unique predictions and requests, zero failed instances, correct model/code revisions, all completion checksums valid |

The fixture moved from Ceres 442 to Ceres 441. Both attempts verified the same 10 GiB
payload SHA-256, `e4c296c41d3a9605f0348961e41ed835b5f42c220751ed2269a726951546c935`.
The replacement retained dashboard run ID `b441a482b60f`. Its Beaker output held a
219-byte receipt plus small Gantry setup logs. The bulk payload, progress, requests,
predictions, metrics, and provider logs remained on WEKA.

The real-model run used model revision
`7ae557604adf67be50417f59c2c2f167def9a775`, `gsm8k` with `limit=32` and
`max_tokens=128`, a 10-minute min-runtime, a 30-minute per-attempt timeout, and one
retry. Its dashboard run ID remained `cb1a46fbe810`. The final score was 0.3125;
6/32 generations reached the token cap. This is a storage/recovery canary, not a
benchmark-quality estimate. The model interruption was the existing debug SIGTERM
hook; the fixture above exercised an actual Beaker preemption.

All evaluation trials used the explicit debug workspace and its bound budget, with
`--no-upload` for development validation. A further local regression simulates a
failed upload and verifies that retry keeps the same run ID and finalized records.
The added YAML-export and upload tests pass in the focused suite (36 selected tests).
GitHub CI for `1190cef0` passed, including Python 3.12 and 3.13.
The final full local suite, including the added upload regression, passed with
5,166 tests, 40 skipped, and 20 subtests. Final lint, formatting, and type checks passed.

### Scheduling and failures

The CPU-only-cluster probe `01M6MKG5N83S8YMFT95DYKDXZY` never started because both
clusters were full; it was canceled when the allocated Ceres probe replaced it.
The first two-host coordination test `01M6MMJHFSS76F3JDAYD7GTDAA` could not schedule
its writer before the reader's peer-wait deadline. It was stopped and replaced by
the passing test above, with zero-GPU requests that avoid reserving GPU slots for
tiny CPU operations. That timeout was a scheduling/test-coordination failure. No
mount, volume I/O, or distributed-lock failures were observed. No volume incidents
needed escalation.

### Remaining gates before changing the default

- Observe a real Starfish prune cycle and confirm scan/prune coverage for the new
  volume. The audit seeded a disposable fixture at
  `/eval-cache/qualification/tmp-1d/expiry-20261104/`: `eligible.txt` has an mtime
  three days old and `fresh.txt` was newly created. Naming and unit tests do not
  verify that the live deleter has run. Scans can delay deletion by a day or more.
- Verify a live dashboard upload from the WEKA directory against the designated
  test deployment. Client unit tests and failed-upload recovery passed; the Beaker
  canary deliberately did not publish development results to production.
- Verify read-write claims in each production workspace before enabling it there.
  Keep the explicit `--resume-volume ai2/olmo-eval` pilot option until these gates pass.

The small-model canary and deterministic workload qualify the shared-storage path.
Long generation, agent, sandbox, and other engine canaries are additional rollout
coverage; external runners remain unsupported by this option.

## Independent review and fixes

Two GPT-6 Astra agents independently reviewed the change after the initial trials:
one for correctness and one for adversarial storage/concurrency behavior. They found
gaps the successful normal-path tests did not exercise:

- An interruption between segment rename and inventory publication could block
  recovery or leave replayed records outside checksum protection.
- An early interruption during task preparation could replace the dashboard run ID.
- Missing output or entire progress directories could be treated as fresh state.
- Directory traversal errors could be hidden during retention refresh or completion.
  A failed segment-directory listing could also reset numbering and overwrite data.

The fixes make the durable inventory authoritative. An empty inventory is published
before the first data segment, and the initialized stream is registered in the parent
session before any data commit. Replay and compaction use only committed, validated
segments. Consumed filenames are never reused after an uncertain index publication,
including a failure after index replacement. Expected output/stream directories may
not disappear and silently restart an evaluation. Shared manifests retain identity
before the first instance is recorded.

Retention and completion now enumerate directories strictly and propagate I/O errors.
Live refresh tolerates files or directories disappearing during normal atomic rename
or rotation; completion stays strict after the child exits. Fault-injection tests cover
publication and compaction boundaries, uncertain commits, missing subtrees, traversal
errors, and the temporary-file rename race.

The same two agents re-reviewed their findings and the follow-up fixes. Both reported
no remaining scoped blockers. The adversarial reviewer passed 39 lifecycle tests; the
correctness reviewer passed 13 focused regressions. No reviewer modified source or
submitted remote jobs.
After the fixes, the full local suite passed with 5,186 tests, 40 skipped, and 20
subtests. Ruff lint, formatting, and type checking passed.

Volume claims must be restricted to mutually trusted writers. Checksums are corruption
checks, not authentication, and progress decoding supports pickle. Per-run UUIDs and
locks prevent accidental collisions; they do not isolate mutually hostile jobs that
can write the same volume under the same operating-system identity.

### Post-review Beaker qualification on `9008bd88`

- `01M6N06K5Y44NJXAE9YW55863Q`: uninterrupted 80-instance fixture with 1 GiB state,
  refreshing retention every 50 milliseconds to stress concurrent publication.
  Completed with exit zero and no retention errors.
- `01M6N0CQ51R9FNKVB6NSP2J64N`: same rapid-refresh workload with the debug midpoint
  SIGTERM hook and one retry. First attempt exited 143; replacement restored 40/80
  instances, scored the remaining 40, and finished with exit zero.
- `01M6N0HDHZR9XM1WW1HE1DA3SH`: independent read-only audit confirmed byte-identical
  requests and predictions versus the uninterrupted control, 80 unique instances,
  zero failed instances, correct source revision, valid artifact checksums, durable
  session/stream registration, and checkpoint cleanup after completion. The same run
  ID, `c24d494ff432`, covered both attempts.

Both prune-fixture files still existed at this audit. Live prune-cycle verification
and live test-dashboard upload remain separate rollout gates. The post-review checks
used zero GPUs; the earlier pinned-model canary exercised the model execution path.

# Size-peer evaluation runs

Shared handoff for Codex and the main Claude session. Updated October 5, 2026,
11:39 AM America/Los_Angeles. Codex session:
`codex://threads/01a10d3f-edd8-7780-8c2b-93905c0aca77`.

## Current status

Evaluation setup is on `codex/ling-peer-evals`; the submitted smoke jobs pin
`e9d09e966adda8bff2481e35f38072fe0a016dd3`. The branch combines
fail-loud-all-errors (#418), reasoning-max-tokens (#420), and generation-counts
(#421), without merging their GitHub PRs. Two GPU smoke jobs are submitted;
there are no capability results yet and full benchmark runs have not launched.

| Peer | Beaker experiment | Resources | Observed status at 11:39 AM |
|---|---|---|---|
| Ling 3.0 tiny | [01M46MZMXFQHJ7VVAJETEH0E5V](https://beaker.org/ex/01M46MZMXFQHJ7VVAJETEH0E5V) | Holmes, 1 B300 | Smoke completed; acceptance checks passed |
| LFM 2.5 8B A1B | [01M46N0BHH7P7X5HGZ2716WHHD](https://beaker.org/ex/01M46N0BHH7P7X5HGZ2716WHHD) | Jupiter, 1 H100 | Smoke completed; acceptance checks passed |

Both use `ai2/olmo-instruct`, normal priority, a 30-minute minimum runtime,
a two-hour task timeout, and no automatic retries. Job IDs are respectively
`01M46MZN1CNJAXCXY4W5QF5GBX` and `01M46N0BNANTDBCZW6C52JYNRP`.

Local verification after the initial follow-up: **2,980 tests passed, 35 skipped**;
the final focused checks pass (17 peer tests and 50 runner tests). Ruff checks and
formatting passed. Type checking has one existing unused suppression, detailed below.
An independent Claude review found follow-up work in bootstrap provenance and
smoke acceptance checks. The follow-up adds a strict smoke acceptance gate,
structured sampling/dirty-checkout/job provenance, a working-directory check,
eval package capture, and periodic serving logs; it removes an unlocked
transformers installation from the external-client environment. These changes
apply to subsequent runs; submitted smoke jobs use the original commit above.
Review follow-up also rejects empty smoke outputs, removes stale acceptance
files before launch/validation, resolves Git provenance from the package path,
requires clean source for full runs, and derives Ling's serving revision from
the same peer definition as its manifest.

LFM smoke artifacts were downloaded to `/tmp/lfm-smoke-results` and checked
against the new acceptance gate: 12 generations, 12 cap hits, 12 unclosed
thinking traces, zero empty outputs, zero unknown finish reasons, and zero
request failures. This confirms the runtime and accounting path only.
Ling artifacts are at `/tmp/ling-smoke-results`: 12 generations, 12 cap hits,
zero empty outputs, zero unknown finish reasons, and zero request failures.
Its zero `unclosed_think` count is a lower bound because the template opens the
trace; the cap-hit count still identifies the incomplete generations.
Submitted job specifications are checked in under `docs/peer-smoke-specs/`.

Next: launch core/knowledge passes using the reviewed follow-up. MiniCPM is
configured but has not launched. Preserve the distinction
between the common T=1 contract and each vendor's recommended settings.

## Claude coordination

This file is the shared handoff on the Mac at
`/Users/abhishekr/repos/olmo-eval/docs/peer-evals-2026-10-05.md`.
Claude can append dated requests or decisions under **Claude notes**; Codex
will read them during this evaluation work and acknowledge handled entries
under **Codex acknowledgments**. Preserve the other session's notes when
editing. File changes are not push messages: they are seen when the other
session reads the file. No background polling has been configured.

Do not switch or reset this checkout while Codex is working. Use a separate
checkout for overlapping code changes, and record the branch or commit here.

### Claude notes

No notes yet.

### Codex acknowledgments

No notes yet.

Prepared from the secretary capability-targets report and Claude's
`progress-check` session (`a855508d-ef0a-4ae4-be30-ba59d234f2ea`).

## Harness prerequisites

This evaluation branch integrates three open PRs at their current heads:

| PR | Head | Reason |
|---|---|---|
| fail-loud-all-errors (#418) | `4fa78428` | Failed batched requests must fail the run |
| reasoning-max-tokens (#420) | `848c1eca` | IFBench must not inherit the 2,048-token instruct budget |
| generation-counts (#421) | `9ce98f0d` | Save cap-hit, empty, unclosed-think and unknown-finish counts |

No GitHub PR has been merged. The combined branch is `codex/ling-peer-evals`.
The two summary-layout tests use a wider console to keep task names readable
after the added generation-count column.

The other requested PRs are not prerequisites for this battery:

- answer-extraction (#417): repairs the OMEGA hill-climb scorer.
- omega-dev-tier (#419): adds the OMEGA in/out score gap.
- hillclimb-dev-suite (#422): combines the existing dev tasks and guards;
  carries #419 and #420 and depends on them. It uses GPQA-main and LCB v3,
  while these comparisons need GPQA-Diamond and eventually exact LCB v6.

AIME 2026 is already on main; the secretary report's missing-task note is stale.
BFCL v4 non-web (#340), chat regimes (#429), HLE (#440), and AA-Omniscience
(#430) remain separate prerequisites for extending the battery.

## First-pass contract

`uv run python -m olmo_eval.launch.peers` runs fixed revisions of Ling-3.0-tiny,
LFM2.5-8B-A1B, and MiniCPM5-1B. Every request uses thinking on, T=1,
top-p 0.95, top-k 20, and one sample. Core and knowledge passes have a
32,768-token output budget and a 65,536-token serving context. Hard request
failures are gated at zero. Requests, predictions, resolved configs, generation
counts, and a commit/model-revision manifest are retained.

Core: AIME 2025, AIME 2026, HMMT Feb 2026, GPQA-Diamond CoT, IFEval and
IFBench (`ifeval_ood`). Report IFEval strict and loose separately. Math is
single-draw accuracy, not avg@32 or pass@32. Knowledge is a separate full
MMLU-Pro CoT pass, weighted over categories.

Smoke checks run two examples per core task with 128 output tokens. They
deliberately exercise truncation accounting and must never be read as
capability measurements.

The acceptance gate requires two saved/processed generations per task, zero
failures or empty outputs, and known finish reasons. Cap hits are allowed for smoke. In full
runs, report cap hits and incomplete traces alongside scores: the current
thinking stripper retains an unclosed trace, so a truncated reasoning response
can reach the scorer. Do not interpret that as a completed final answer.

```bash
uv run python -m olmo_eval.launch.peers --peer lfm --phase smoke \
  --output-dir /results/smoke --dry-run
uv run python -m olmo_eval.launch.peers --peer lfm --phase core \
  --output-dir /results/core
```

Ling's architecture is absent from the pinned vLLM 0.19.1 registry. Its
wrapper, `scripts/peers/ling_tiny.sh`, starts the documented SGLang runtime
and connects through the existing OpenAI-compatible `vllm_server` client.
It preserves raw thinking text by leaving the reasoning parser unset and
keeps SGLang's GPU packages separate from the eval environment.

Runtime image:
`lmsysorg/sglang@sha256:0e259c844df22da2ba8969ae7a310f64580f554357de519e5f82fd73ff13346c`
(amd64 digest of `dev-Ling-3.0-tiny`, CUDA 13.0.3; use compatible nodes).

## Remaining passes

- Recommended settings: Ling matches this sampling contract. MiniCPM's
  thinking recommendation is T=0.9/top-p 0.95; LFM recommends T=0.2,
  top-k 80, repetition penalty 1.05. The current API client does not expose
  repetition penalty, so an exact LFM recommended-settings run needs that
  support first. The common T=1 run is not a vendor reproduction.
- Math repeated draws to reduce sampling noise, with an appropriate repeated-
  sample analysis rather than treating draws as independent questions.
- Full MMLU-Pro; exact LCB v6; BFCL v4 non-web and thinking-off tool protocol.
- LongBench v2, RULER, AA-LCR/MRCR at an explicit context/truncation contract.
  The first-pass 64K context is not a long-context measurement.
- Thinking off, chat, factuality and safety guards.
- Medium peers after the small serving paths pass smoke checks.

## Validation

The three prerequisite PRs' targeted tests pass (78 tests). New tests resolve
every peer command through the actual CLI and runner override machinery,
including suite propagation and bounded smoke generation. Shell syntax and
CLI dry runs are checked. Local type checking reports one existing unused
suppression in `harness/sandbox/executor.py:802`; it is unrelated to this work.

The live job IDs and last observed statuses are in **Current status** above.

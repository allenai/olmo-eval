# Size-peer evaluation runs

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

GPU job IDs and observed statuses are recorded here after submission.

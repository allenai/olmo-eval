# Qwen routing-policy sweep launch record

Launched on 2026-07-17 after the parameter matrix in
`qwen_routing_policy_sweep_plan.md` was approved.

## Placement and shared configuration

- Beaker group:
  [adaptive-experts-qwen-routing-policies-20260717](https://beaker.org/orgs/ai2/workspaces/holmes-testing/groups/01KXQ3H1TN88E0YTXWKTNFND56)
- Workspace: `ai2/holmes-testing`
- Cluster: `ai2/jupiter`
- Priority: `urgent`
- Model: Qwen3-30B-A3B hybrid-thinking (`qwen3-30b-a3b`)
- Suite: `adaptive_experts:hybrid_pilot` (MATH-500, GPQA Diamond, IFEval OOD)
- Generation limit: 32,768 tokens
- Serving: four independent TP=1 vLLM engines per evaluation
- Full evaluations: 40, with 40 unique run tags
- Immutable source snapshot:
  `/weka/oe-adapt-default/jacobm/olmoe3/adaptive-compute/.run_snapshots/olmo-eval/ca1cd217b47ac53ab70d6fa2bb176522b1051f7a95de73b02e392050d749867e`
- Source hash:
  `ca1cd217b47ac53ab70d6fa2bb176522b1051f7a95de73b02e392050d749867e`
- Append-only experiment ledger: `notes/beaker_jobs.jsonl`

The required Hugging Face and W&B secrets already existed in `ai2/holmes-testing`; no secret
copy was necessary.

At the post-launch audit on 2026-07-17 04:14:00 UTC, 8 full evaluations were running and 32 were
queued. None had failed. The audit verified all 40 expected tags, source hashes, tasks, placement
settings, priorities, and policy parameters.

The first running full job logged the intended `temperature(exponent=0.5)` hook and all four vLLM
providers reached ready state in about 109 seconds. The remaining conditions may stay queued until
Jupiter capacity becomes available.

The matrix launcher is `scripts/adaptive_experts/launch_qwen_routing_policy_sweep.sh`. It is a
submission script, not an idempotent reconciler: do not rerun `--stage full` wholesale. Relaunch an
individual failed condition through `scripts/adaptive_experts/launch.sh` with a new run tag so the
ledger and result aggregation can distinguish the replacement.

## Smoke gate

Seven intended code-path conditions passed. There are eight ledger rows because the original
fixed-rank-profile smoke failed and was replaced after a code fix.

| Condition | Beaker experiment | Outcome |
|:--|:--|:--|
| Temperature `p=0.5` | [01KXQ3H60VVH7Q37DTKNDBKE8T](https://beaker.org/ex/01KXQ3H60VVH7Q37DTKNDBKE8T) | Passed |
| Temperature `p=2.0` | [01KXQ3HEJYJN6YA64GSK0GFQAW](https://beaker.org/ex/01KXQ3HEJYJN6YA64GSK0GFQAW) | Passed |
| Fixed rank profile, original | [01KXQ3HQ0MZSCWGZRF6QFG9Z3A](https://beaker.org/ex/01KXQ3HQ0MZSCWGZRF6QFG9Z3A) | Invalid; failed during CUDA graph capture |
| Fixed rank profile, replacement | [01KXQ3WRP4TDN0A5BPY762GGKF](https://beaker.org/ex/01KXQ3WRP4TDN0A5BPY762GGKF) | Passed |
| Adaptive mass `tau=0.50` | [01KXQ3HZS5J4WY4QA6SGRCJ0AK](https://beaker.org/ex/01KXQ3HZS5J4WY4QA6SGRCJ0AK) | Passed |
| Adaptive mass `tau=0.90` | [01KXQ3J8S9G7Y2RD08AJS4T22P](https://beaker.org/ex/01KXQ3J8S9G7Y2RD08AJS4T22P) | Passed |
| K=4, K=8-reference scaled | [01KXQ3JH7PCANPJ57M9HSN14CB](https://beaker.org/ex/01KXQ3JH7PCANPJ57M9HSN14CB) | Passed |
| K=12, K=8-reference scaled | [01KXQ3JT2DAZFCTJJ4R97S2PBN](https://beaker.org/ex/01KXQ3JT2DAZFCTJJ4R97S2PBN) | Passed |

The invalid fixed-profile smoke attempted a CPU-to-GPU copy while vLLM was capturing a CUDA
graph (`operation not permitted when stream is capturing`). The policy now constructs the fixed
profile directly from graph-safe GPU tensor operations. The replacement smoke and all subsequent
full fixed-profile jobs use the corrected source snapshot.

## Full sweep

Every cell below is one complete three-task evaluation. Replicates use independent sampling but
otherwise identical settings.

| Policy condition | Replicate 1 | Replicate 2 | Replicate 3 |
|:--|:--|:--|:--|
| Temperature `p=0.5` | [01KXQ42DGCEHGED8NQKK19VXW0](https://beaker.org/ex/01KXQ42DGCEHGED8NQKK19VXW0) | [01KXQ43EHVAQFQ13J63NWYQCQV](https://beaker.org/ex/01KXQ43EHVAQFQ13J63NWYQCQV) | [01KXQ44FW8FDPDZE63VY86WDJB](https://beaker.org/ex/01KXQ44FW8FDPDZE63VY86WDJB) |
| Temperature `p=0.75` | [01KXQ42NH7CTQATBYMWHF0NM0R](https://beaker.org/ex/01KXQ42NH7CTQATBYMWHF0NM0R) | [01KXQ43PMXDFWRN9T619492D40](https://beaker.org/ex/01KXQ43PMXDFWRN9T619492D40) | [01KXQ44SVT8HZ4EDKRTQYMJJNJ](https://beaker.org/ex/01KXQ44SVT8HZ4EDKRTQYMJJNJ) |
| Temperature `p=1.5` | [01KXQ42X2X15PW5FR0R6YGGWZA](https://beaker.org/ex/01KXQ42X2X15PW5FR0R6YGGWZA) | [01KXQ43YY4R5AD8KAMRWBTQVJ3](https://beaker.org/ex/01KXQ43YY4R5AD8KAMRWBTQVJ3) | [01KXQ4528S49E0ZYTAYMPCNR1Q](https://beaker.org/ex/01KXQ4528S49E0ZYTAYMPCNR1Q) |
| Temperature `p=2.0` | [01KXQ434WEJWDQ02A3H1X9XPBX](https://beaker.org/ex/01KXQ434WEJWDQ02A3H1X9XPBX) | [01KXQ446XS228H81V4ZMT16Y2N](https://beaker.org/ex/01KXQ446XS228H81V4ZMT16Y2N) | [01KXQ45AESFPZJWZ80V3430NP4](https://beaker.org/ex/01KXQ45AESFPZJWZ80V3430NP4) |
| Temperature `p=1.0` identity sanity | [01KXQ45JVEASADZSAN4022N9DR](https://beaker.org/ex/01KXQ45JVEASADZSAN4022N9DR) | — | — |
| Fixed global rank profile | [01KXQ45VVWBPMWB6ZQEDGAXA5B](https://beaker.org/ex/01KXQ45VVWBPMWB6ZQEDGAXA5B) | [01KXQ464ASNXK6WA4H3EBZ3EYD](https://beaker.org/ex/01KXQ464ASNXK6WA4H3EBZ3EYD) | [01KXQ46DPMV1G1YPERYB9PGVM8](https://beaker.org/ex/01KXQ46DPMV1G1YPERYB9PGVM8) |
| Adaptive mass `tau=0.50` | [01KXQ46QQEHWQCJGYFX63AT6EN](https://beaker.org/ex/01KXQ46QQEHWQCJGYFX63AT6EN) | [01KXQ4705B1EZ04TQG750RYF7G](https://beaker.org/ex/01KXQ4705B1EZ04TQG750RYF7G) | [01KXQ47BDDN5ZNFZSSARX087ZH](https://beaker.org/ex/01KXQ47BDDN5ZNFZSSARX087ZH) |
| Adaptive mass `tau=0.60` | [01KXQ47KRQJA5Q5D8Y0924YC32](https://beaker.org/ex/01KXQ47KRQJA5Q5D8Y0924YC32) | [01KXQ47VSXN1HVZGNMME45QS5M](https://beaker.org/ex/01KXQ47VSXN1HVZGNMME45QS5M) | [01KXQ48443ZK3W0WZB6XRPZ165](https://beaker.org/ex/01KXQ48443ZK3W0WZB6XRPZ165) |
| Adaptive mass `tau=0.70` | [01KXQ48GEZWP2YQ0JMCBJFMA24](https://beaker.org/ex/01KXQ48GEZWP2YQ0JMCBJFMA24) | [01KXQ48SD3VYHE9WF3E91B14YQ](https://beaker.org/ex/01KXQ48SD3VYHE9WF3E91B14YQ) | [01KXQ492413VQ8RAK30XR0425Q](https://beaker.org/ex/01KXQ492413VQ8RAK30XR0425Q) |
| Adaptive mass `tau=0.80` | [01KXQ49AGW00J086PKGM6RPVBW](https://beaker.org/ex/01KXQ49AGW00J086PKGM6RPVBW) | [01KXQ49JFB0YDSH3VS7V6DJB4H](https://beaker.org/ex/01KXQ49JFB0YDSH3VS7V6DJB4H) | [01KXQ49TY39PX1J0RKGQW0GRFC](https://beaker.org/ex/01KXQ49TY39PX1J0RKGQW0GRFC) |
| Adaptive mass `tau=0.90` | [01KXQ4A35E0TYGNX0CB9AYAPTV](https://beaker.org/ex/01KXQ4A35E0TYGNX0CB9AYAPTV) | [01KXQ4AB4FB909R7PD2F5W9QFM](https://beaker.org/ex/01KXQ4AB4FB909R7PD2F5W9QFM) | [01KXQ4AKYAGDX7SERVXVZ9NT0P](https://beaker.org/ex/01KXQ4AKYAGDX7SERVXVZ9NT0P) |
| K=4, K=8-reference scaled | [01KXQ4AWF0AA8K79QQ1X5M47E4](https://beaker.org/ex/01KXQ4AWF0AA8K79QQ1X5M47E4) | [01KXQ4BPFTRA5KBFXN1CRV30BR](https://beaker.org/ex/01KXQ4BPFTRA5KBFXN1CRV30BR) | [01KXQ4CESR2K42NX166TKMJXCH](https://beaker.org/ex/01KXQ4CESR2K42NX166TKMJXCH) |
| K=6, K=8-reference scaled | [01KXQ4B5GQH6C0J0008NQRH74B](https://beaker.org/ex/01KXQ4B5GQH6C0J0008NQRH74B) | [01KXQ4BYH6Y69B01FX8W813TCV](https://beaker.org/ex/01KXQ4BYH6Y69B01FX8W813TCV) | [01KXQ4CPPX0ND1EFA7H5Y6QY9N](https://beaker.org/ex/01KXQ4CPPX0ND1EFA7H5Y6QY9N) |
| K=12, K=8-reference scaled | [01KXQ4BDYY4W3A4A6PNS1SVHMB](https://beaker.org/ex/01KXQ4BDYY4W3A4A6PNS1SVHMB) | [01KXQ4C6W9YZNF6RZY72HYP5C4](https://beaker.org/ex/01KXQ4C6W9YZNF6RZY72HYP5C4) | [01KXQ4CYSACXYQV1PZE2Z4ABRJ](https://beaker.org/ex/01KXQ4CYSACXYQV1PZE2Z4ABRJ) |

The normalized K=4, K=6, and K=12 comparison points are reused from the completed native Qwen
sweep, as approved; the jobs above are only the new K=8-reference-scaled counterparts.

## Fixed-profile provenance and normalization semantics

The fixed profile is:

```text
[0.21562765, 0.16642320, 0.13938729, 0.12015191,
 0.10476987, 0.09296770, 0.08376831, 0.07690407]
```

It was derived from the saved native K=8 router samples with prompt-level, task-level, and then
layer-level balancing, as detailed in the plan.

For the K=4 and K=6 reference-scaled conditions, retained weights preserve their shares from the
native normalized K=8 mixture, so their sum is below one. For K=12, native top-twelve weights are
divided by their top-eight subtotal: ranks 1–8 retain the K=8 scale and ranks 9–12 add mass above
one. This is intentionally different from native K=12 renormalization.

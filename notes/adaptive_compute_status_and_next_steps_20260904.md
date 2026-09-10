# Adaptive compute: status and next-experiment assessment

**Review date:** 2026-09-04 UTC.  
**Scope:** Research assessment of the existing results, experiments, and plans, saved from the conversation. Operational statements reflect that review, not a continuously updated job monitor. Saving this note did not launch jobs or change experimental configurations.

## Overall assessment

We have a substantial inference-time result, some useful mechanistic evidence, and promising training pilots. The two biggest missing pieces are **measured serving savings** and **a convincing training intervention that improves low-K performance**. I would focus the next experiments there.

## What the evidence supports

- **Inference-time K reduction works across families, but the trade-offs vary substantially.** GPT-OSS-120B is our cleanest positive example: its four-task aggregate is essentially unchanged from K=4 to K=2, **83.23 -> 83.10**. Qwen3 and GLM-4.5-Air also have substantial headroom. Nemotron loses more on GPQA than on math. Qwen3.5-35B is an important counterexample: even reference-scaled K=4 leaves HumanEval well below native performance. Qwen3.6 adds another wrinkle: different tasks prefer different K values. See the [cross-model findings](adaptive_compute_handoff_20260824.md).

- **GLM-5.2 extends the evidence to very large models, with a meaningful quality penalty.** Across three seeds, AIME 2026 pass@1 falls from **82.71% at native K=8 to 76.63% at reference-truncated K=4**; mean pass@32 falls from **96.67% to 94.44%**. Typical outputs remain coherent and slightly shorter, but mathematical errors increase, alongside rare repetition and empty-output failures. The gap survives like-for-like parsing comparisons. This is a useful trade-off result, not near-lossless halving. Uncertainty across problems remains substantial because AIME contains only 30 distinct questions. See the [three-seed analysis](glm52_aime_three_seed_analysis_20260902.md).

- **Routing scale is one of our strongest explanatory findings.** Removing experts and renormalizing the survivors changes both information and branch magnitude. Preserving the native denominator often substantially improves quality and generation length, although it is not universally best after SFT. Expert-contribution measurements support this explanation; router-weight concentration alone does not explain family differences.

- **Train-hard/eval-light remains unproven.** At deployment K=4, the OpenThoughts fixed-K=4 and fixed-K=12 SFT models both score **34% on TBLite**. The phased K=12->8->4 model reaches **38%**, which is worth following up, but needs training-seed replication and broader evaluation. These results currently favor investigating cooldown over assuming that higher training K alone helps. See the [SFT results](openthoughts_all_sft_tblite_results.csv).

- **A usable K dial looks more plausible than a learned automatic allocator.** Mixed-K math RL supports K=4/6/8 with competitive performance. However, ordinary K=8 training already produces an almost identical inference-K curve, and the training budgets differ. We therefore have evidence of elasticity, not yet of a special mixed-K advantage. Threshold routing mostly matches the fixed-K frontier; prompt-level minimum-K labels are noisy. The wider mixed-K RL run also developed repetition despite apparently healthy reward. See the [math RL comparison](mixed_k_cost_aware_rl.md).

## Where the recent infrastructure work stands

The Qwen3.5 native-K TBLite controls finished at **53%, 54%, and 59%**, averaging **55.3%**. These use a different starting checkpoint/scaffold from the older SFT experiments, so they are not a direct SFT comparison. Their result root is `/weka/oe-adapt-default/jacobm/tmax-eval/qwen35-default-k8-tblite-20260901`.

Both TMax/Slime runs completed updates and saved fifth-update checkpoints, but failed before sustained training; we have **no demonstrated downstream RL improvement yet**. Also, this integration uses TMax environments with Slime GRPO, not the complete published DPPO recipe. The latest saved production checkpoints are `iter_0000004` for Qwen3.5 v66 and Qwen3.6 v65 under `slime-runs/tmax-vanillux/checkpoints/`; their presence does not substitute for a successful resume test.

DeepSeek's native AIME run finished at **98.23% pass@1**, already making AIME a weak discriminator for that model; reduced-K remains untested successfully. Qwen-397B has a successful shorter-context smoke, but the full K=10/K=5 evaluations and newer startup diagnostic failed.

Finally, we still lack a controlled serving-efficiency study. Some evaluation interventions mask native-width dispatch, and halving routed K does not halve total model compute or resident weight memory.

## Recommended next experiments

1. **Make one convincing quality-versus-real-cost result.** Start with a reliably serving model: Qwen3.6-35B, with GPT-OSS as a useful comparison. Verify true reduced dispatch, then measure native/intermediate/half K under both fixed-length generation and realistic workloads. Keep FLOPs, tokens, latency, and throughput separate. Establish this in one serving stack before expanding the implementation matrix.

2. **Fill informative gaps in the existing inference curves.** GLM K=6 is more immediately useful than another family: it tests whether a moderate reduction avoids much of the K=4 penalty. Add a non-AIME hard workload. Keep Qwen-397B debugging time-bounded and instrumented; its within-family scale comparison remains valuable, but should not block other research.

3. **Run a focused training experiment around the cooldown signal.** First evaluate our existing SFT checkpoints on general capabilities and output reliability. Then replicate low-K, native-K, high-K, and high-to-low training with matched tokens and a low-K, matched-FLOP control. If preparing for subsequent TMax RL, use a consistent native-tool SFT protocol and starting checkpoint throughout.

4. **Turn the mechanistic observations into causal tests.** Compare a robust model and a sensitive one using the same prompts: expert removal versus branch rescaling, plus selective layer interventions. Explaining why robustness differs would add more scientific value than another dense sweep alone.

Keep production TMax RL paused until checkpoint resume and sustained execution pass a clear gate. Automatic K allocation, quantization/speculative-decoding combinations, and larger training studies can follow the first decisive results.

The [three-part framing](adaptive_compute_three_phase_plan_20260825.md) still makes sense, but execution should center on **proving practical savings and improving low-K quality**, with the fixed-K dial as the deployment goal. Fully automatic allocation can remain optional.

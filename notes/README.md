# Experiment notes

- `adaptive_compute_three_phase_plan_20260825.md` is the agreed three-phase research plan:
  inference elasticity and real efficiency, train-hard/deploy-light experiments, and an elastic
  manual K dial with automatic adaptation as a stretch goal.
- `adaptive_compute_handoff_20260824.md` is the detailed project handoff: recovered status,
  verified findings, implementation reality, caveats, repository risks, and the candidate
  three-part tech-report map as of 2026-08-24.
- `adaptive_experts.md` contains the research and execution plan.
- `adaptive_experts_results.md` is the live Markdown score and completion table.
- `project_overview.md` is the living cross-experiment summary: completed and missing work,
  current priorities, plotting decisions, and conclusions.
- `latest_status.md` records the most recent full-ledger Beaker audit, active-matrix states,
  newly collected bundles, and production failures.
- `qwen_routing_policy_sweep_plan.md` contains the implemented routing-policy interventions and
  approved parameter matrix.
- `qwen_routing_policy_sweep_launch.md` records the smoke gate, complete 40-evaluation launch
  matrix, Beaker links, placement, source snapshot, and normalization semantics.
- `qwen_routing_policy_sweep_results.md` is the live partial/final score and collection table for
  that sweep; `routing_policy/qwen3_runs.csv` contains its per-run machine-readable values.
- `qwen_capability_policy_sweep.md` records the IFBench + standard HumanEval follow-up and the
  design of the denser adaptive-mass and reference-scaled expert-count sweeps.
- `expanded_routing_sweeps_launch.md` records the implemented Qwen/GPT-OSS routing definitions,
  smoke tests, complete 132-job production matrix, Beaker groups, placement, and ledger audit.
- `qwen_reference_adaptive_threshold_sweep.md` records the non-renormalized adaptive-mass smoke,
  thresholds, 12-job production matrix, placement, telemetry checks, and Beaker links.
- `adaptive_training_and_rl_plan.md` sketches the staged elastic-K, prompt-level selector,
  token-level controller, and cost-aware DAPO roadmap.
- `qwen35_adaptive_k_sft_plan.md` records the proposed Qwen3.5 Base fixed-K, mixed-K, and curriculum
  SFT study, learning-rate gate, routing semantics, evaluation matrix, and pre-launch decisions.
- `qwen35_35b_a3b_hybrid_sweep.md` records the Qwen3.5 hybrid-model compatibility work, smoke
  gate, and completed 30-job normalized/reference-preserving K=4--8 evaluation matrix.
- `above_native_k_routing_audit.md` audits every retained above-native expert-count sweep and
  explains why historical Qwen3.6 K=10/12 points were silently native K=8 while Qwen3 and
  GPT-OSS above-native points remain valid.
- `qwen35_hybrid_router_shared_analysis.md` records the native-K=8 hybrid-versus-full-attention
  router profile, shared-expert contribution definitions, validation gate, and full analysis job.
- `qwen36_expanded_k_diagnostic.md` records the matched-token native-K=8 versus K=32 router-mass,
  expert-tail, shared-branch, reference-scaled, and normalized counterfactual diagnostic.
- `qwen36_tblite_extreme_reference_sweep_jobs.md` tracks the gated K=64/128/256
  reference-scaled Qwen3.6 TBLite extension.
- `slime_qwen3_base_dapo_plan.md` records the Beaker/Slime baseline design, prepared DAPO assets,
  reward and chat-template decisions, GPU-floor probes, SGLang K override, and live smoke links.
- `mixed_k_cost_aware_rl.md` records the multi-engine K=4/6/8 cost-aware RL implementation,
  staged smokes, topology, and matched prompt-level evidence for a future K selector.
- `slime_qwen3_base_dapo_prompt_k_transitions.md` analyzes exact prompt-level correctness
  transitions across inference K for the three step-350 RL checkpoints.
- `current_suite_expert_sweeps.csv` and the corresponding Qwen/GPT-OSS plots contain the adopted
  MATH-500 + GPQA + IFBench-32k + HumanEval normalized/reference-scaled curves.
- `corrected_dolci_terminal_eos_v2_sweep.md` records the corrected Qwen + OLMo-3 reasoning SFT
  normalized/reference-scaled K=3--12 full-suite sweep, smoke gate, launch placement, partial
  score table, and current plot.
- `qwen_math_response_coherence.md` compares matched Qwen MATH-500 responses across the full
  normalized K=1--16 and K=32 sweep, including completion, formatting, false-start, and
  generated-token diagnostics, fixed prompt cohorts, and hidden-reasoning versus returned-text
  token decompositions for both normalized and K=8-reference-scaled common-correct prompts.
- `qwen_gpqa_ifbench_response_analysis.md` applies the same normalized versus
  K=8-reference-scaled matched-success token decomposition to GPQA Diamond and the task-balanced
  IFBench-32k macro.
- `nemotron3_super_routing_sweep.md` records the normalized/raw K=22 and K=11 LatentMoE sweep,
  the completed normalized K={11,13,15,17,19,21,22} results, plot, routing semantics, smoke gate,
  placement, and launch links.
- `nemotron_router_contribution_analysis.md` records the matched 60-prompt BF16 mechanistic
  profile, aggregation, Qwen comparison, counterfactuals, and layer-resolved results.
- `beaker_jobs.jsonl` is the append-only ledger of submitted Beaker experiments.
- `qwen36_kgt8_replacement_jobs.md` records the Qwen3.6 above-native routing
  correction, exact K=10/K=12 smoke evidence, every corrected capability,
  legacy TB2.0, TBLite, and TB2.1 replacement, and plot-admission rules.
- `../results/adaptive_experts/` contains the downloaded final-sweep result bundles, keyed by
  Beaker experiment ID. This directory is intentionally gitignored.

Each ledger row records the UTC submission time, phase, expert count, checkpoint, task specs,
cluster, workspace, priority, group, experiment ID, URL, source commit/hash/snapshot, result
storage mode, and exact shell command. Query it with:

```bash
jq -r '[.launched_at_utc, .phase, .expert_count, .beaker_experiment_id, .url] | @tsv' \
  notes/beaker_jobs.jsonl
```

The ledger records submission state. Beaker remains the source of truth for live status.

Check every recorded experiment:

```bash
while read -r id; do
  beaker experiment get "$id"
done < <(jq -r '.beaker_experiment_id' notes/beaker_jobs.jsonl)
```

Fetch a result dataset, including predictions and metrics, for one experiment:

```bash
id=01EXPERIMENT_ID
dataset_id=$(beaker experiment get "$id" --format json | jq -r '.[0].jobs[0].result.beaker')
beaker dataset fetch "$dataset_id" --output "results/$id"
```

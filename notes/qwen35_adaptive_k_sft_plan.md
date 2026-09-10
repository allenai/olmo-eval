# Qwen3.5 adaptive-expert SFT plan

Updated: 2026-08-13

## Goal

Test whether supervised fine-tuning can make `Qwen/Qwen3.5-35B-A3B-Base` useful across a range of
active routed-expert counts, and distinguish three questions:

1. Does training at a fixed K specialize the model to that compute level?
2. Can one checkpoint learn a broad K=4/6/8/10/12 operating range?
3. Does a high-to-low or low-to-high K curriculum improve low-K robustness or final quality?

The first study is about model behavior under actual sparse routing. It is not yet an inference
systems benchmark; throughput and wall-clock savings should be measured separately once the model
quality results justify optimized variable-K kernels and batching.

## Candidate data

The leading starting point is
[`open-thoughts/OpenThoughts-Agent-SFT-100K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K),
which contains roughly 94K multi-turn agent trajectories from SWE-Smith, SuperUser, augmented Tezos,
and IssueTasks. It is a sensible match for Terminal-Bench-style terminal work, but it is coding and
software-engineering heavy. Before committing a full run, audit a sample for chat/action formatting,
sequence lengths, duplicated tasks, tool-result masking, and licenses, and evaluate whether mixing in
some broader instruction/reasoning data is needed to preserve non-agent capabilities.

## Routing semantics

- The shared expert remains active in every condition and is not counted in K.
- K=8 is native Qwen routing.
- For K<8, keep the top K of the native top eight and preserve their original router scale
  (reference-scaled / non-renormalized), matching the strongest inference intervention so far.
- For K>8, route the requested number of experts and scale against the native top-eight reference,
  matching our existing K=10/K=12 semantics.
- Apply K before expert dispatch during training so lower K represents real expert-compute reduction,
  rather than computing all expert outputs and zeroing them afterward.

## Stage 0: infrastructure and data smoke

Use a very short run to verify:

- Qwen3.5 Base chat template and tool/action serialization;
- assistant/action-token loss masking, with user/system/tool-observation tokens excluded from loss;
- routed K and shared-expert behavior in forward and backward passes;
- checkpoint save/resume and conversion;
- per-K token/batch telemetry;
- no expert-parallel or replay incompatibility when K changes between optimizer steps.

## Stage 1: small learning-rate gate

Use a fixed 10K-example subset and one seed. Run K={4,8,12} at learning rates
`{1e-5, 2e-5, 4e-5}`. Keep the data order, optimizer, batch size, token budget, scheduler, and all
other settings paired. Select one learning rate that is robust across all three K values, rather
than choosing a different best learning rate for each K and confounding the routing comparison.

## Stage 2: routing-strategy pilots

At the selected learning rate, compare:

1. Fixed K={4,6,8,10,12}.
2. Mixed K: choose uniformly from {4,6,8,10,12} once per optimizer batch, using a balanced shuffled
   schedule so every K receives the same number of batches.
3. Descending curriculum: K=12 -> 10 -> 8 -> 6 -> 4.
4. Ascending curriculum: K=4 -> 6 -> 8 -> 10 -> 12.
5. Mixed-plus-cooldown: balanced mixed-K training followed by a short K=4 cooldown, if the initial
   mixed run is broad but still weaker at K=4.

Use equal total training tokens rather than equal example counts. Log loss and gradient statistics
by K, plus realized routed-expert counts, so differences cannot be explained by a different token
budget or an incorrect schedule.

## Stage 3: full finalists

Promote roughly four conditions after the pilots:

- native fixed K=8 control;
- fixed low-K winner;
- balanced mixed-K;
- best curriculum or mixed-plus-cooldown variant.

Run at least two training seeds for finalists. One seed is sufficient for the screening pilots;
evaluation should retain the project's usual three-run replication where sampling applies.

## Common training settings

- Full-parameter SFT initially; freeze the vision tower and other unused multimodal components.
- Official Qwen3.5 chat template, with the dataset's Terminus-style prompt/action protocol adapted
  consistently for training and evaluation.
- Loss on assistant reasoning/action tokens only unless a format audit gives a concrete reason to do
  otherwise.
- Start with a 32K sequence cutoff and packing, then quantify truncation before deciding whether a
  longer context is necessary.
- BF16, gradient checkpointing as needed, cosine decay with about 5--10% warmup, and a global token
  batch held constant across conditions.
- Save exact source snapshot, dataset revision, optimizer state, routing settings, and per-K batch
  counts with every checkpoint.

The published OpenThinkerAgent recipe is a useful reference point, not a setting to copy blindly:
it used a 32K cutoff, global batch 96, learning rate 4e-5, cosine decay, 10% warmup, and five epochs
for dense Qwen3-32B. The MoE/base-model study should use the small LR gate above before scaling.

## Evaluation design

Cross-evaluate every trained checkpoint at inference K={4,6,8,10,12}; the resulting train-K by
eval-K matrix is more informative than evaluating only at the training K. Primary agent evaluation
should use Terminal-Bench 2.0 (with a smaller fixed subset or Terminal-Bench Lite for pilot gates).
Retain IFBench, MATH-500, GPQA, and AIME to detect capability loss, and add an agent/tool benchmark
such as BFCL or GAIA if the harness is reliable. Compare against both Qwen3.5 Base and the released
Qwen3.5 chat checkpoint under the identical agent scaffold.

Report quality together with realized K, routed-expert compute, output length, task errors/timeouts,
and end-to-end throughput. Avoid interpreting a low-K model as cheaper when its trajectories become
longer enough to erase the per-token MoE savings.

## Decisions to make before launch

1. Dataset audit result and whether to add a general reasoning/instruction mixture.
2. Exact pilot token budget and full-run token budget.
3. Trainer and parallelism stack that can change K before expert dispatch.
4. Fixed validation subset and Terminal-Bench pilot subset.
5. Whether K changes once per microbatch or optimizer batch; optimizer-batch changes are the simpler
   initial comparison.


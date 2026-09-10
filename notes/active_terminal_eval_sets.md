# Active terminal-evaluation sets

The current terminal-evaluation program is intentionally tracked as three
separate populations:

1. Qwen3.6 on TBLite: three matched runs at K=4, 6, 8, 10, and 12. See
   `qwen36_tblite_reference_sweep_jobs.md`.
2. Qwen3.6 on Terminal-Bench 2.1: one matched upper-level baseline run at
   K=4, 6, 8, 10, and 12. See `qwen36_tb21_reference_sweep_jobs.md`.
3. OpenThoughts-Agent SFT checkpoints on TBLite: each model is evaluated at
   its training expert count. See `openthoughts_sft_reference_tblite_jobs.md`
   and `tmax_clean_k8_tblite_jobs.md`.

The two Qwen3.6 populations use the same agent, serving, routing, generation,
and timeout protocol. Their only intended evaluation-protocol difference is
the benchmark dataset.

As of `2026-08-15T17:29:38Z`, K=4/6/8 are complete in both Qwen3.6
populations. The original K=10/12 jobs failed before evaluation because a
shallow HF override did not reach Qwen3.6's nested text router; corrected
K=10/12 jobs using `text_config.num_experts_per_tok` are queued. The three SFT
TBLite evaluations are complete.

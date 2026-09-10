# Above-native expert-count routing audit

Last updated: 2026-08-15

## Why Qwen3.6 K<=8 worked

Qwen3.6's checkpoint router emits its native top eight experts. The
reference-scaled K=4 and K=6 conditions intentionally leave that router at K=8,
then the routing-weight patch zeros ranks 5--8 or 7--8 after routing. Their
historical commands therefore used:

- actual router K=8;
- retained K=4 or K=6; and
- reference K=8.

The harmless shallow `{"num_experts_per_tok": 8}` override did not change the
nested text config, but no change was needed: the checkpoint was already K=8.
Native K=8 likewise required no override.

Above-native routing is different. Experts 9--12 cannot be recovered from a
router that emitted only eight entries, so the actual vLLM MoE layer must be
constructed with the wider K. Qwen3.6 is a composite
`Qwen3_5MoeForConditionalGeneration` checkpoint whose real setting is
`text_config.num_experts_per_tok`. The historical K=10/12 commands wrote a new,
unused outer `num_experts_per_tok` attribute instead. The text router stayed at
K=8 and the K=10/12 routing patch did not match any layer.

The corrected form is:

```json
{"text_config": {"num_experts_per_tok": 10}}
```

The corrected launchers also require a startup log assertion proving that the
routing patch matched a real layer at the requested router K.

## Model-by-model audit

| Model family | Native K | Retained above-native evaluations | Config location used by vLLM | Verdict |
|---|---:|---|---|---|
| Qwen3-30B-A3B Base/chat, Dolci checkpoints, and Qwen3 RL exports | 8 | K=9--16 and selected K=32 runs; several Qwen3 RL evaluations/training arms also use K=10/12 | Top-level `num_experts_per_tok` | **Valid.** Historical shallow overrides changed the real router setting. A representative Qwen K=12 result records the top-level K=12 override and the installed patch reports `router_k=12`; vLLM's Qwen3 MoE layer reads `config.num_experts_per_tok`. |
| Qwen3.5-35B-A3B off the shelf | 8 | None; the completed sweep is K=4--8 | Nested `text_config.num_experts_per_tok` | **No above-native exposure.** Its launcher already used the nested form for the completed K=4--8 sweep. |
| Qwen3.5 OpenThoughts SFT K=12 checkpoint | 12 in the converted checkpoint | TBLite at K=12 | Checkpoint itself contains nested `text_config.num_experts_per_tok=12` | **Valid.** It is served directly without a shallow K override, and the exact router-match startup assertion passed. The K=4 and K=8 SFT checkpoints similarly contain nested K=4 and K=8. |
| Qwen3.6-35B-A3B | 8 | Historical nominal K=10/12 capability, Terminal-Bench 2.0, TBLite, and TB2.1 runs | Nested `text_config.num_experts_per_tok` | **Historical K=10/12 invalid.** They actually ran native K=8. The newest assertion-gated attempts failed before evaluation, while older jobs silently produced native-K=8 outputs. Corrected TBLite/TB2.1 replacements use nested overrides. |
| GPT-OSS-120B | 4 | Completed K=5--8 normalized and K=4-reference-scaled sweeps | Top-level `num_experts_per_tok` | **Valid.** Both the HF config and vLLM GPT-OSS layer use the top-level field. The accidental K=9--16 launch wave was deleted and never retained in scientific tables or plots. |
| GLM-4.5-Air | 8 | None; completed sweep ends at K=8 | Top-level `num_experts_per_tok` | **No above-native exposure.** |
| Nemotron-3-Super | 22 | None; completed normalized sweep is K=11--22 | Top-level `num_experts_per_tok` | **No above-native exposure.** |

## Consequences for existing results

- All Qwen3.6 K=4/6/8 conclusions remain valid.
- Historical Qwen3.6 K=10/12 scores are best understood as extra stochastic
  native-K=8 replicates. They cannot support a claim about adding experts and
  must stay out of plots and comparisons.
- Qwen3.6 capability K=10/12 has not yet been corrected and rerun.
- Corrected Qwen3.6 TBLite and TB2.1 K=10/12 replacements are tracked in
  `qwen36_tblite_reference_sweep_jobs.md` and
  `qwen36_tb21_reference_sweep_jobs.md`.
- No retained Qwen3, Qwen3.5, GPT-OSS, GLM, or Nemotron conclusion needs to be
  invalidated for this specific issue.

This is therefore a Qwen3.5/Qwen3.6 composite-config nesting mistake in the
historical Qwen3.6 launcher, not a general vLLM failure to support K above a
checkpoint's native active-expert count.

# HELMET KILT Passage Trim: Does It Cut Away the Gold Passage?

Status: **open question, needs investigation**. The code described here is
merged and working; this document records a correctness concern about what it
does to scores, not a known bug.

## Background

Commit `8037efd3` ("Trim HELMET RAG passages to the tier budget like HELMET
does") added end-trimming of the retrieved passage block in
`src/olmo_eval/data/helmet_kilt_loader.py`.

It was needed because HELMET's pre-retrieved KILT files overshoot their nominal
length tier (the 128k files are reportedly 142-155k Llama-2 tokens) and HELMET
relies on trimming at inference time. Without the trim, a sizeable fraction of
128k prompts exceeded a 131072 `max_model_len`. Worse, because the in-process
vLLM provider submits a whole mixed-task batch in one call, each overflow
hard-failed every other KILT instance sharing its batch, including 8k-64k ones.

The relevant pieces:

- `_truncate_prompt_context` in `src/olmo_eval/data/helmet_kilt_loader.py`
- `task["max_prompt_tokens"] = size - max_gen_toks` in
  `src/olmo_eval/data/helmet_tasks.py`
- `max_prompt_tokens=self.helmet_config["max_prompt_tokens"]` in
  `HelmetKiltTask.load_data`, `src/olmo_eval/evals/tasks/helmet.py`

The port is faithful to upstream HELMET. Compare `tokenize` in HELMET's
`model_utils.py` (https://github.com/princeton-nlp/HELMET), which computes
`truncate_length` against `max_length - generation_max_length` and slices
`sample["context"]` at `offset_mapping[-truncate_length][0]`. Our version
matches it down to the offset indexing.

## The concern

These KILT files plant the gold passage at fixed relative depths through the
passage stack: 0.0, 0.2, 0.4, 0.6, 0.8, 0.95 for nq/triviaqa/popqa (the `dep6`
in the filenames) and three depths for hotpotqa. Averaging over depth is the
point of the task, since it is what measures the "lost in the middle" effect.

The trim removes text from the **end** of the passage block, so it eats the
deepest-planted gold passages first. An instance whose gold passage is cut away
is unanswerable no matter how good the model is.

### Rough magnitude (UNVERIFIED)

At the 128k tier the budget is `131072 - 20 = 131052` tokens
(`_RAG_MAX_GEN_TOKS = 20`). Taking the 142-155k figure from the commit message
at face value, overflow is ~11k-24k tokens, so roughly 8-16% of the passage
block is cut. A gold passage at depth 0.95 sits in the final 5% of the stack and
would therefore fall inside the cut region in essentially every case. Depth 0.8
sits outside even a 16% cut (which begins at 84%), but not by much.

If that holds, the depth-0.95 slice — one of six instances for nq, triviaqa and
popqa — is unanswerable at 128k, capping those tasks near 83% and depressing
measured accuracy by ~17% relative.

**None of this was measured.** `allenai/helmet-plus` returned 403/401 to the
investigating agent because the org's fine-grained token policy blocked the
available HF token. The numbers above are arithmetic on the commit message's
stated token range and should be confirmed against the real files before anyone
acts on them.

## The tokenizer divergence

This is the part most worth a careful look, because it is where we differ from
HELMET rather than merely reproducing it.

HELMET measures the budget with the **model's own** tokenizer. We measure with a
fixed reference tokenizer (`NousResearch/Llama-2-7b-hf`, via `REFERENCE_TOKENIZER`
in `helmet_infbench_loader.py`), so that the prompt is identical across models.

Llama-2 has a 32k vocabulary. The OLMo/Llama-3-class models on the scaling ladder
have ~128k vocabularies and encode the same English text in meaningfully fewer
tokens. For those models HELMET might not truncate at all — the text could fit
under 131072 in their own tokenization — and they would see every passage,
including the one at depth 0.95. Our port trims regardless.

So for exactly the models we care about, the current behavior is *stricter* than
HELMET and may discard gold passages the model had room for.

The trade-off is real in both directions:

- Fixed reference tokenizer: prompts are identical across models, so ladder
  comparisons are apples-to-apples. Costs fidelity to HELMET's published numbers
  and may throw away context the model could have used.
- Model's own tokenizer: matches HELMET per-model and preserves more passages
  for efficient tokenizers. Costs cross-model comparability, since two models
  would be scored on different prompts.

Note also that the budget has no safety margin (`size - max_gen_toks`, with no
equivalent of `_LONGQA_PROMPT_RESERVE`). A model whose tokenizer is *less*
efficient than Llama-2 can still overflow `max_model_len`, which is the original
failure this commit set out to prevent.

## Questions to answer

1. What is the actual rendered-prompt token length distribution per task per
   tier, measured with the Llama-2 reference tokenizer? Does overflow occur only
   at 128k, or at the smaller tiers too?
2. How many instances get trimmed at all, and by how much?
3. How many trimmed instances lose their gold passage entirely? The `ctxs`
   entries in HELMET's KILT data typically carry a `has_answer` field, which
   should make this directly checkable.
4. Is the loss concentrated in the depth-0.95 slice as predicted, or does it
   reach depth 0.8?
5. Measured with an OLMo tokenizer instead, how many of these prompts would have
   fit without any trim? That quantifies what the fixed-reference-tokenizer
   choice costs us.
6. Do HELMET's own published RAG numbers show a comparable ceiling at 128k? If
   the benchmark authors' scores also saturate below 100%, that is evidence the
   ceiling is intended and we should match it.

## Possible outcomes

- **Keep as is.** The ceiling is HELMET's, comparability across the ladder is
  preserved, and our numbers stay comparable to published ones.
- **Instrument it.** Log per task/tier how many instances were trimmed and how
  many lost their gold passage, so the ceiling is visible in results rather than
  silent. Cheap and useful regardless of which other option is chosen.
- **Switch to the model's own tokenizer.** Maximum HELMET fidelity per model,
  at the cost of cross-model prompt identity.
- **Drop instances whose gold passage was trimmed away**, rather than scoring
  them as failures. Changes the denominator and diverges from HELMET, so it
  would need justification.

## Practical notes for whoever picks this up

- Access to `allenai/helmet-plus` requires a fine-grained HF token; the default
  org token is rejected with 403 on `repo_info` and 401 on `hf_hub_download`.
- The 128k tier files are 1-3 GB each, cached by `huggingface_hub` after the
  first download. Start with a small tier (4k/8k) to establish whether overshoot
  is tier-specific before paying for the large ones.
- `max_samples` caps *questions*, not instances: every depth variant of a
  sampled question is kept, so a limit of 100 yields ~600 instances for nq.
- Setting `max_prompt_tokens=None` disables the trim, which is a convenient way
  to measure untrimmed prompt lengths.

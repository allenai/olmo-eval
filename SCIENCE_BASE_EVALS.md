# Science / science-literature evals for base-model training monitoring

Everything here is already registered in olmo-eval (branch `roryd/deepscholar-bench-s2patch`).
Selection criteria: no tools, no agentic loop, no LLM judge, short prompts, and a
logprob or short-generation scoring path so it can run repeatedly during training.

Variant conventions used below:
- `:rc` — cloze/completion scoring, per-char normalized logprob. Standard base-model format.
- `:bpb` — bits per byte. Smoothest signal early in training; comparable across tokenizers.
- `:mc` — lettered multiple choice with logprob over A/B/C/D. Needs a reasonably capable model
  before it rises off chance.
- `:olmo3base` — the Olmo 3 base preset for that task (fixed few-shot count and seed).

## Recommended core set

Ready-made suites, cheapest path to a full science signal:

| Suite | Tasks | What it covers |
| --- | --- | --- |
| `olmobase:mcqa_stem` | 23 | ARC + MMLU-STEM + MedMCQA + MedQA + SciQ, all `mc:olmo3base` |
| `mmlu:stem:rc:olmo3base` / `mmlu:stem:mc:olmo3base` | 18 | Biology, chemistry, physics, astronomy, CS, math, EE |
| `science:core` | 21 | ARC easy/challenge + SciQ + `mmlu:stem` (default variants) |
| `science:medicine` | 10 | MedMCQA, MedQA-en, 8 medicine-heavy MMLU subjects |

## Individual tasks

Science knowledge and exam QA:

| Task | Variants for base models | n | Median prompt (chars) |
| --- | --- | --- | --- |
| `sciq` | `rc:olmo3base`, `bpb:olmo3base`, `mc:olmo3base` | 1000 | 590 |
| `arc_easy`, `arc_challenge` | `rc:olmo3base`, `bpb`, `mc:olmo3base` | 1172 (challenge) | 880 |
| `mmlu_<stem subject>` | `rc:olmo3base`, `rc:bpb`, `mc:olmo3base` | 100–1500 each | ~1200 |
| `gpqa_main`, `gpqa_extended`, `gpqa_diamond` | `mc`, `bpb` | 448 / 546 / 198 | ~560 |
| `gpqa_{main,extended,diamond}_{biology,chemistry,physics}` | `mc`, `bpb` | subject slices | ~560 |
| `medmcqa` | `rc:olmo3base`, `bpb`, `mc:olmo3base` | 4183 | 510 |
| `medqa_en` | `rc:olmo3base`, `bpb`, `mc:olmo3base` | 1273 | 4300 |

GPQA's default variant is a chat generation with answer extraction — use `:mc` or `:bpb` for base
models. Expect GPQA to sit near chance for most of pretraining; it is more useful as a late-run
check than an early monitoring signal.

Science literature and evidence use (the closest thing to "science literature" that is safe for
base models — all are logprob-scored yes/no or MC over a supplied passage, no retrieval):

| Task | Variants | n | Median prompt (chars) | Note |
| --- | --- | --- | --- | --- |
| `qasper_yesno` | `rc:olmo3base`, `bpb:olmo3base` | 319 | 4600 | Yes/No over NLP-paper evidence spans |
| `sciriff_yesno` | `rc:olmo3base`, `bpb:olmo3base` | 1582 | 8200 (max 19k) | Yes/No over scientific abstracts; ~2–5k tokens |
| `lab_bench_dbqa` | `olmo3base`, `bpb:olmo3base`, `mc` | 520 | 980 | Biology database QA |
| `lab_bench_protocolqa` | `olmo3base`, `bpb:olmo3base`, `mc` | 108 | 900 | Wet-lab protocol troubleshooting |
| `lab_bench_litqa2` | `mc`, `bpb`, `olmo3base` | 199 | 240 | Recall from recent biology literature |
| `lab_bench_suppqa` | `mc`, `bpb`, `olmo3base` | 82 | 190 | QA over paper supplementary material |
| `lab_bench_seqqa` | `mc`, `bpb`, `olmo3base` | 600 | 2000 (max 8k) | DNA/protein sequence reasoning |

`qasper_yesno`, `sciriff_yesno`, `lab_bench_dbqa`, and `lab_bench_protocolqa` are already in
`olmobase:easy:qa:rc` and `olmobase:easy:qa:bpb`, so they need no new plumbing.

## Suggested monitoring configuration

Two tiers, both logprob-only:

1. Every checkpoint (cheap, ~10k instances): `olmobase:mcqa_stem` plus
   `sciq:bpb:olmo3base`, `qasper_yesno:bpb:olmo3base`, `sciriff_yesno:bpb:olmo3base`,
   `lab_bench_dbqa:bpb:olmo3base`, `lab_bench_protocolqa:bpb:olmo3base`.
   BPB moves early and monotonically, which is what you want for a training curve.
2. Periodic (every N checkpoints): the same tasks in `rc`/`mc` form plus
   `gpqa_main:mc`, `gpqa_extended:mc`, and the GPQA subject slices, for accuracy numbers
   comparable to published results.

Launch note for whoever runs these: these are all `prompt_logprobs` tasks, and MMLU-style logprob
runs have OOM'd on us when launched with the generation-task vLLM config. Set
`max_model_len` near the actual prompt length (4k covers everything above except
`lab_bench_cloning_scenarios`), use the full GPU count via `tensor_parallel_size`, and lower
`gpu_memory_utilization`. See `MMLU_LOGPROB_OOM_ASSESSMENT.md`.

## Deliberately excluded

| Task / suite | Why |
| --- | --- |
| `astabench_scholarqa`, `expertqa`, `expertqa:cite`, `researchqa` | Long-form generation scored by an external LLM judge; needs an OpenAI key and costs money per checkpoint |
| `litsearch`, `sage_short_form`, `sage_open_ended` | Agentic — live Semantic Scholar retrieval through a tool-providing harness |
| `litsearch_rerank` | Judge-free and tool-free, but a chat-format reranking instruction over a large candidate pool; base models will not follow it |
| `geneturing_*` (14 tasks) | Free-form chat generation with fuzzy string scorers; base models score near zero for formatting reasons, not knowledge |
| `lab_bench_cloning_scenarios` | Median prompt 9.4k chars, max 51k — the long-context case you wanted to avoid |
| `ruler_*` | Long-context by construction (4k–128k) |
| `science:all`, `science:nojudge`, `science:research` | Convenient umbrellas, but they pull in the agentic and judge-based tasks above and use default (chat) variants |

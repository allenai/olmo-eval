# Evaluating Olmo 3.5 multimodal SFT checkpoints

Two suites per checkpoint:

- **Image QA**, through the `olmo_core_vlm` provider on the OLMo-core export, with prompts in the
  SFT chat layout (`prompt_format=olmo35`).
- **Text**, the ledger's GPU text cells, on a text-only HF export of the checkpoint's language model.

## One command

```bash
export OLMO_CORE_DIR=/path/to/OLMo-core   # the training code (allenai/OLMo-core#942's branch)
export SCALING_LADDERS_DIR=/path/to/scaling-ladders LEDGER_DIR=/path/to/olmo-post-training-ledger
export OLMO_EVAL_PY=/path/to/olmo-eval/.venv/bin/python PY=/path/to/python-with-olmo-core
scripts/olmo35_mmsft/eval_sft_checkpoint.sh <name> <checkpoint step dir> vision   # export, image QA, HF conversion job
scripts/olmo35_mmsft/eval_sft_checkpoint.sh <name> <checkpoint step dir> text     # after the conversion finished
```

Outputs go to `$OUT_ROOT/<name>/` (`olmo-core-export`, `hf-multimodal`, `hf-text`). Jobs go to
`ai2/olmo-instruct` on `ai2/holmes`, priority urgent, never preemptible.

## What the chat layout does

`src/olmo_eval/inference/providers/olmo_core_vlm/olmo35_chat.py` builds each prompt so its tokens
equal the training tokens of OLMo-core's olmo35 layout up to the answer:

- the Olmo 3.5 chat template of `allenai/dolma2-tokenizer-olmo35@8b97170`, with its default system turn;
- `<think>` and `</think>` as single tokens (ids 100256 and 100266), promoted into reserved slots
  exactly as open-instruct's `--reserved_slot_tokens` does;
- the `tokenizer.json` pre-tokenizer (not GPT-2's regex);
- image blocks at the start of the first user turn, `Image {i}` before each when there are several;
- thinking off by default: the prompt ends `<|im_start|>assistant\n<think></think>`, as a
  thinking-off training turn does. `-o provider.kwargs.thinking=on` leaves the block open and
  strips the reasoning from the output.

`tests/providers/test_olmo35_chat.py` checks the prompt equals the training prefix for text-only,
one-image, two-image and multi-turn conversations (run it with OLMo-core #942 on `PYTHONPATH`).

Mid-training checkpoints keep their plain-document prompts (`prompt_format=document`, the default for
them).

## Pieces

| Script | Does |
|---|---|
| `eval_vision.sh <name> <export> [olmo35\|document]` | image-QA suite (default: ChartQA, VQAv2, DocVQA, TextVQA, AI2D, CountBenchQA, RealWorldQA); `LIMIT`, `GPUS`, `TASKS` |
| `mm_hf_to_text_hf.py <mm hf> <out>` | text-only HF model: the `language_model.*` tensors byte for byte, `text_config`, the promoted Olmo 3.5 chat tokenizer |
| `text_cells.py <text hf> <name> --ledger <dir>` | the ledger's recorded GPU text cells with only model, names, secrets and placement changed |

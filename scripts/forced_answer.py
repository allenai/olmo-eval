#!/usr/bin/env python3
"""Forced-answer scoring of finished chat multiple-choice evaluations.

For every item of a finished olmo-eval run of a chain-of-thought multiple-choice task, this
rebuilds the chat prompt, appends the model's own generation and then the text "Therefore, the
answer is (", and reads the model's next-token log-probabilities over the option letters. The
predicted option is the letter with the highest log-probability. A generation whose <think>
block never closed is closed first, and one too long to fit the context window is cut from the
end, so every item gets a score, including those where the model looped and never answered.

Inputs are an olmo-eval results directory (predictions/ and requests/ as written by
``olmo-eval run``). Writes, into --out:
  <task>-forced.jsonl   one row per item: gold, letter log-probabilities, forced prediction,
                        forced correctness, generated answer, closed/truncated flags
  forced-metrics.json   per task: generated accuracy, forced accuracy, letter coverage

Usage:
    python scripts/forced_answer.py --model <path> [--tokenizer <path>] --inputs <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNO"
SUFFIX = "Therefore, the answer is ("
CLOSE = "\n</think>\n\n"


def _gold(doc: dict, label: object) -> int | None:
    if isinstance(doc.get("gold_idx"), int):
        return doc["gold_idx"]
    text = str(label).strip()
    if len(text) == 1 and text in LETTERS:
        return LETTERS.index(text)
    if text.isdigit():
        return int(text)
    return None


def _output(pred: dict) -> tuple[str, str]:
    mo = pred["model_output"]
    mo = ast.literal_eval(mo) if isinstance(mo, str) else mo
    o = mo[0] if isinstance(mo, list) else mo
    return (o.get("original_text") or o.get("text") or ""), (o.get("extracted_answer") or "")


def load_items(inputs: Path, limit: int | None) -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = {}
    for pred_file in sorted(inputs.glob("predictions/**/*-predictions.jsonl")):
        stem = pred_file.name[: -len("-predictions.jsonl")]
        req_files = list(inputs.glob(f"requests/**/{stem}-requests.jsonl"))
        assert len(req_files) == 1, (stem, req_files)
        with open(req_files[0]) as f:
            reqs = {r["doc_id"]: r for r in map(json.loads, f)}
        items = []
        with open(pred_file) as f:
            preds = [json.loads(line) for line in f]
        for p in preds:
            r = reqs[p["doc_id"]]
            doc = r.get("doc") or {}
            choices = doc.get("choices") or []
            gold = _gold(doc, p.get("label", r.get("label")))
            if not choices or gold is None or not 0 <= gold < len(choices):
                continue
            text, answer = _output(p)
            items.append(
                {
                    "doc_id": p["doc_id"],
                    "native_id": p.get("native_id"),
                    "messages": r["request"]["context"],
                    "n": len(choices),
                    "gold": gold,
                    "generation": text,
                    "generated_answer": answer,
                }
            )
            if limit and len(items) >= limit:
                break
        tasks[stem] = items
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--limit-per-task", type=int)
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no model")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    tasks = load_items(a.inputs, a.limit_per_task)
    suffix_ids = tok.encode(SUFFIX, add_special_tokens=False)
    close_ids = tok.encode(CLOSE, add_special_tokens=False)
    prompts, index = [], []
    for task, items in tasks.items():
        for it in items:
            head = tok.apply_chat_template(
                it["messages"], add_generation_prompt=True, tokenize=False
            )
            head_ids = tok.encode(head, add_special_tokens=False)
            gen_ids = tok.encode(it["generation"], add_special_tokens=False)
            unclosed = "<think>" in it["generation"] and "</think>" not in it["generation"]
            budget = a.max_model_len - len(head_ids) - len(close_ids) - len(suffix_ids) - 2
            truncated = len(gen_ids) > budget
            if truncated:
                gen_ids = gen_ids[: max(budget, 0)]
            # The generation already ends where the model stopped; a cut generation inside a think
            # block, or one that never closed it, gets the block closed before the answer prompt.
            if unclosed:
                gen_ids = gen_ids + close_ids
            sep = tok.encode("\n\n", add_special_tokens=False)
            ids = head_ids + gen_ids + sep + suffix_ids
            it.update(unclosed=unclosed, truncated=truncated, prompt_tokens=len(ids))
            prompts.append(ids)
            index.append((task, it))
    longest = max(map(len, prompts), default=0)
    print(f"{len(prompts)} prompts over {len(tasks)} tasks; longest {longest} tokens", flush=True)
    if a.dry_run:
        task, it = index[0]
        print(tok.decode(prompts[0][-400:]))
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=a.model,
        tokenizer=a.tokenizer or a.model,
        tensor_parallel_size=a.tp,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    letter_ids: dict[int, str] = {}
    for letter in LETTERS:
        for variant in (letter, " " + letter):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                letter_ids.setdefault(ids[0], letter)
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    summary: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {}
    for (task, it), out in zip(index, outs, strict=True):
        top = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
        scores: dict[str, float] = {}
        for tid, lp in top.items():
            letter = letter_ids.get(tid)
            if letter is None:
                decoded = (lp.decoded_token or "").strip()
                letter = decoded if len(decoded) == 1 and decoded in LETTERS else None
            if letter and LETTERS.index(letter) < it["n"]:
                scores[letter] = max(scores.get(letter, -math.inf), lp.logprob)
        pred = max(scores, key=scores.get) if scores else None
        mass = sum(math.exp(v) for v in scores.values())
        row = {
            k: it[k]
            for k in (
                "doc_id",
                "native_id",
                "n",
                "gold",
                "generated_answer",
                "unclosed",
                "truncated",
                "prompt_tokens",
            )
        }
        row.update(
            gold_letter=LETTERS[it["gold"]],
            letter_logprobs=scores,
            forced=pred,
            forced_correct=int(pred == LETTERS[it["gold"]]),
            generated_correct=int(
                it["generated_answer"].strip("() ").upper() == LETTERS[it["gold"]]
            ),
            letter_mass=mass,
        )
        rows.setdefault(task, []).append(row)
    for task, rs in rows.items():
        n = len(rs)
        summary[task] = {
            "n": n,
            "generated_acc": sum(r["generated_correct"] for r in rs) / n,
            "forced_acc": sum(r["forced_correct"] for r in rs) / n,
            "no_generated_answer": sum(not r["generated_answer"] for r in rs) / n,
            "unclosed": sum(r["unclosed"] for r in rs) / n,
            "truncated": sum(r["truncated"] for r in rs) / n,
            "letter_coverage": sum(bool(r["letter_logprobs"]) for r in rs) / n,
            "mean_letter_mass": sum(r["letter_mass"] for r in rs) / n,
        }
        with open(a.out / f"{task}-forced.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    (a.out / "forced-metrics.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

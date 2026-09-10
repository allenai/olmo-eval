#!/usr/bin/env python3
"""Compare parser-returned Qwen MATH-500 responses across active-expert counts.

This is intentionally a response-quality analysis rather than another evaluator. It summarizes
observable failure proxies in saved predictions and emits matched examples for manual review.
The saved prediction schema does not retain Qwen's hidden reasoning_content, so token counts cover
the full generation while textual coherence metrics cover the parser-returned assistant content.
That content is often a full user-facing worked solution, not only the terse extracted answer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "adaptive_experts"

RUNS = {
    1: (
        "01KX56NQMVH3TVZKGAM9DPGF05",
        "01KX6KNJBM5KRF349TF72QWCFY",
        "01KX6KQHSH0PYEYP9Q69FNAEZ4",
    ),
    2: (
        "01KX56NZQVWQBKXH6VP6CXV345",
        "01KX6KNTYNVNE16WXK0DNTGGP5",
        "01KX6KQTXKYB6ENVAV2Q30KDMS",
    ),
    3: (
        "01KX7GFW9RCAPT76WTTBRWT0SR",
        "01KX7GDJXSF8XTP79Y5WK0BCPE",
        "01KX7GH43M5RXNRJPPTJN62TT1",
    ),
    4: (
        "01KX56P8M9TD1MPMCT582AXZ87",
        "01KX6KP4V9Z43MR9RJPD4YV6JN",
        "01KX6KR3YH8YET0JZACDSQA8XP",
    ),
    5: (
        "01KX6KSNE491THB9DT4Y9PPJ7W",
        "01KX6KTWW70DP1684C2CR10TCH",
        "01KX6KW6Z0GV5KHQTAMP1KA35J",
    ),
    6: (
        "01KX6KSYGH7H8RQPEGJFYC3PEW",
        "01KX6KV5MDA3AAG9ET53W91FEE",
        "01KX6KWKTN48AHZPYN5A9GEPFS",
    ),
    7: (
        "01KX6KT77JRKJSW67J2NY5BFQQ",
        "01KX6KVEM81QYVVBJZK6JTTS0V",
        "01KX6KX1QMYC8KEN1FXV56S2TZ",
    ),
    8: (
        "01KX56PH4KDPVQFSBB3ZHDEMB7",
        "01KX6KPE77ANFQHN4D4MP1VSEZ",
        "01KX6KRCDK4ZSA4VZ4TABXG3CD",
    ),
    9: (
        "01KX7GG47N7XP7K8NYW9NE7B73",
        "01KX7GDX9FWDR43N8SDD232ZZK",
        "01KX7GHBWTRAJ5DVSAYASRDHTD",
    ),
    10: (
        "01KX7GGCK1WZX5XKEW3VHHDCB4",
        "01KX7GE5PFF8FJX62HHVS9Z0B9",
        "01KX7GHKQNRSH6JP8QX6A449QF",
    ),
    11: (
        "01KX7GGMF6J8VV1YJ8F1REKFQ3",
        "01KX7GEK926AYKMV7R7JE5AP1N",
        "01KX7GHVSSYXR16CSZ8WW8X997",
    ),
    12: (
        "01KX7GGVVD8NQS8W2RFNAWNWXS",
        "01KX7GF02PRGQR0SNF23BX230K",
        "01KX7GJ4S0513935Q6P96HD3CE",
    ),
    13: (
        "01KX7VBBDJDTWW1YK3010QB7R0",
        "01KX7VCEF5H37P48MBQ2QY8R3R",
    ),
    14: (
        "01KX7VBM2Z141G67RM7GB8C0R9",
        "01KX7VCQJ441PPKSMCJ5TD9P1Q",
    ),
    15: (
        "01KX7VBW0H13C928JGVQ9D3GV1",
        "01KX7VD8PC4VMCWSM9VYMEJGMB",
    ),
    16: (
        "01KX56PSK9WJMQ68BTQ3P54P5D",
        "01KX6KPPW8A7PPF9GKT8VT7PXQ",
        "01KX6KRMQ39F6SYMDM12KSJJBP",
    ),
    32: (
        "01KX5ACWAEQSRM0E54YFBB4Z8H",
        "01KX6KPZFHG49NF7MBQX0QGXFP",
        "01KX6KRXF9V7TER9WWTMPCNFHZ",
    ),
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
RESTART_RE = re.compile(
    r"\b(?:wait|actually|correction|reconsider|start over|let(?:'s| us) restart|"
    r"made (?:an?|the) (?:error|mistake)|that was (?:wrong|incorrect))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Response:
    k: int
    replicate: int
    experiment_id: str
    doc_id: int
    level: int | None
    subject: str | None
    query: str
    gold: str
    text: str
    extracted_answer: str | None
    correct: bool
    generated_tokens: int
    visible_chars: int
    visible_words: int
    visible_sentences: int
    cap_hit: bool
    has_boxed_answer: bool
    empty: bool
    unbalanced_dollars: bool
    unbalanced_braces: bool
    repeated_line_fraction: float
    duplicate_sentence_fraction: float
    repeated_4gram_fraction: float
    restart_markers: int
    max_identical_word_run: int


def one_file(experiment_id: str, kind: str) -> Path:
    candidates = list((RESULTS / experiment_id / kind).rglob("math500*jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one MATH-500 {kind} file for {experiment_id}, got {candidates}"
        )
    return candidates[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalized_units(units: list[str]) -> list[str]:
    return [re.sub(r"\s+", " ", unit.strip().lower()) for unit in units if unit.strip()]


def duplicate_fraction(units: list[str]) -> float:
    normalized = normalized_units(units)
    if not normalized:
        return 0.0
    return 1.0 - len(set(normalized)) / len(normalized)


def repeated_ngram_fraction(words: list[str], n: int = 4) -> float:
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def max_word_run(words: list[str]) -> int:
    if not words:
        return 0
    best = current = 1
    for previous, current_word in zip(words, words[1:], strict=False):
        if current_word == previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def load_run(
    k: int, replicate: int, experiment_id: str, requests: dict[int, dict]
) -> list[Response]:
    predictions = read_jsonl(one_file(experiment_id, "predictions"))
    if len(requests) != 500 or len(predictions) != 500:
        raise RuntimeError(
            f"Incomplete MATH-500 run {experiment_id}: {len(requests)}, {len(predictions)}"
        )

    output = []
    for row in predictions:
        request = requests[row["doc_id"]]
        sample = row["model_output"][0]
        text = row.get("final_output") or sample.get("text") or ""
        words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
        sentences = [unit for unit in SENTENCE_RE.split(text) if unit.strip()]
        lines = [line for line in text.splitlines() if line.strip()]
        doc = request["doc"]
        accuracy = row["instance_metrics"]["accuracy"]["minerva_math_flex"]
        output.append(
            Response(
                k=k,
                replicate=replicate,
                experiment_id=experiment_id,
                doc_id=row["doc_id"],
                level=doc.get("level"),
                subject=doc.get("type"),
                query=doc["query"],
                gold=row["label"],
                text=text,
                extracted_answer=sample.get("extracted_answer"),
                correct=bool(accuracy),
                generated_tokens=int(sample.get("num_tokens_all", sample.get("num_tokens", 0))),
                visible_chars=len(text),
                visible_words=len(words),
                visible_sentences=len(sentences),
                cap_hit=int(sample.get("num_tokens_all", sample.get("num_tokens", 0))) >= 32768,
                has_boxed_answer="\\boxed" in text,
                empty=not text.strip(),
                unbalanced_dollars=text.count("$") % 2 != 0,
                unbalanced_braces=text.count("{") != text.count("}"),
                repeated_line_fraction=duplicate_fraction(lines),
                duplicate_sentence_fraction=duplicate_fraction(sentences),
                repeated_4gram_fraction=repeated_ngram_fraction(words),
                restart_markers=len(RESTART_RE.findall(text)),
                max_identical_word_run=max_word_run(words),
            )
        )
    return output


def mean(values: list[float | int | bool]) -> float:
    return statistics.mean(float(value) for value in values)


def percentile(values: list[float | int], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(responses: list[Response]) -> dict[str, float]:
    if not responses:
        return {"n": 0}
    return {
        "n": len(responses),
        "accuracy_pct": 100 * mean([item.correct for item in responses]),
        "generated_tokens_mean": mean([item.generated_tokens for item in responses]),
        "generated_tokens_median": percentile([item.generated_tokens for item in responses], 0.5),
        "generated_tokens_p90": percentile([item.generated_tokens for item in responses], 0.9),
        "visible_words_mean": mean([item.visible_words for item in responses]),
        "visible_words_median": percentile([item.visible_words for item in responses], 0.5),
        "cap_hit_pct": 100 * mean([item.cap_hit for item in responses]),
        "boxed_pct": 100 * mean([item.has_boxed_answer for item in responses]),
        "empty_pct": 100 * mean([item.empty for item in responses]),
        "unbalanced_dollars_pct": 100 * mean([item.unbalanced_dollars for item in responses]),
        "unbalanced_braces_pct": 100 * mean([item.unbalanced_braces for item in responses]),
        "format_issue_pct": 100
        * mean([item.unbalanced_dollars or item.unbalanced_braces for item in responses]),
        "restart_marker_pct": 100 * mean([item.restart_markers > 0 for item in responses]),
        "repeated_line_fraction_mean": mean([item.repeated_line_fraction for item in responses]),
        "duplicate_sentence_fraction_mean": mean(
            [item.duplicate_sentence_fraction for item in responses]
        ),
        "repeated_4gram_fraction_mean": mean([item.repeated_4gram_fraction for item in responses]),
        "word_run_3plus_pct": 100 * mean([item.max_identical_word_run >= 3 for item in responses]),
    }


def suspicion(item: Response) -> float:
    return (
        8 * item.cap_hit
        + 4 * item.duplicate_sentence_fraction
        + 3 * item.repeated_line_fraction
        + 2 * item.repeated_4gram_fraction
        + 0.75 * min(item.restart_markers, 4)
        + 2 * item.unbalanced_braces
        + 1 * item.unbalanced_dollars
        + min(item.visible_words / 3000, 2)
    )


def candidate_rows(responses: list[Response]) -> list[dict]:
    lookup = {(item.k, item.replicate, item.doc_id): item for item in responses}
    groups: dict[str, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
    for replicate in range(1, 4):
        for doc_id in range(500):
            trio = {k: lookup[(k, replicate, doc_id)] for k in (3, 4, 8)}
            pattern = "".join("1" if trio[k].correct else "0" for k in (3, 4, 8))
            group = {
                "111": "all_correct",
                "011": "k3_wrong_k4_k8_correct",
                "001": "only_k8_correct",
                "101": "k4_wrong_k3_k8_correct",
                "110": "low_k_correct_k8_wrong",
            }.get(pattern, f"pattern_{pattern}")
            low_suspicion = max(suspicion(trio[3]), suspicion(trio[4]))
            groups[group].append((low_suspicion, (replicate, doc_id)))

    selected = []
    preferred = (
        "all_correct",
        "k3_wrong_k4_k8_correct",
        "only_k8_correct",
        "k4_wrong_k3_k8_correct",
        "low_k_correct_k8_wrong",
    )
    for group in preferred:
        for _, (replicate, doc_id) in sorted(groups.get(group, []), reverse=True)[:5]:
            trio = {k: lookup[(k, replicate, doc_id)] for k in (3, 4, 8)}
            selected.append(
                {
                    "group": group,
                    "replicate": replicate,
                    "doc_id": doc_id,
                    "query": trio[8].query,
                    "subject": trio[8].subject,
                    "level": trio[8].level,
                    "responses": {
                        str(k): {
                            "correct": trio[k].correct,
                            "generated_tokens": trio[k].generated_tokens,
                            "visible_words": trio[k].visible_words,
                            "cap_hit": trio[k].cap_hit,
                            "suspicion": suspicion(trio[k]),
                            "extracted_answer": trio[k].extracted_answer,
                            "text": trio[k].text,
                        }
                        for k in (3, 4, 8)
                    },
                }
            )
    return selected


def paired_subset_summary(
    lookup: dict[tuple[int, int, int], Response], patterns: set[str]
) -> dict:
    triples = []
    for replicate in range(1, 4):
        for doc_id in range(500):
            trio = {k: lookup[(k, replicate, doc_id)] for k in (3, 4, 8)}
            pattern = "".join("1" if trio[k].correct else "0" for k in (3, 4, 8))
            if pattern in patterns:
                triples.append(trio)

    output = {
        "n": len(triples),
        "by_k": {
            str(k): summarize([trio[k] for trio in triples])
            for k in (3, 4, 8)
        },
    }
    for low_k in (3, 4):
        ratios = [
            trio[low_k].generated_tokens / max(trio[8].generated_tokens, 1)
            for trio in triples
        ]
        visible_ratios = [
            trio[low_k].visible_words / max(trio[8].visible_words, 1)
            for trio in triples
        ]
        output[f"k{low_k}_vs_k8"] = {
            "generated_token_ratio_mean": mean(ratios),
            "generated_token_ratio_median": percentile(ratios, 0.5),
            "generated_tokens_1p5x_or_more_pct": 100 * mean([ratio >= 1.5 for ratio in ratios]),
            "generated_tokens_2x_or_more_pct": 100 * mean([ratio >= 2 for ratio in ratios]),
            "visible_word_ratio_mean": mean(visible_ratios),
            "visible_word_ratio_median": percentile(visible_ratios, 0.5),
        }
    return output


def compare_all_k_to_k8(responses: list[Response]) -> dict[str, dict]:
    lookup = {(item.k, item.replicate, item.doc_id): item for item in responses}
    output = {}
    for k in RUNS:
        pairs = []
        for replicate in range(1, min(len(RUNS[k]), len(RUNS[8])) + 1):
            for doc_id in range(500):
                pairs.append((lookup[(k, replicate, doc_id)], lookup[(8, replicate, doc_id)]))

        both_correct = [
            (item, reference)
            for item, reference in pairs
            if item.correct and reference.correct
        ]
        ratios = [
            item.generated_tokens / max(reference.generated_tokens, 1)
            for item, reference in both_correct
        ]
        visible_ratios = [
            item.visible_words / max(reference.visible_words, 1)
            for item, reference in both_correct
        ]
        output[str(k)] = {
            "n": len(pairs),
            "replicates": min(len(RUNS[k]), len(RUNS[8])),
            "correctness_agreement_pct": 100
            * mean([item.correct == reference.correct for item, reference in pairs]),
            "both_correct_n": len(both_correct),
            "only_k_correct_n": sum(
                item.correct and not reference.correct for item, reference in pairs
            ),
            "only_k8_correct_n": sum(
                not item.correct and reference.correct for item, reference in pairs
            ),
            "both_wrong_n": sum(
                not item.correct and not reference.correct for item, reference in pairs
            ),
            "both_correct_k_summary": summarize([item for item, _ in both_correct]),
            "both_correct_k8_summary": summarize([reference for _, reference in both_correct]),
            "both_correct_generated_token_ratio_mean": mean(ratios) if ratios else None,
            "both_correct_generated_token_ratio_median": percentile(ratios, 0.5)
            if ratios
            else None,
            "both_correct_generated_tokens_1p5x_or_more_pct": 100
            * mean([ratio >= 1.5 for ratio in ratios])
            if ratios
            else None,
            "both_correct_generated_tokens_2x_or_more_pct": 100
            * mean([ratio >= 2 for ratio in ratios])
            if ratios
            else None,
            "both_correct_visible_word_ratio_median": percentile(visible_ratios, 0.5)
            if visible_ratios
            else None,
        }
    return output


def build_prompt_groups(
    responses: list[Response],
) -> dict[tuple[int, int], list[Response]]:
    groups: dict[tuple[int, int], list[Response]] = defaultdict(list)
    for item in responses:
        groups[(item.k, item.doc_id)].append(item)
    for k, experiment_ids in RUNS.items():
        for doc_id in range(500):
            items = groups[(k, doc_id)]
            if len(items) != len(experiment_ids):
                raise RuntimeError(
                    f"Expected {len(experiment_ids)} responses for K={k}, doc={doc_id}; "
                    f"got {len(items)}"
                )
            items.sort(key=lambda item: item.replicate)
    return groups


def strict_majority_correct(items: list[Response]) -> bool:
    return sum(item.correct for item in items) > len(items) / 2


def all_runs_correct(items: list[Response]) -> bool:
    return all(item.correct for item in items)


def median_field(items: list[Response], field: str) -> float:
    return percentile([getattr(item, field) for item in items], 0.5)


def prompt_matched_cohort_summary(
    groups: dict[tuple[int, int], list[Response]], doc_ids: set[int]
) -> dict:
    """Summarize every K on one fixed prompt cohort.

    Repetitions are aggregated within each prompt before ratios are formed. This gives every prompt
    equal weight even when a K has two rather than three repetitions.
    """
    ordered_doc_ids = sorted(doc_ids)
    reference_tokens = {
        doc_id: median_field(groups[(8, doc_id)], "generated_tokens")
        for doc_id in ordered_doc_ids
    }
    reference_words = {
        doc_id: median_field(groups[(8, doc_id)], "visible_words")
        for doc_id in ordered_doc_ids
    }
    by_k = {}
    for k in RUNS:
        prompt_tokens = [
            median_field(groups[(k, doc_id)], "generated_tokens")
            for doc_id in ordered_doc_ids
        ]
        prompt_words = [
            median_field(groups[(k, doc_id)], "visible_words")
            for doc_id in ordered_doc_ids
        ]
        token_ratios = [
            value / max(reference_tokens[doc_id], 1)
            for doc_id, value in zip(ordered_doc_ids, prompt_tokens, strict=True)
        ]
        visible_word_ratios = [
            value / max(reference_words[doc_id], 1)
            for doc_id, value in zip(ordered_doc_ids, prompt_words, strict=True)
        ]
        items = [item for doc_id in ordered_doc_ids for item in groups[(k, doc_id)]]
        response_summary = summarize(items)
        by_k[str(k)] = {
            "replicates": len(RUNS[k]),
            "response_summary": response_summary,
            "majority_correct_prompt_pct": 100
            * mean([strict_majority_correct(groups[(k, doc_id)]) for doc_id in ordered_doc_ids]),
            "all_runs_correct_prompt_pct": 100
            * mean([all_runs_correct(groups[(k, doc_id)]) for doc_id in ordered_doc_ids]),
            "prompt_median_generated_tokens_mean": mean(prompt_tokens),
            "prompt_median_generated_tokens_median": percentile(prompt_tokens, 0.5),
            "generated_token_ratio_vs_k8_mean": mean(token_ratios),
            "generated_token_ratio_vs_k8_median": percentile(token_ratios, 0.5),
            "visible_word_ratio_vs_k8_median": percentile(visible_word_ratios, 0.5),
        }
    return {
        "n_prompts": len(ordered_doc_ids),
        "doc_ids": ordered_doc_ids,
        "by_k": by_k,
    }


def fixed_prompt_cohorts(responses: list[Response]) -> dict[str, dict]:
    groups = build_prompt_groups(responses)
    all_doc_ids = set(range(500))
    k8_majority_correct = {
        doc_id
        for doc_id in all_doc_ids
        if strict_majority_correct(groups[(8, doc_id)])
    }
    common_all_correct_k4_k8 = {
        doc_id
        for doc_id in all_doc_ids
        if all(all_runs_correct(groups[(k, doc_id)]) for k in range(4, 9))
    }
    common_all_correct_k4_k16 = {
        doc_id
        for doc_id in all_doc_ids
        if all(all_runs_correct(groups[(k, doc_id)]) for k in range(4, 17))
    }
    return {
        "all_prompts": {
            "definition": "All 500 MATH-500 prompt IDs; no outcome conditioning.",
            **prompt_matched_cohort_summary(groups, all_doc_ids),
        },
        "k8_majority_correct": {
            "definition": (
                "Fixed prompts correct in at least two of the three K=8 repetitions; "
                "the cohort is reused unchanged at every K."
            ),
            **prompt_matched_cohort_summary(groups, k8_majority_correct),
        },
        "common_all_correct_k4_k8": {
            "definition": (
                "Prompts correct in every repetition at every K from 4 through 8; "
                "secondary successful-response diagnostic."
            ),
            **prompt_matched_cohort_summary(groups, common_all_correct_k4_k8),
        },
        "common_all_correct_k4_k16": {
            "definition": (
                "Prompts correct in every available repetition at every K from 4 through 16; "
                "secondary successful-response diagnostic."
            ),
            **prompt_matched_cohort_summary(groups, common_all_correct_k4_k16),
        },
    }


def grouped_summary(
    responses: list[Response], field: str
) -> dict[str, dict[str, dict[str, float]]]:
    values = sorted({getattr(item, field) for item in responses}, key=str)
    return {
        str(value): {
            str(k): summarize(
                [item for item in responses if item.k == k and getattr(item, field) == value]
            )
            for k in (3, 4, 8)
        }
        for value in values
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical_requests = {
        row["doc_id"]: row
        for row in read_jsonl(one_file(RUNS[8][0], "requests"))
    }
    responses = []
    for k, experiment_ids in RUNS.items():
        for replicate, experiment_id in enumerate(experiment_ids, start=1):
            responses.extend(load_run(k, replicate, experiment_id, canonical_requests))

    by_k = {k: [item for item in responses if item.k == k] for k in RUNS}
    pattern_counts = Counter()
    lookup = {(item.k, item.replicate, item.doc_id): item for item in responses}
    for replicate in range(1, 4):
        for doc_id in range(500):
            pattern = "".join(
                "1" if lookup[(k, replicate, doc_id)].correct else "0" for k in (3, 4, 8)
            )
            pattern_counts[pattern] += 1

    payload = {
        "schema": {
            "correctness_pattern_order": [3, 4, 8],
            "visible_text_limitation": (
                "Saved predictions contain parser-returned assistant content, often a full "
                "worked solution, but not hidden reasoning_content; generated_tokens covers "
                "the full response."
            ),
        },
        "runs": RUNS,
        "summary": {str(k): summarize(items) for k, items in by_k.items()},
        "summary_correct": {
            str(k): summarize([item for item in items if item.correct]) for k, items in by_k.items()
        },
        "summary_incorrect": {
            str(k): summarize([item for item in items if not item.correct])
            for k, items in by_k.items()
        },
        "paired_correctness_patterns": dict(sorted(pattern_counts.items())),
        "all_k_vs_k8": compare_all_k_to_k8(responses),
        "fixed_prompt_cohorts": fixed_prompt_cohorts(responses),
        "by_subject": grouped_summary(responses, "subject"),
        "by_level": grouped_summary(responses, "level"),
        "paired_subsets": {
            "all_three_correct": paired_subset_summary(lookup, {"111"}),
            "k3_and_k8_correct": paired_subset_summary(lookup, {"101", "111"}),
            "k4_and_k8_correct": paired_subset_summary(lookup, {"011", "111"}),
        },
        "manual_review_candidates": candidate_rows(responses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

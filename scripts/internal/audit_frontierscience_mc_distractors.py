"""Run a higher-stringency audit over flagged FrontierScience MC sets.

This consumes the proposal, review, and triage artifacts written by
``build_frontierscience_mc_distractors.py``. Strict-clean items are carried
forward unchanged. Flagged items receive an independent audit that may retain
the selected choices, substitute another proposed candidate, synthesize a
replacement, or block the item because the source stem/gold is unsound.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from build_frontierscience_mc_distractors import (
    DEFAULT_MODEL,
    append_jsonl,
    extract_tagged_values,
    flatten_candidates,
    load_rows,
    normalized_text,
    read_jsonl_by_key,
    structured_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_frontierscience_mc_distractors")

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gold_verdict": {
            "type": "string",
            "enum": ["valid", "likely_valid", "ambiguous", "incorrect"],
        },
        "gold_analysis": {"type": "string", "maxLength": 1400},
        "stem_verdict": {
            "type": "string",
            "enum": ["clear", "minor_issue", "ambiguous", "broken"],
        },
        "stem_analysis": {"type": "string", "maxLength": 1000},
        "final_choices": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "answer": {"type": "string"},
                    "definitely_incorrect": {"type": "boolean"},
                    "plausible": {"type": "boolean"},
                    "same_answer_family": {"type": "boolean"},
                    "artifact_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "error_mechanism": {"type": "string", "maxLength": 500},
                    "analysis": {"type": "string", "maxLength": 700},
                },
                "required": [
                    "source_id",
                    "answer",
                    "definitely_incorrect",
                    "plausible",
                    "same_answer_family",
                    "artifact_risk",
                    "error_mechanism",
                    "analysis",
                ],
                "additionalProperties": False,
            },
        },
        "outcome": {
            "type": "string",
            "enum": ["clean_as_is", "repaired", "blocked_gold_or_stem"],
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string", "maxLength": 700},
            "maxItems": 8,
        },
    },
    "required": [
        "gold_verdict",
        "gold_analysis",
        "stem_verdict",
        "stem_analysis",
        "final_choices",
        "outcome",
        "warnings",
    ],
    "additionalProperties": False,
}

AUDITOR_PROMPT = """You are the senior second-pass adjudicator for an expert science olympiad
multiple-choice conversion. Independently solve enough of the problem to assess the reference
answer; prior reviewers may be wrong. Then produce the best three distractors.

A usable set requires a valid or defensibly likely-valid gold and a stem whose intended answer is
unique. Each distractor must be definitely incorrect, scientifically plausible, in the same answer
family and specificity as the gold, and low in formatting/style artifact risk. The three choices
must be distinct and encode distinct mistakes. At least two must survive superficial dimensional,
syntactic, stoichiometric, or ontological checks appropriate to the subject. Reject answers that
become correct under ordinary conventions, aliases, rounding, or reasonable interpretations.

Warnings about duplicate or poor *unselected* candidates do not disqualify an otherwise clean set.
If the prior three choices are clean, return them exactly and use clean_as_is. Otherwise, you may
choose a different supplied candidate, minimally rewrite one, or create a new distractor; use a
source_id such as audit-new-1 for a new answer and repaired as the outcome. Preserve a uniform
representation across all four displayed choices. If the gold is incorrect/ambiguous or the stem
does not determine a unique answer, use blocked_gold_or_stem. Still return the least problematic
three choices for diagnostic continuity, but do not pretend the set is usable.

For multi-structure chemistry answers, preserve all labeled components and use one SMILES per
component in every choice. Keep the response terse and structured.
"""


def audit_input(
    row: dict[str, Any],
    triage: dict[str, Any],
    review: dict[str, Any],
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "subject": row["subject"],
        "problem": row["problem"],
        "reference_answer": row["answer"],
        "prior_tier": triage["tier"],
        "prior_selected": triage["selected"],
        "prior_warnings": triage["api_warnings"],
        "prior_gold_assessment": review["result"]["gold_assessment"],
        "candidate_pool": candidates,
    }
    return [
        {"role": "developer", "content": AUDITOR_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def display_choices(row: dict[str, Any], answers: list[str]) -> tuple[str, list[str], bool]:
    """Normalize structural answers when all choices provide matching SMILES counts."""
    gold_smiles = extract_tagged_values(row["answer"], "SMILES")
    if not gold_smiles:
        return row["answer"], answers, True
    candidate_smiles = [extract_tagged_values(answer, "SMILES") for answer in answers]
    if not all(len(values) == len(gold_smiles) for values in candidate_smiles):
        return row["answer"], answers, False

    if len(gold_smiles) == 1:
        return (
            f"<SMILES>{gold_smiles[0]}</SMILES>",
            [f"<SMILES>{values[0]}</SMILES>" for values in candidate_smiles],
            True,
        )

    def labeled(values: list[str]) -> str:
        return "; ".join(
            f"B{index}: <SMILES>{value}</SMILES>" for index, value in enumerate(values, start=1)
        )

    return labeled(gold_smiles), [labeled(values) for values in candidate_smiles], True


def materialize_v2(
    rows: list[dict[str, Any]],
    triage_records: dict[str, dict[str, Any]],
    audit_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        item_id = row["task_group_id"]
        prior = triage_records[item_id]
        if prior["tier"] == "auto_clean":
            record = {
                **prior,
                "prior_tier": prior["tier"],
                "tier": "strict_clean",
                "audit": None,
            }
            records.append(record)
            continue

        audit = audit_records[item_id]["result"]
        choices = audit["final_choices"]
        answers = [choice["answer"].strip() for choice in choices]
        distinct = len({normalized_text(answer) for answer in answers}) == 3
        display_gold, display_distractors, representation_ok = display_choices(row, answers)
        choice_checks = all(
            choice["definitely_incorrect"]
            and choice["plausible"]
            and choice["same_answer_family"]
            and choice["artifact_risk"] == "low"
            for choice in choices
        )
        gold_ok = audit["gold_verdict"] in {"valid", "likely_valid"}
        stem_ok = audit["stem_verdict"] in {"clear", "minor_issue"}
        outcome_ok = audit["outcome"] in {"clean_as_is", "repaired"}
        usable = all([distinct, representation_ok, choice_checks, gold_ok, stem_ok, outcome_ok])
        if usable and audit["outcome"] == "clean_as_is":
            tier = "audit_clean"
        elif usable:
            tier = "repaired_clean"
        else:
            tier = "blocked_source"
        records.append(
            {
                "item_id": item_id,
                "subject": row["subject"],
                "tier": tier,
                "prior_tier": prior["tier"],
                "problem": row["problem"],
                "gold": row["answer"],
                "display_gold": display_gold,
                "display_distractors": display_distractors,
                "selected": choices,
                "audit": audit,
                "checks": {
                    "distinct": distinct,
                    "representation_ok": representation_ok,
                    "all_choices_clean": choice_checks,
                    "gold_ok": gold_ok,
                    "stem_ok": stem_ok,
                    "outcome_ok": outcome_ok,
                },
            }
        )
    return records


async def async_main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    rows = load_rows(args.input)
    triage_path = args.artifact_dir / "triage.jsonl"
    review_path = args.artifact_dir / "reviews.jsonl"
    proposal_path = args.artifact_dir / "proposals.jsonl"
    audit_path = args.artifact_dir / "audits.jsonl"
    attempt_path = args.artifact_dir / "audit_attempts.jsonl"
    v2_path = args.artifact_dir / "triage_v2.jsonl"

    triage_records = read_jsonl_by_key(triage_path, "item_id")
    review_records = read_jsonl_by_key(review_path, "item_id")
    proposal_records = read_jsonl_by_key(proposal_path, "request_id")
    audit_records = read_jsonl_by_key(audit_path, "item_id")
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(timeout=300.0)

    async def run_one(row: dict[str, Any]) -> None:
        item_id = row["task_group_id"]
        triage = triage_records[item_id]
        if triage["tier"] == "auto_clean" or item_id in audit_records:
            return
        proposals = [
            proposal_records[f"{item_id}:proposal:{index}"]
            for index in range(1, args.proposals_per_item + 1)
        ]
        candidates = flatten_candidates(proposals)
        effort = "medium" if triage["tier"] == "manual_review" else "high"
        logger.info("Auditing %s (%s, %s effort)", item_id, triage["tier"], effort)
        result, metadata = await structured_response(
            client,
            semaphore=semaphore,
            request_id=f"{item_id}:audit",
            attempt_log_path=attempt_path,
            model=args.model,
            reasoning_effort=effort,
            messages=audit_input(row, triage, review_records[item_id], candidates),
            schema_name="frontierscience_distractor_audit",
            schema=AUDIT_SCHEMA,
            max_output_tokens=18_000,
        )
        record = {
            "item_id": item_id,
            "subject": row["subject"],
            "prior_tier": triage["tier"],
            "result": result,
            "metadata": metadata,
        }
        append_jsonl(audit_path, record)
        audit_records[item_id] = record

    await asyncio.gather(*(run_one(row) for row in rows))
    audit_records = read_jsonl_by_key(audit_path, "item_id")
    v2_records = materialize_v2(rows, triage_records, audit_records)
    with v2_path.open("w") as handle:
        for record in v2_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for record in v2_records:
        counts[record["tier"]] = counts.get(record["tier"], 0) + 1
    logger.info("Wrote %s with tiers %s", v2_path, counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--proposals-per-item", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()

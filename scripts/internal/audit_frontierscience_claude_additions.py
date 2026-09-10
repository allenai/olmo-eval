"""Adjudicate source-answer concerns added by an independent Claude review."""

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
    read_jsonl_by_key,
    structured_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_frontierscience_claude_additions")

CLAIMS = {
    "85b4f862-d881-4a79-8c5d-3e927b486b71": (
        "The Fe/Fe2O3 percentages are an arithmetic error: standard atomic weights give about "
        "76.97% Fe and 23.03% Fe2O3, not 76.89% and 23.11%."
    ),
    "ae667a28-9659-4f33-a51e-7a19f133e111": (
        "The magnetic thermodynamic identity may be missing an overall mu_0 in SI units. Decide "
        "whether this is wrong or merely an unstated unit/work convention."
    ),
    "2c46387b-36e6-43cb-9e73-dbf1341bfafd": (
        "The figure-eight orbit question may be under-specified. Prior review derived 1.56 by "
        "using equal-mass center-of-mass symmetry at P; assess whether that follows from the stem."
    ),
    "f3ba1aae-2fc3-4d9b-a5a3-42bb91de4d7d": (
        "The stem calls the oscillation period omega. The answer has dimensions of time. Decide "
        "whether this is only a defined-symbol mislabel or makes the answer/key invalid."
    ),
    "319ae242-19fc-4542-b75a-d32b325ea923": (
        "The triafulvene answer was flagged because the stated [4+4] dimer and tetrasubstituted "
        "cyclohexane description may not match triafulvene dimerization. Do not assume niche "
        "literature facts that cannot be established from the supplied evidence."
    ),
    "fcbd76ad-1b45-4f40-ad9d-d2bc5016f02c": (
        "The source paper (Lesbats et al., Nature 2025, DOI 10.1038/s41586-025-08629-4) "
        "attributes the viability result to killed bacteria providing an additional nutrient "
        "source, including amino acids. It separately attributes lower ROS to provision of "
        "metabolic intermediates for antioxidant responses. The gold 'Increased antioxidant "
        "availability' may conflate those two conclusions."
    ),
    "b188cad8-92b9-4c72-98a2-b78e04b764f1": (
        "Negative control: an independent reviewer initially flagged but then cleared A4B9X20. "
        "Check site sharing directly and avoid manufacturing an issue."
    ),
}

AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gold_status": {
            "type": "string",
            "enum": ["valid", "ambiguous", "incorrect", "needs_domain_evidence"],
        },
        "stem_status": {
            "type": "string",
            "enum": ["clear", "minor_notation_issue", "ambiguous", "broken"],
        },
        "claim_verdict": {
            "type": "string",
            "enum": ["confirm_new_flag", "retain_clean", "external_expert_only"],
        },
        "derivation_or_evidence": {"type": "string", "maxLength": 1800},
        "brief_reason": {"type": "string", "maxLength": 700},
        "corrected_answer": {"type": "string", "maxLength": 1000},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "gold_status",
        "stem_status",
        "claim_verdict",
        "derivation_or_evidence",
        "brief_reason",
        "corrected_answer",
        "confidence",
    ],
    "additionalProperties": False,
}

PROMPT = """Act as a conservative senior adjudicator for a science benchmark. Independently
analyze the literal stem and supplied reference answer, then assess the new concern. A notation or
unit-convention issue is not automatically a wrong key if the stem defines the symbol or an ordinary
convention makes the answer valid. Conversely, do not excuse arithmetic errors or replace literal
experimental conclusions with recoverable author intent. For literature-specific chemistry or
biology, use only evidence supplied in the request; choose external_expert_only when it is
insufficient. The negative control should be cleared if direct calculation supports it. Return terse
structured data.
"""


async def async_main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    source_records = read_jsonl_by_key(args.input, "task_group_id")
    existing = read_jsonl_by_key(args.output, "item_id")
    semaphore = asyncio.Semaphore(args.concurrency)
    client = AsyncOpenAI(timeout=300.0)

    async def run_one(item_id: str, claim: str) -> None:
        if item_id in existing:
            return
        source = source_records[item_id]
        content = {
            "subject": source["subject"],
            "problem": source["problem"],
            "supplied_reference_answer": source["reference_answer"],
            "new_concern": claim,
        }
        logger.info("Auditing %s", item_id)
        result, metadata = await structured_response(
            client,
            semaphore=semaphore,
            request_id=f"{item_id}:claude-addition-audit",
            attempt_log_path=args.attempts,
            model=args.model,
            reasoning_effort="high",
            messages=[
                {"role": "developer", "content": PROMPT},
                {"role": "user", "content": json.dumps(content, ensure_ascii=False, indent=2)},
            ],
            schema_name="frontierscience_claude_addition_audit",
            schema=AUDIT_SCHEMA,
            max_output_tokens=18_000,
        )
        record = {
            "item_id": item_id,
            "subject": source["subject"],
            "claim": claim,
            "result": result,
            "metadata": metadata,
        }
        append_jsonl(args.output, record)
        existing[item_id] = record

    await asyncio.gather(*(run_one(item_id, claim) for item_id, claim in CLAIMS.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()

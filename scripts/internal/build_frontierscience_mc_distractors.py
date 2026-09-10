"""Generate and review candidate distractors for FrontierScience Olympiad MC.

This is an offline curation helper, not part of routine evaluation. It sends one
problem per API request, keeps the raw structured proposals and reviews, and
writes a human-readable pilot report. Outputs default to the gitignored
``build/frontierscience_mc`` directory.

The script never reads or writes the value of ``OPENAI_API_KEY`` itself; the
OpenAI client reads it from the process environment.

Example:
    uv run --extra clients python scripts/internal/build_frontierscience_mc_distractors.py \
        --input /path/to/frontierscience/olympiad/test.jsonl \
        --pilot
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_frontierscience_mc_distractors")

FRONTIERSCIENCE_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"
DEFAULT_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "build/frontierscience_mc"

# Chosen to exercise numerical and symbolic physics, numerical and identity
# chemistry, structural chemistry, and several biology answer shapes.
PILOT_IDS = (
    "516da073-8d2a-4db5-a739-c752655ec14c",  # physics: vector field
    "bb0539ef-d9fd-4215-bf16-b0eca44a8778",  # physics: numerical
    "b90fc8dc-2fca-4ff7-9c9f-a2e6a0ac7812",  # physics: short symbolic
    "f9fdb988-7a10-4510-a8ab-d011e663be3f",  # physics: piecewise field
    "1440c195-fba2-48dc-a03d-7b9420891daf",  # chemistry: precipitation
    "8f121749-9809-45f1-bcb8-7118ccb5d4cc",  # chemistry: identification
    "469cc7c7-ae71-4ea2-bc13-3af3770fd8da",  # chemistry: balancing
    "8be6ce44-a0c8-4636-80ca-2667dcc9d2af",  # chemistry: structure/NMR
    "0949d931-d0ad-4146-86e7-ef725919e3de",  # chemistry: reaction product
    "111d5872-2705-42c0-8df4-0fb386ba7084",  # biology: mechanism
    "4ee82eb3-a457-42c5-9c62-5730d41f67a6",  # biology: pathway
    "1510d6bd-6785-456d-8988-b9382a3aef9c",  # biology: ordered list
)

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "enum": ["physics", "chemistry", "biology"]},
        "answer_type": {
            "type": "string",
            "enum": [
                "numeric",
                "symbolic",
                "equation_or_reaction",
                "chemical_identity",
                "chemical_structure",
                "biological_entity_or_process",
                "ordered_or_multi_part",
                "other",
            ],
        },
        "solution_outline": {"type": "string"},
        "gold_checks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 8,
        },
        "candidates": {
            "type": "array",
            "minItems": 6,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "error_mechanism": {"type": "string"},
                    "distance": {
                        "type": "string",
                        "enum": ["single_error", "conceptual", "distant_but_plausible"],
                    },
                    "why_plausible": {"type": "string"},
                    "why_incorrect": {"type": "string"},
                    "format_check": {"type": "string"},
                    "equivalence_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": [
                    "answer",
                    "error_mechanism",
                    "distance",
                    "why_plausible",
                    "why_incorrect",
                    "format_check",
                    "equivalence_risk",
                ],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": [
        "subject",
        "answer_type",
        "solution_outline",
        "gold_checks",
        "candidates",
        "warnings",
    ],
    "additionalProperties": False,
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gold_assessment": {"type": "string", "maxLength": 1000},
        "candidate_reviews": {
            "type": "array",
            "minItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["accept", "reject", "revise"]},
                    "plausible": {"type": "boolean"},
                    "definitely_incorrect": {"type": "boolean"},
                    "same_answer_family": {"type": "boolean"},
                    "artifact_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "analysis": {"type": "string", "maxLength": 500},
                    "revised_answer": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "verdict",
                    "plausible",
                    "definitely_incorrect",
                    "same_answer_family",
                    "artifact_risk",
                    "analysis",
                    "revised_answer",
                ],
                "additionalProperties": False,
            },
        },
        "selected": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "answer": {"type": "string"},
                    "error_mechanism": {"type": "string", "maxLength": 400},
                    "selection_rationale": {"type": "string", "maxLength": 500},
                },
                "required": [
                    "candidate_id",
                    "answer",
                    "error_mechanism",
                    "selection_rationale",
                ],
                "additionalProperties": False,
            },
        },
        "item_status": {
            "type": "string",
            "enum": ["ready_for_human_review", "needs_specialist_review", "no_viable_set"],
        },
        "warnings": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": [
        "gold_assessment",
        "candidate_reviews",
        "selected",
        "item_status",
        "warnings",
    ],
    "additionalProperties": False,
}

COMMON_PROPOSER_PROMPT = """You are constructing a high-quality multiple-choice adaptation of
an expert science olympiad benchmark. The supplied reference answer is the gold answer. Solve the
problem independently enough to understand why the gold is correct, then propose exactly eight
incorrect answer choices.

Each distractor must correspond to a recognizable wrong solution path. Prefer a diverse set of
single-error, conceptual, and more distant-but-still-plausible mistakes. A distractor must be
unambiguously inequivalent to the gold under reasonable rounding, algebraic reformulation, unit
conversion, chemical aliases, or biological synonyms. Match the gold's answer family, variables,
units, approximate length, syntax, and specificity. Do not create a deliberately silly option.
Include at least two candidates that satisfy the problem's superficial constraints and would tempt
a competent solver who selected the wrong model or made a subtle derivation error. Do not fill the
set with options defeated immediately by a stated boundary condition, definition, or observation.
Do not alter the question or criticize its writing except in the warnings field.

The final evaluation scores choice labels, so answer length is not scored directly; nevertheless,
surface polish, verbosity, and malformed notation can reveal the gold. Return structured data only.
"""

DOMAIN_PROPOSER_PROMPTS = {
    "physics": """For physics, check dimensions, signs, vector directions, boundary conditions,
limiting cases, and algebraic equivalence. Good distractors arise from explicit derivation errors,
wrong constraints, or a nearby physical model. Keep dimensionally invalid answers only when the
mistake is exceptionally recognizable; ordinarily all candidates should have correct dimensions.
""",
    "chemistry": """For chemistry, preserve requested units and precision. Check stoichiometry,
charge, oxidation states, equilibria, stereochemistry, and molecular identity as applicable. For
structures, every name, formula, InChI, and SMILES included in one candidate must identify the same
molecule. If the gold includes a SMILES representation, every structural candidate must include a
valid SMILES representation; make the candidate's answer exactly `<SMILES>...</SMILES>` so all
display choices can be normalized to SMILES alone. Prefer nearby valid species, products, or isomers
to nonsense compounds.
""",
    "biology": """For biology, use competing mechanisms or entities at the same ontological level
and specificity. Good errors include neighboring pathways, upstream/downstream confusion,
cause-versus-correlate errors, assay artifacts, and order/cardinality mistakes. Reject synonyms,
partially correct broader answers, and mechanisms that could reasonably coexist as equally correct.
""",
}

COMMON_REVIEWER_PROMPT = """Act as an independent, adversarial reviewer of proposed multiple-choice
distractors for an expert science olympiad problem. First solve or analyze the problem yourself and
verify the supplied gold. Then assess every candidate without trusting its source.

A selectable distractor must be scientifically plausible, definitely incorrect, inequivalent to
the gold, in the same answer family and specificity, and free from style or formatting artifacts.
Select exactly three candidates with distinct error mechanisms. You may make a minimal correction
to a candidate and mark it revise. If a safe three-choice set cannot be established, still identify
the least problematic three but set item_status accordingly and explain why in warnings.

Apply a demanding difficulty floor. At least two selected choices should survive superficial unit,
syntax, and stem-restatement checks and plausibly tempt a competent but mistaken solver. Select no
more than one answer that simply ignores the central phenomenon or contradicts an explicit
definition, geometry, direction, boundary condition, or observation. Set ready_for_human_review
only when all three selections meet the scientific and artifact requirements and this difficulty
floor; otherwise use needs_specialist_review.

Keep the review terse. Use the analysis field for the decisive scientific or artifact issue, not a
full derivation. Do not repeat the candidate answer in analysis. Leave revised_answer empty unless
the verdict is revise.
"""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"problem", "answer", "subject", "task_group_id"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def select_rows(
    rows: list[dict[str, Any]], *, pilot: bool, limit: int | None
) -> list[dict[str, Any]]:
    if pilot:
        by_id = {row["task_group_id"]: row for row in rows}
        missing = [item_id for item_id in PILOT_IDS if item_id not in by_id]
        if missing:
            raise ValueError(f"Pilot IDs absent from input: {missing}")
        selected = [by_id[item_id] for item_id in PILOT_IDS]
    else:
        selected = rows
    return selected[:limit] if limit is not None else selected


def proposal_input(row: dict[str, Any], pass_index: int) -> list[dict[str, str]]:
    subject = row["subject"]
    prompt = COMMON_PROPOSER_PROMPT + "\n" + DOMAIN_PROPOSER_PROMPTS[subject]
    content = (
        f"Independent proposal pass: {pass_index}\n"
        f"Subject: {subject}\n\n"
        f"Problem:\n{row['problem']}\n\n"
        f"Reference answer:\n{row['answer']}"
    )
    return [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": content},
    ]


def flatten_candidates(proposals: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    seen: set[str] = set()
    for proposal in proposals:
        pass_index = int(proposal["pass_index"])
        for index, candidate in enumerate(proposal["result"]["candidates"], start=1):
            answer = candidate["answer"].strip()
            normalized = " ".join(answer.split()).casefold()
            if not answer or normalized in seen:
                continue
            seen.add(normalized)
            flattened.append(
                {
                    "candidate_id": f"p{pass_index}-c{index}",
                    "answer": answer,
                }
            )
    return flattened


def review_input(row: dict[str, Any], candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    subject = row["subject"]
    prompt = COMMON_REVIEWER_PROMPT + "\n" + DOMAIN_PROPOSER_PROMPTS[subject]
    content = (
        f"Subject: {subject}\n\n"
        f"Problem:\n{row['problem']}\n\n"
        f"Reference answer:\n{row['answer']}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": content},
    ]


def read_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record[key])] = record
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


async def structured_response(
    client: Any,
    *,
    semaphore: asyncio.Semaphore,
    request_id: str,
    attempt_log_path: Path,
    model: str,
    reasoning_effort: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    max_output_tokens: int,
    max_attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with semaphore:
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            effort = reasoning_effort if attempt == 1 else "medium"
            response = None
            try:
                response = await client.responses.create(
                    model=model,
                    input=messages,
                    reasoning={"effort": effort},
                    text={
                        "verbosity": "low",
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    max_output_tokens=max_output_tokens,
                )
                metadata = {
                    "response_id": response.id,
                    "model": response.model,
                    "usage": usage_dict(response),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "reasoning_effort": effort,
                }
                result = json.loads(response.output_text)
                append_jsonl(
                    attempt_log_path,
                    {
                        "request_id": request_id,
                        "attempt": attempt,
                        "outcome": "success",
                        "metadata": metadata,
                    },
                )
                return result, metadata
            except Exception as exc:
                attempt_record: dict[str, Any] = {
                    "request_id": request_id,
                    "attempt": attempt,
                    "outcome": "failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "reasoning_effort": effort,
                }
                if response is not None:
                    attempt_record["response_id"] = getattr(response, "id", None)
                    attempt_record["status"] = getattr(response, "status", None)
                    attempt_record["incomplete_details"] = str(
                        getattr(response, "incomplete_details", None)
                    )
                    attempt_record["usage"] = usage_dict(response)
                    attempt_record["partial_output"] = getattr(response, "output_text", "")
                append_jsonl(attempt_log_path, attempt_record)
                if attempt >= max_attempts:
                    raise
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "API attempt %d/%d failed: %s; retrying in %ds",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def run_proposals(
    client: Any,
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    attempt_log_path: Path,
    model: str,
    reasoning_effort: str,
    second_reasoning_effort: str,
    proposals_per_item: int,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    existing = read_jsonl_by_key(output_path, "request_id")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(row: dict[str, Any], pass_index: int) -> None:
        item_id = row["task_group_id"]
        request_id = f"{item_id}:proposal:{pass_index}"
        if request_id in existing:
            return
        logger.info("Generating %s", request_id)
        pass_effort = reasoning_effort if pass_index == 1 else second_reasoning_effort
        result, metadata = await structured_response(
            client,
            semaphore=semaphore,
            request_id=request_id,
            attempt_log_path=attempt_log_path,
            model=model,
            reasoning_effort=pass_effort,
            messages=proposal_input(row, pass_index),
            schema_name="frontierscience_distractor_proposal",
            schema=PROPOSAL_SCHEMA,
            max_output_tokens=12_000,
        )
        record = {
            "request_id": request_id,
            "item_id": item_id,
            "pass_index": pass_index,
            "subject": row["subject"],
            "result": result,
            "metadata": metadata,
        }
        append_jsonl(output_path, record)
        existing[request_id] = record

    await asyncio.gather(
        *(
            run_one(row, pass_index)
            for row in rows
            for pass_index in range(1, proposals_per_item + 1)
        )
    )
    return read_jsonl_by_key(output_path, "request_id")


async def run_reviews(
    client: Any,
    *,
    rows: list[dict[str, Any]],
    proposal_records: dict[str, dict[str, Any]],
    output_path: Path,
    attempt_log_path: Path,
    model: str,
    reasoning_effort: str,
    proposals_per_item: int,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    existing = read_jsonl_by_key(output_path, "item_id")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(row: dict[str, Any]) -> None:
        item_id = row["task_group_id"]
        if item_id in existing:
            return
        proposals = [
            proposal_records[f"{item_id}:proposal:{pass_index}"]
            for pass_index in range(1, proposals_per_item + 1)
        ]
        candidates = flatten_candidates(proposals)
        logger.info("Reviewing %s (%d unique candidates)", item_id, len(candidates))
        result, metadata = await structured_response(
            client,
            semaphore=semaphore,
            request_id=f"{item_id}:review",
            attempt_log_path=attempt_log_path,
            model=model,
            reasoning_effort=reasoning_effort,
            messages=review_input(row, candidates),
            schema_name="frontierscience_distractor_review",
            schema=REVIEW_SCHEMA,
            max_output_tokens=16_000,
        )
        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        unknown = {
            selected["candidate_id"]
            for selected in result["selected"]
            if selected["candidate_id"] not in candidate_ids
        }
        if unknown:
            result["warnings"].append(f"Reviewer selected unknown candidate IDs: {sorted(unknown)}")
            result["item_status"] = "needs_specialist_review"
        record = {
            "item_id": item_id,
            "subject": row["subject"],
            "gold": row["answer"],
            "candidate_count": len(candidates),
            "result": result,
            "metadata": metadata,
        }
        append_jsonl(output_path, record)
        existing[item_id] = record

    await asyncio.gather(*(run_one(row) for row in rows))
    return read_jsonl_by_key(output_path, "item_id")


def sum_usage(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("metadata", {}).get("usage", {})
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def extract_tagged_values(text: str, tag: str) -> list[str]:
    """Extract XML-like chemistry fields, accepting HTML-escaped dataset tags."""
    unescaped = html.unescape(text)
    pattern = re.compile(rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>", re.I | re.S)
    return [match.strip() for match in pattern.findall(unescaped) if match.strip()]


def normalized_text(text: str) -> str:
    return " ".join(html.unescape(text).split()).casefold()


def build_triage_record(row: dict[str, Any], review_record: dict[str, Any]) -> dict[str, Any]:
    result = review_record["result"]
    selected = result["selected"]
    reviews = {review["candidate_id"]: review for review in result["candidate_reviews"]}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, severity: str, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "severity": severity, "detail": detail})

    answers = [choice["answer"].strip() for choice in selected]
    normalized_answers = [normalized_text(answer) for answer in answers]
    check("three_choices", len(answers) == 3, "hard", f"found {len(answers)}")
    check(
        "distinct_choices",
        len(set(normalized_answers)) == len(normalized_answers),
        "hard",
        "selected answers must be distinct",
    )
    gold_normalized = normalized_text(row["answer"])
    check(
        "not_literal_gold",
        all(answer != gold_normalized for answer in normalized_answers),
        "hard",
        "no selected answer may literally equal the gold",
    )

    selected_reviews = [reviews.get(choice["candidate_id"]) for choice in selected]
    check(
        "review_records_present",
        all(review is not None for review in selected_reviews),
        "hard",
        "every selected candidate must have a review",
    )
    present_reviews = [review for review in selected_reviews if review is not None]
    check(
        "reviewer_definitely_incorrect",
        len(present_reviews) == 3
        and all(review["definitely_incorrect"] for review in present_reviews),
        "hard",
        "reviewer must mark all three definitely incorrect",
    )
    check(
        "reviewer_plausible",
        len(present_reviews) == 3 and all(review["plausible"] for review in present_reviews),
        "soft",
        "reviewer must mark all three plausible",
    )
    check(
        "same_answer_family",
        len(present_reviews) == 3
        and all(review["same_answer_family"] for review in present_reviews),
        "soft",
        "reviewer must mark all three in the same answer family",
    )
    check(
        "low_artifact_risk",
        len(present_reviews) == 3
        and all(review["artifact_risk"] == "low" for review in present_reviews),
        "soft",
        "strict tier requires low artifact risk for every selection",
    )

    gold_smiles = extract_tagged_values(row["answer"], "SMILES")
    display_gold = row["answer"]
    display_distractors = answers
    multi_structure = len(gold_smiles) > 1
    if len(gold_smiles) == 1:
        candidate_smiles = [extract_tagged_values(answer, "SMILES") for answer in answers]
        smiles_complete = all(len(values) == 1 for values in candidate_smiles)
        check(
            "single_smiles_representation",
            smiles_complete,
            "soft",
            "single-structure chemistry choices should all provide one SMILES",
        )
        if smiles_complete:
            display_gold = f"<SMILES>{gold_smiles[0]}</SMILES>"
            display_distractors = [f"<SMILES>{values[0]}</SMILES>" for values in candidate_smiles]
    elif multi_structure:
        check(
            "multi_structure_representation",
            False,
            "specialist",
            "multi-structure answers require item-specific normalization and validation",
        )

    warnings_text = " ".join(result["warnings"])
    gold_issue = bool(
        re.search(
            r"(?:gold|reference answer).{0,80}(?:wrong|suspect|error|incorrect|inconsistent)",
            warnings_text,
            re.I,
        )
    )
    check("no_gold_warning", not gold_issue, "gold", "reviewer must not flag a suspect gold")
    check(
        "no_reviewer_warnings",
        not result["warnings"],
        "soft",
        f"{len(result['warnings'])} reviewer warning(s)",
    )
    api_ready = result["item_status"] == "ready_for_human_review"
    check("api_ready", api_ready, "specialist", result["item_status"])

    failed = [item for item in checks if not item["passed"]]
    if gold_issue:
        tier = "gold_issue"
    elif any(item["severity"] in {"hard", "specialist"} for item in failed):
        tier = "specialist_required"
    elif failed:
        tier = "manual_review"
    else:
        tier = "auto_clean"

    return {
        "item_id": row["task_group_id"],
        "subject": row["subject"],
        "tier": tier,
        "problem": row["problem"],
        "gold": row["answer"],
        "display_gold": display_gold,
        "display_distractors": display_distractors,
        "selected": selected,
        "api_status": result["item_status"],
        "api_warnings": result["warnings"],
        "checks": checks,
    }


def write_triage(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    review_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records = {
        row["task_group_id"]: build_triage_record(row, review_records[row["task_group_id"]])
        for row in rows
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            record = records[row["task_group_id"]]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote triage to %s", path)
    return records


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    proposal_records: dict[str, dict[str, Any]],
    review_records: dict[str, dict[str, Any]],
    triage_records: dict[str, dict[str, Any]],
    model: str,
    proposals_per_item: int,
) -> None:
    proposal_subset = [
        proposal_records[f"{row['task_group_id']}:proposal:{pass_index}"]
        for row in rows
        for pass_index in range(1, proposals_per_item + 1)
    ]
    review_subset = [review_records[row["task_group_id"]] for row in rows]
    proposal_usage = sum_usage(proposal_subset)
    review_usage = sum_usage(review_subset)
    statuses: dict[str, int] = {}
    tiers: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    for record in review_subset:
        result = record["result"]
        status = result["item_status"]
        statuses[status] = statuses.get(status, 0) + 1
        tier = triage_records[record["item_id"]]["tier"]
        tiers[tier] = tiers.get(tier, 0) + 1
        for review in result["candidate_reviews"]:
            verdict = review["verdict"]
            verdicts[verdict] = verdicts.get(verdict, 0) + 1

    lines = [
        "# FrontierScience Olympiad MC distractor generation report",
        "",
        f"- Dataset revision: `{FRONTIERSCIENCE_REVISION}`",
        f"- Model: `{model}`",
        f"- Items: {len(rows)}",
        f"- Proposal passes per item: {proposals_per_item}",
        f"- Proposal token usage: `{proposal_usage}`",
        f"- Review token usage: `{review_usage}`",
        f"- Item statuses: `{statuses}`",
        f"- Triage tiers: `{tiers}`",
        f"- Candidate verdicts: `{verdicts}`",
        "",
    ]
    for row in rows:
        item_id = row["task_group_id"]
        record = review_records[item_id]
        result = record["result"]
        tier = triage_records[item_id]["tier"]
        lines.extend(
            [
                f"## {row['subject'].title()}: `{item_id}`",
                "",
                f"Status: **{result['item_status']}**",
                "",
                f"Triage: **{tier}**",
                "",
                f"Gold: {row['answer']}",
                "",
                "Selected distractors:",
                "",
            ]
        )
        for selected in result["selected"]:
            lines.append(f"- {selected['answer']}")
            lines.append(f"  - Error: {selected['error_mechanism']}")
            lines.append(f"  - Rationale: {selected['selection_rationale']}")
        if result["warnings"]:
            lines.extend(["", "Warnings:", ""])
            lines.extend(f"- {warning}" for warning in result["warnings"])
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("Wrote report to %s", path)


async def async_main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    rows = select_rows(load_rows(args.input), pilot=args.pilot, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = args.output_dir / "proposals.jsonl"
    review_path = args.output_dir / "reviews.jsonl"
    attempt_log_path = args.output_dir / "attempts.jsonl"
    triage_path = args.output_dir / "triage.jsonl"
    report_path = args.output_dir / "report.md"

    client = AsyncOpenAI(timeout=300.0)
    proposals = await run_proposals(
        client,
        rows=rows,
        output_path=proposal_path,
        attempt_log_path=attempt_log_path,
        model=args.model,
        reasoning_effort=args.proposer_effort,
        second_reasoning_effort=args.second_proposer_effort,
        proposals_per_item=args.proposals_per_item,
        concurrency=args.concurrency,
    )
    reviews = await run_reviews(
        client,
        rows=rows,
        proposal_records=proposals,
        output_path=review_path,
        attempt_log_path=attempt_log_path,
        model=args.model,
        reasoning_effort=args.reviewer_effort,
        proposals_per_item=args.proposals_per_item,
        concurrency=args.concurrency,
    )
    triage = write_triage(triage_path, rows=rows, review_records=reviews)
    write_report(
        report_path,
        rows=rows,
        proposal_records=proposals,
        review_records=reviews,
        triage_records=triage,
        model=args.model,
        proposals_per_item=args.proposals_per_item,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Olympiad test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pilot", action="store_true", help="Use the fixed 12-item pilot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--proposals-per-item", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--proposer-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument(
        "--second-proposer-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument(
        "--reviewer-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
    )
    args = parser.parse_args()
    if args.proposals_per_item < 1:
        parser.error("--proposals-per-item must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()

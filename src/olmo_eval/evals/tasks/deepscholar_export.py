"""Convert saved deepscholar_bench predictions into DeepScholar-Bench folders.

The upstream scorer reads one directory per query holding ``intro.md``,
``final_report.md`` and ``paper.csv``; this script writes exactly that shape
from a run's ``*-predictions.jsonl``, so generation and scoring stay two
separate passes and a scoring change never requires re-running the model.

``intro.md`` and ``final_report.md`` are both the answer text, byte for byte,
matching what lit-agents' exporter writes. ``paper.csv`` holds every arXiv
source the search tool showed the agent -- not only the cited ones -- because
the contract the scorer checks is that cited IDs are a subset of the CSV.

Publication dates come from the ``Published`` line the search tool renders: day
precision when Semantic Scholar dated the paper, month precision when only the
arXiv ID could. The distinction matters because the contract rejects a
month-precise source dated inside the cutoff's own month, so reporting a known
day as a month would silently discard citable work.

Usage:
    python -m olmo_eval.evals.tasks.deepscholar_export \\
        --predictions <run>/deepscholar_bench-predictions.jsonl \\
        --output <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.tasks.deepscholar_bench import sources_from_trajectory
from olmo_eval.harness.tools.search import date_from_arxiv_id

logger = logging.getLogger(__name__)

PAPER_CSV_FIELDS = ("id", "title", "snippet", "published_date", "date_precision", "paper_id")
EXPORTED_FILES = ("intro.md", "final_report.md", "paper.csv")


def read_predictions(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the prediction rows of a JSONL file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable prediction on line %d", line_number)


def answer_text(prediction: Mapping[str, Any]) -> str:
    """The final answer a prediction row carries, or "" when it has none."""
    final_output = prediction.get("final_output")
    if isinstance(final_output, str) and final_output.strip():
        return final_output
    outputs = prediction.get("model_output") or []
    if outputs and isinstance(outputs[0], Mapping):
        text = outputs[0].get("text")
        if isinstance(text, str):
            return text
    return ""


def query_id(prediction: Mapping[str, Any]) -> str:
    """The upstream positional query ID this prediction belongs to.

    Folders are named for it, and upstream identifies a query by its row
    position in the dataset CSV, so a row without one cannot be exported.
    """
    native_id = prediction.get("native_id")
    if isinstance(native_id, str) and native_id.isdigit():
        return native_id
    doc_id = prediction.get("doc_id")
    return str(doc_id) if isinstance(doc_id, int) else ""


def build_paper_rows(sources: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Build paper.csv rows, dropping sources nothing can date.

    Dates come from the search tool's Published line, falling back to the month
    the arXiv ID encodes. A row the upstream contract cannot date is one it
    would reject, so dropping it here keeps the export self-consistent rather
    than shipping a row that fails validation later.
    """
    rows: list[dict[str, str]] = []
    for source in sources:
        arxiv_id = source["arxiv_id"]
        published = source.get("published_date", "")
        precision = source.get("date_precision", "")
        if not published or not precision:
            # Predictions saved before the tool rendered a Published line still
            # carry the month their arXiv ID encodes.
            fallback = date_from_arxiv_id(arxiv_id)
            if fallback is None:
                logger.warning("Dropping source %s: it has no resolvable date", arxiv_id)
                continue
            published, precision = fallback.isoformat(), "month"
        rows.append(
            {
                "id": arxiv_id,
                "title": source.get("title", ""),
                "snippet": source.get("abstract", ""),
                "published_date": published,
                "date_precision": precision,
                # The rendered tool output carries no Semantic Scholar paper ID,
                # and the arXiv ID is already the unique key for a source here.
                "paper_id": arxiv_id,
            }
        )
    return rows


def write_query_folder(folder: Path, answer: str, rows: Sequence[Mapping[str, str]]) -> None:
    """Write one query's three scorer inputs into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "intro.md").write_text(answer, encoding="utf-8")
    (folder / "final_report.md").write_text(answer, encoding="utf-8")
    with (folder / "paper.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PAPER_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def export_predictions(predictions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Export every exportable prediction, returning a summary of what happened."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    skipped: list[dict[str, str]] = []
    for prediction in read_predictions(predictions_path):
        identifier = query_id(prediction)
        if not identifier:
            skipped.append({"idx": "", "reason": "no query ID"})
            continue

        answer = answer_text(prediction)
        if not answer.strip():
            skipped.append({"idx": identifier, "reason": "no answer text"})
            continue

        raw_trajectory = prediction.get("trajectory")
        trajectory = (
            AgentTrajectory.from_dict(raw_trajectory) if isinstance(raw_trajectory, dict) else None
        )
        rows = build_paper_rows(sources_from_trajectory(trajectory))
        if not rows:
            # An empty paper.csv fails the upstream contract, so an unciteable
            # run is recorded as skipped rather than written out as a folder.
            skipped.append({"idx": identifier, "reason": "no arXiv sources in trajectory"})
            continue

        write_query_folder(root / identifier, answer, rows)
        exported.append(identifier)

    return {
        "output_dir": str(root),
        "exported_ids": exported,
        "exported_count": len(exported),
        "skipped": skipped,
        "skipped_count": len(skipped),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a deepscholar_bench *-predictions.jsonl file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write the per-query scorer folders into",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = export_predictions(args.predictions, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["exported_count"] else 1


if __name__ == "__main__":
    sys.exit(main())

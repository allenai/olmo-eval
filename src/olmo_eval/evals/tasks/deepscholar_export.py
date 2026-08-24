"""Convert saved deepscholar_bench predictions into DeepScholar-Bench folders.

The upstream scorer reads one directory per query holding ``intro.md``,
``final_report.md`` and ``paper.csv``; this writes exactly that shape from a
run's ``*-predictions.jsonl``, so generation and scoring stay separate passes
and a scoring change never requires re-running the model.

``intro.md`` is NOT the raw answer. The benchmark prompt mandates numbered
inline citations, while ``DeepScholarBaseParser`` credits only markdown links to
arxiv.org/abs and returns no documents for a query without one, so the answer's
citations are rewritten by
:mod:`olmo_eval.evals.tasks.deepscholar_citations`, which mirrors lit-agents'
``render_intro``. ``final_report.md`` is the same rewritten bytes, matching what
lit-agents' ``_write_query_folder`` writes.

``paper.csv`` carries only the sources the rewritten intro actually cites, which
is what lit-agents exports and what keeps the contract's "cited IDs are a subset
of paper.csv" check meaningful. Its ``snippet`` column is scored, so abstracts
are re-fetched in full from Semantic Scholar rather than reused from the
truncated preview the agent was shown; ``snippet_source`` records which row fell
back to that preview.

``export_manifest.json``, ``summary.json`` and ``generation_manifest.json``
mirror lit-agents' own, so the output can be handed to its ``preflight_folder``
rather than only to the parser.

Usage:
    python -m olmo_eval.evals.tasks.deepscholar_export \\
        --predictions <run>/deepscholar_bench-predictions.jsonl \\
        --output <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx

from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.tasks.deepscholar_bench import (
    DEEPSCHOLAR_COMMIT,
    _first_nonempty,
    _parse_iso_date,
    build_deepscholar_prompt,
    download_deepscholar_dataset,
    sources_from_trajectory,
)
from olmo_eval.evals.tasks.deepscholar_citations import rewrite_intro
from olmo_eval.harness.tools.search import date_from_arxiv_id, normalize_arxiv_id

logger = logging.getLogger(__name__)

# The six columns lit-agents' _write_query_folder writes, plus one recording
# where the snippet came from. The upstream parser reads id/title/snippet by
# name and ignores the rest, so the extra column is inert to scoring.
PAPER_CSV_FIELDS = (
    "id",
    "title",
    "snippet",
    "published_date",
    "date_precision",
    "paper_id",
    "snippet_source",
)
EXPORTED_FILES = ("intro.md", "final_report.md", "paper.csv")
EXPORT_SCHEMA_VERSION = 1
QUERY_MANIFEST_NAME = "export_manifest.json"
GENERATION_MANIFEST_NAME = "generation_manifest.json"
# lit-agents' preflight only accepts systems it knows. This run's shape -- one
# agent, searching and writing its own numbered reference list -- is
# single_agent_1's, so that is the name the manifests carry by default.
DEFAULT_SYSTEM = "single_agent_1"
# Dropping a row can drop a citation, which can drop another row. Three
# passes is far past what any real answer needs; the fourth means a bug.
_MAX_EXPORT_PASSES = 4

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_FIELDS = "title,abstract,publicationDate"
S2_BATCH_MAX_IDS = 500


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Read every prediction row, refusing the file if any line is unreadable.

    A malformed line used to be skipped with a warning, which silently shrank
    both the export and the manifests built from it: a truncated run produced a
    folder that looked complete and scored as though it were. Naming the lines
    and stopping is the only honest option, because nothing here can tell a
    half-written line from a query that legitimately produced nothing.
    """
    rows: list[dict[str, Any]] = []
    malformed: list[int] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                malformed.append(line_number)
    if malformed:
        raise ValueError(
            f"{path} has unreadable prediction rows on line(s) "
            f"{', '.join(str(number) for number in malformed)}; refusing to export a "
            "partial run. Re-run the task or remove the truncated rows deliberately."
        )
    return rows


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


def query_id(prediction: Mapping[str, Any]) -> int:
    """The upstream positional query ID this prediction belongs to.

    Upstream identifies a query by its row position in the dataset CSV and the
    export folders are named for it, so a wrong ID silently scores one paper's
    answer against another's ground truth. ``doc_id`` is a within-run counter
    that coincides with the position only when nothing was filtered or limited,
    so it is not an acceptable substitute and this raises instead.
    """
    native_id = prediction.get("native_id")
    if isinstance(native_id, str) and native_id.isdigit():
        return int(native_id)
    raise ValueError(
        f"Prediction has no positional native_id (got {native_id!r}); it cannot be "
        "matched to a DeepScholar query. Re-run the task so predictions carry "
        "Instance.metadata['id']."
    )


def fetch_full_abstracts(
    arxiv_ids: Sequence[str],
    *,
    client: httpx.Client | None = None,
) -> dict[str, dict[str, str]]:
    """Re-fetch title and full abstract for each arXiv ID from Semantic Scholar.

    paper.csv's snippet column is scored, and what the agent saw is truncated to
    keep its context affordable, so the export asks for the text again rather
    than shipping a preview as if it were the abstract.

    Failure is survivable by design: an unreachable API, a missing paper or a
    paper with no abstract simply leaves that ID out of the result, and the
    caller falls back to the rendered preview and records that it did.
    """
    if not arxiv_ids:
        return {}

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("S2_API_KEY", "").split(",")[0].strip()
    if api_key:
        headers["x-api-key"] = api_key

    owned = client is None
    http = client or httpx.Client(timeout=60.0)
    fetched: dict[str, dict[str, str]] = {}
    try:
        for start in range(0, len(arxiv_ids), S2_BATCH_MAX_IDS):
            chunk = list(arxiv_ids[start : start + S2_BATCH_MAX_IDS])
            try:
                response = http.post(
                    S2_BATCH_URL,
                    params={"fields": S2_BATCH_FIELDS},
                    json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in chunk]},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                logger.warning(
                    "Semantic Scholar batch lookup failed for %d IDs: %s", len(chunk), error
                )
                continue
            if not isinstance(payload, list):
                logger.warning(
                    "Semantic Scholar batch returned %s, not a list", type(payload).__name__
                )
                continue
            for arxiv_id, raw in zip(chunk, payload, strict=False):
                if not isinstance(raw, Mapping):
                    continue
                record = cast("Mapping[str, Any]", raw)
                abstract = str(record.get("abstract") or "").strip()
                if not abstract:
                    continue
                fetched[arxiv_id] = {
                    "title": str(record.get("title") or "").strip(),
                    "abstract": abstract,
                    "published_date": str(record.get("publicationDate") or "").strip(),
                }
    finally:
        if owned:
            http.close()
    return fetched


def _row_precedes_cutoff(published: str, precision: str, cutoff: date) -> bool:
    """The contract's own date test, mirroring lit-agents' _validate_source_rows.

    Month precision compares months, day precision compares days, and any other
    precision is invalid rather than assumed. Strictly before, in both cases.
    """
    parsed = _parse_iso_date(published)
    if parsed is None:
        return False
    if precision == "month":
        return (parsed.year, parsed.month) < (cutoff.year, cutoff.month)
    if precision == "day":
        return parsed < cutoff
    return False


def build_paper_rows(
    sources: Sequence[Mapping[str, str]],
    fetched: Mapping[str, Mapping[str, str]] | None = None,
    cutoff: date | None = None,
) -> list[dict[str, str]]:
    """Build paper.csv rows for the cited sources, dropping any that cannot stand.

    A row needs a non-empty title and snippet and a resolvable date, because
    those are exactly the conditions the contract checks; shipping a row that
    fails them turns a scoreable query into a rejected one.

    ``cutoff`` is the strict half of the two-layer design: retrieval admits a
    paper on the month its arXiv ID encodes, and the batch re-fetch can then
    return a real publication date that turns out to fall on or after the query's
    cutoff. Such a paper was never citable, and only this test catches it.
    """
    fetched = fetched or {}
    rows: list[dict[str, str]] = []
    for source in sources:
        arxiv_id = normalize_arxiv_id(str(source.get("arxiv_id") or ""))
        if not arxiv_id:
            continue
        record = fetched.get(arxiv_id) or {}

        if record.get("abstract"):
            title = record.get("title") or str(source.get("title") or "")
            snippet = record["abstract"]
            snippet_source = "s2_batch"
        else:
            title = str(source.get("title") or "")
            snippet = str(source.get("abstract") or "")
            snippet_source = "tool_output"

        published, precision = _resolve_row_date(source, record, arxiv_id)
        if not published:
            logger.warning("Dropping source %s: it has no resolvable date", arxiv_id)
            continue
        if not title.strip() or not snippet.strip():
            logger.warning("Dropping source %s: it has no title or no abstract", arxiv_id)
            continue
        if cutoff is not None and not _row_precedes_cutoff(published, precision, cutoff):
            logger.warning(
                "Dropping source %s: %s (%s precision) does not precede the cutoff %s",
                arxiv_id,
                published,
                precision,
                cutoff.isoformat(),
            )
            continue

        rows.append(
            {
                "id": arxiv_id,
                "title": title.strip(),
                "snippet": snippet.strip(),
                "published_date": published,
                "date_precision": precision,
                # The rendered tool output carries no Semantic Scholar paper ID,
                # and the arXiv ID is already the unique key for a source here.
                "paper_id": arxiv_id,
                "snippet_source": snippet_source,
            }
        )
    return rows


def resolve_export(
    answer: str,
    sources: Sequence[Mapping[str, str]],
    fetched: Mapping[str, Mapping[str, str]],
    cutoff: date | None,
) -> tuple[str, list[Mapping[str, str]], list[dict[str, str]]]:
    """Rewrite the answer and build its rows until the two agree.

    A cited source can still fail at export -- no title, no abstract, a date the
    re-fetch moved on or after the cutoff -- and dropping its row while leaving
    its link standing in intro.md leaves the folder contradicting itself: the
    contract checks that every cited ID has a row, and the parser would render a
    citation whose title and snippet are empty. So the rewrite is re-run with the
    failed source withheld, which is what render_intro does with a source it
    rejects, and the citation comes out of the prose along with the row.

    Returns empty results when nothing survives, which is a query to record as
    unscoreable rather than an error: an answer citing only papers that fail the
    contract has genuinely produced nothing this benchmark can score.
    """
    allowed = list(sources)
    for _ in range(_MAX_EXPORT_PASSES):
        intro, cited = rewrite_intro(answer, allowed)
        if not cited:
            return "", [], []
        rows = build_paper_rows(cited, fetched, cutoff)
        kept = {row["id"] for row in rows}
        cited_ids = {normalize_arxiv_id(str(source.get("arxiv_id") or "")) for source in cited}
        if kept == cited_ids:
            return intro, cited, rows
        allowed = [
            source
            for source in allowed
            if normalize_arxiv_id(str(source.get("arxiv_id") or "")) in kept
        ]
        if not allowed:
            return "", [], []
    logger.warning(
        "Export did not settle after %d passes; treating the query as unscoreable",
        _MAX_EXPORT_PASSES,
    )
    return "", [], []


def _resolve_row_date(
    source: Mapping[str, str],
    record: Mapping[str, str],
    arxiv_id: str,
) -> tuple[str, str]:
    """Best date available for a row: the batch lookup, the tool's line, the ID."""
    batch_date = _parse_iso_date(str(record.get("published_date") or ""))
    if batch_date is not None:
        return batch_date.isoformat(), "day"
    published = str(source.get("published_date") or "")
    precision = str(source.get("date_precision") or "")
    if published and precision:
        return published, precision
    fallback = date_from_arxiv_id(arxiv_id)
    return ("", "") if fallback is None else (fallback.isoformat(), "month")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def query_fingerprint(record: Mapping[str, Any]) -> str:
    """Fingerprint one query exactly as lit-agents' _query_fingerprint does."""
    payload = {
        "idx": record["idx"],
        "query": record["query"],
        "cutoff_date": record["cutoff_date"],
        "arxiv_id": record["arxiv_id"],
        "title": record["title"],
        "abstract": record["abstract"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_query_records(dataset_path: str | Path | None = None) -> dict[int, dict[str, Any]]:
    """Rebuild the query records the manifests are fingerprinted against.

    Mirrors lit-agents' ``load_queries``: positional IDs, the same cutoff and
    title fallbacks, and the query text this task actually sent, which is the
    verbatim template.
    """
    path = Path(dataset_path) if dataset_path is not None else download_deepscholar_dataset()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    records: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        cutoff = _parse_iso_date(_first_nonempty(row, "published_date", "cutoff_date"))
        abstract = _first_nonempty(row, "abstract")
        if cutoff is None or not abstract:
            continue
        records[index] = {
            "idx": index,
            "query": build_deepscholar_prompt(cutoff.isoformat(), abstract),
            "cutoff_date": cutoff.isoformat(),
            "arxiv_id": normalize_arxiv_id(_first_nonempty(row, "arxiv_id")),
            "title": _first_nonempty(row, "title") or f"DeepScholar query {index}",
            "abstract": abstract,
        }
    return records


def write_query_folder(
    root: Path,
    query_id_value: int,
    intro: str,
    rows: Sequence[Mapping[str, str]],
    *,
    system: str,
    fingerprint: str,
) -> None:
    """Write one query's scorer inputs and its export manifest, atomically.

    Mirrors lit-agents' ``_write_query_folder``: intro.md and final_report.md
    hold the same bytes, and the manifest checksums all three files so a
    truncated write cannot pass for a complete export.
    """
    target = root / str(query_id_value)
    if target.exists():
        raise FileExistsError(f"Query output already exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".query-{query_id_value}-", dir=root) as tmp:
        temporary = Path(tmp)
        (temporary / "intro.md").write_text(intro, encoding="utf-8")
        (temporary / "final_report.md").write_text(intro, encoding="utf-8")
        with (temporary / "paper.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(PAPER_CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        _write_json_atomic(
            temporary / QUERY_MANIFEST_NAME,
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "system": system,
                "idx": query_id_value,
                "query_fingerprint": fingerprint,
                "num_papers": len(rows),
                "files": {name: _sha256_file(temporary / name) for name in EXPORTED_FILES},
            },
        )
        os.replace(temporary, target)


def _prepare_output_root(root: Path, *, force: bool) -> None:
    """Refuse to write into an export that is already there.

    Numeric query folders from an earlier run survive a new run that happens not
    to export the same queries, and preflight compares the folders on disk
    against summary.json -- so a stale folder fails the whole export with an
    error naming the run that is not at fault.
    """
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    existing = sorted(child.name for child in root.iterdir() if not child.name.startswith("."))
    if not existing:
        return
    if not force:
        shown = ", ".join(existing[:5]) + (", ..." if len(existing) > 5 else "")
        raise FileExistsError(
            f"Output directory {root} already holds {shown}. Exporting into it would "
            "mix two runs; pass --force to replace its contents, or choose a new directory."
        )
    shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def export_predictions(
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    system: str = DEFAULT_SYSTEM,
    dataset_path: str | Path | None = None,
    fetch_abstracts: bool = True,
    client: httpx.Client | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Export every prediction, returning the summary it also writes to disk."""
    root = Path(output_dir)
    _prepare_output_root(root, force=force)
    records = load_query_records(dataset_path)

    prepared = []
    for prediction in read_predictions(predictions_path):
        identifier = query_id(prediction)
        if identifier not in records:
            raise ValueError(
                f"Prediction {identifier} is not a query in the pinned dataset at "
                f"{DEEPSCHOLAR_COMMIT}"
            )
        answer = answer_text(prediction)
        raw_trajectory = prediction.get("trajectory")
        trajectory = (
            AgentTrajectory.from_dict(raw_trajectory) if isinstance(raw_trajectory, dict) else None
        )
        sources = sources_from_trajectory(trajectory)
        # A first pass only to learn which abstracts are worth re-fetching; the
        # binding pass runs below, once the fetched dates can be validated.
        _, provisional = rewrite_intro(answer, sources)
        prepared.append((identifier, answer, sources, provisional))

    fetched: dict[str, dict[str, str]] = {}
    if fetch_abstracts:
        wanted = list(
            dict.fromkeys(
                normalize_arxiv_id(str(source.get("arxiv_id") or ""))
                for _, _, _, provisional in prepared
                for source in provisional
            )
        )
        fetched = fetch_full_abstracts([w for w in wanted if w], client=client)

    summary: list[dict[str, Any]] = []
    exported: list[int] = []
    for identifier, answer, sources, _ in prepared:
        record = records[identifier]
        if not answer.strip():
            summary.append(
                {
                    "idx": identifier,
                    "arxiv_id": record["arxiv_id"],
                    "status": "no_answer",
                    "reason": "the run produced no answer text",
                    "termination_reason": "empty_response",
                    "papers_retrieved": len(sources),
                }
            )
            continue

        cutoff = _parse_iso_date(record["cutoff_date"])
        intro, cited, rows = resolve_export(answer, sources, fetched, cutoff)
        if not rows:
            summary.append(
                {
                    "idx": identifier,
                    "arxiv_id": record["arxiv_id"],
                    "status": "no_eligible_source",
                    "reason": (
                        "no citation in the answer resolved to a retrieved arXiv source that "
                        "satisfies the export contract"
                    ),
                    "sources_considered": len(sources),
                    "source_rejections": {"unexportable_after_validation": len(sources)},
                }
            )
            continue

        write_query_folder(
            root,
            identifier,
            intro,
            rows,
            system=system,
            fingerprint=query_fingerprint(record),
        )
        exported.append(identifier)
        summary.append(
            {
                "idx": identifier,
                "arxiv_id": record["arxiv_id"],
                "status": "success",
                "num_papers": len(rows),
            }
        )

    summary.sort(key=lambda item: item["idx"])
    _write_json_atomic(root / "summary.json", summary)
    _write_json_atomic(
        root / GENERATION_MANIFEST_NAME,
        {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "system": system,
            "queries": [
                {"idx": item["idx"], "fingerprint": query_fingerprint(records[item["idx"]])}
                for item in summary
            ],
        },
    )

    fallback_rows = sum(
        1
        for identifier in exported
        for row in _read_rows(root / str(identifier) / "paper.csv")
        if row.get("snippet_source") == "tool_output"
    )
    return {
        "output_dir": str(root),
        "system": system,
        "exported_ids": exported,
        "exported_count": len(exported),
        "skipped": [item for item in summary if item["status"] != "success"],
        "skipped_count": sum(1 for item in summary if item["status"] != "success"),
        "snippet_fallback_rows": fallback_rows,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help=(
            "System name recorded in the manifests. lit-agents' preflight only "
            f"accepts names it knows; default {DEFAULT_SYSTEM}."
        ),
    )
    parser.add_argument(
        "--no-fetch-abstracts",
        action="store_true",
        help="Skip the Semantic Scholar lookup and use the truncated tool snippets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the contents of a non-empty output directory",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = export_predictions(
        args.predictions,
        args.output,
        system=args.system,
        fetch_abstracts=not args.no_fetch_abstracts,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["exported_count"] else 1


if __name__ == "__main__":
    sys.exit(main())

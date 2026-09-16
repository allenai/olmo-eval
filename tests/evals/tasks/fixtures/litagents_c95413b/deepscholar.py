"""The DeepScholar-Bench export contract, as lit-agents implements it.

Verbatim excerpt of shared/deepscholar.py from ai2-multi-agent/integrated, which targets
github.com/guestrin-lab/deepscholar at commit
c95413b3b2f3255b461b90d0ce650f685ae2d1ff. Only the definitions the export tests
need are copied; nothing has been reworded.

It is here because the export is written to satisfy these functions, and a test
suite that only exercises our own reimplementation of them proves nothing about
whether the real ones would accept the output.

Do not edit. Re-extract from the reference if the benchmark is re-pinned.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from .retrieval_policy import (
    arxiv_id_from_url as _arxiv_id_from_url,
    normalize_arxiv_id as _normalize_arxiv_id,
)


EXPORT_SCHEMA_VERSION = 1


QUERY_MANIFEST_NAME = "export_manifest.json"


_EXPORTED_FILES = ("intro.md", "final_report.md", "paper.csv")


_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^)\s?#]+)",
    re.IGNORECASE,
)


class DeepScholarContractError(RuntimeError):
    """Raised when generated output cannot satisfy the benchmark contract."""


@dataclass(frozen=True)
class QueryRecord:
    """One DeepScholar query and its source-policy cutoff."""

    query_id: int
    query: str
    cutoff_date: date
    arxiv_id: str
    title: str
    abstract: str

    @property
    def groundtruth(self) -> dict[str, str]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "arxiv_link": (
                f"https://arxiv.org/abs/{self.arxiv_id}"
                if self.arxiv_id
                else ""
            ),
            "related_works_section": "",
            "arxiv_id": self.arxiv_id,
        }


def _query_fingerprint(record: QueryRecord) -> str:
    payload = {
        "idx": record.query_id,
        "query": record.query,
        "cutoff_date": record.cutoff_date.isoformat(),
        "arxiv_id": record.arxiv_id,
        "title": record.title,
        "abstract": record.abstract,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_query_export(
    folder: Path,
    record: QueryRecord,
    system_name: str,
) -> dict[str, Any]:
    query_id = record.query_id
    if not folder.is_dir():
        raise DeepScholarContractError(
            f"Query {query_id} output folder is missing: {folder}"
        )
    manifest_path = folder / QUERY_MANIFEST_NAME
    manifest = _read_json_mapping(
        manifest_path,
        f"query {query_id} export manifest",
    )
    required_keys = {
        "schema_version",
        "system",
        "idx",
        "query_fingerprint",
        "num_papers",
        "files",
    }
    if set(manifest) != required_keys:
        raise DeepScholarContractError(
            f"Query {query_id} {QUERY_MANIFEST_NAME} has unexpected fields"
        )
    expected_values = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "system": system_name,
        "idx": query_id,
        "query_fingerprint": _query_fingerprint(record),
    }
    for field, expected_value in expected_values.items():
        if manifest.get(field) != expected_value:
            raise DeepScholarContractError(
                f"Query {query_id} {QUERY_MANIFEST_NAME} has {field} "
                f"{manifest.get(field)!r}; expected {expected_value!r}"
            )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(_EXPORTED_FILES):
        raise DeepScholarContractError(
            f"Query {query_id} {QUERY_MANIFEST_NAME} must checksum "
            f"{', '.join(_EXPORTED_FILES)}"
        )
    for name in _EXPORTED_FILES:
        path = folder / name
        if not path.is_file():
            raise DeepScholarContractError(
                f"Query {query_id} is missing required file {name}"
            )
        actual_checksum = _sha256_file(path)
        if files.get(name) != actual_checksum:
            raise DeepScholarContractError(
                f"Query {query_id} {name} checksum does not match "
                f"{QUERY_MANIFEST_NAME}"
            )

    intro_text = (folder / "intro.md").read_text(encoding="utf-8")
    final_report = (folder / "final_report.md").read_text(encoding="utf-8")
    if intro_text != final_report:
        raise DeepScholarContractError(
            f"Query {query_id} intro.md and final_report.md differ"
        )
    rows = _read_paper_csv(folder / "paper.csv")
    _validate_source_rows(rows, record.cutoff_date, query_id)
    num_papers = manifest.get("num_papers")
    if type(num_papers) is not int or num_papers != len(rows):
        raise DeepScholarContractError(
            f"Query {query_id} {QUERY_MANIFEST_NAME} reports "
            f"{num_papers!r} papers but paper.csv has {len(rows)} rows"
        )

    cited_arxiv_ids = _arxiv_ids_from_report(intro_text)
    if not cited_arxiv_ids:
        raise DeepScholarContractError(
            f"Query {query_id} has no arXiv citation URLs"
        )
    citation_urls = re.findall(
        r"\]\((https?://[^)\s]+)\)",
        intro_text,
    )
    non_arxiv_urls = [
        url for url in citation_urls if _arxiv_id_from_url(url) is None
    ]
    if non_arxiv_urls:
        raise DeepScholarContractError(
            f"Query {query_id} contains non-arXiv citation URLs: "
            + ", ".join(non_arxiv_urls)
        )
    known_ids = {
        _normalize_arxiv_id(str(row.get("id") or ""))
        for row in rows
    }
    unknown = sorted(set(cited_arxiv_ids) - known_ids)
    if unknown:
        raise DeepScholarContractError(
            f"Query {query_id} cites arXiv IDs absent from paper.csv: "
            + ", ".join(unknown)
        )
    return dict(manifest)


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DeepScholarContractError(f"Missing {label}: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise DeepScholarContractError(
            f"Cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise DeepScholarContractError(f"{label} must contain a mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_rows(
    rows: Sequence[Mapping[str, str]],
    cutoff_date: date,
    query_id: int,
) -> None:
    if not rows:
        raise DeepScholarContractError(f"Query {query_id} paper.csv is empty")
    for row_number, row in enumerate(rows, start=2):
        arxiv_id = _normalize_arxiv_id(row.get("id", ""))
        if not arxiv_id:
            raise DeepScholarContractError(
                f"Query {query_id} paper.csv row {row_number} has no arXiv ID"
            )
        if not str(row.get("title") or "").strip():
            raise DeepScholarContractError(
                f"Query {query_id} paper.csv row {row_number} has an empty title"
            )
        if not str(row.get("snippet") or "").strip():
            raise DeepScholarContractError(
                f"Query {query_id} paper.csv row {row_number} has an empty snippet"
            )
        published = _parse_date(
            str(row.get("published_date") or ""),
            label=f"query {query_id} paper.csv row {row_number}",
        )
        precision = str(row.get("date_precision") or "day")
        if precision == "month":
            valid = (published.year, published.month) < (
                cutoff_date.year,
                cutoff_date.month,
            )
        elif precision == "day":
            valid = published < cutoff_date
        else:
            raise DeepScholarContractError(
                f"Query {query_id} paper.csv row {row_number} has invalid "
                f"date_precision {precision!r}"
            )
        if not valid:
            raise DeepScholarContractError(
                f"Query {query_id} source {arxiv_id} does not precede "
                f"the cutoff {cutoff_date.isoformat()}"
            )


def _read_paper_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _arxiv_ids_from_report(report: str) -> list[str]:
    return [
        _normalize_arxiv_id(match.group(1))
        for match in _ARXIV_URL_RE.finditer(report)
    ]


def _parse_date(value: str, *, label: str) -> date:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} is empty")
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(normalized[:10])
        except ValueError as error:
            raise ValueError(f"{label} is not an ISO date: {value!r}") from error


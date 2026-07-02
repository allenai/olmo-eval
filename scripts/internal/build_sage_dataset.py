"""Offline builder for the SAGE retrieval dataset mirror.

Downloads the SAGE JSON files, enriches gold papers with Semantic Scholar
metadata, writes local JSONL configs, and can push the two configs to the HF Hub.
This script is intentionally not imported or run by tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_sage_dataset")

POOLS_REPO = "yilunzhao/sage-retrieval"  # HF repo set (OT-1)
SAGE_REPO = "yilunzhao/sage-retrieval"  # HF repo set (OT-1)

SAGE_RAW_ROOT = "https://raw.githubusercontent.com/HughieHu/Sage/main"
DOMAINS = ("computer_science", "natural_science", "healthcare", "humanities")
QUERY_TYPES = ("short_form", "open_ended")
S2_FIELDS = "paperId,title,abstract,corpusId,externalIds"
S2_BATCH_SIZE = 500
S2_BATCH_MAX_ATTEMPTS = 5
S2_BATCH_URL = f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={S2_FIELDS}"

QUERY_TYPE_DIRS = {
    "short_form": "Sage_Short_Form_Questions",
    "open_ended": "Sage_Open_Ended_Questions",
}


def load_json_url(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "olmo-eval-sage-builder"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_domain_json(raw_root: str, query_type: str, domain: str) -> Any:
    directory = QUERY_TYPE_DIRS[query_type]
    url = f"{raw_root.rstrip('/')}/{directory}/{domain}.json"
    try:
        logger.info("Downloading %s", url)
        return load_json_url(url)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            f"Could not download SAGE {query_type}/{domain} from {url}: {exc}"
        ) from exc


class SemanticScholarClient:
    """Tiny Semantic Scholar Graph API client with an in-memory paper cache."""

    def __init__(self, api_key: str | None, sleep_seconds: float) -> None:
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.cache: dict[str, dict[str, Any]] = {}

    def prefetch(self, paper_ids: Iterable[str]) -> None:
        ids: list[str] = []
        seen: set[str] = set()
        for paper_id in paper_ids:
            paper_id = str(paper_id or "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            ids.append(paper_id)

        if not ids:
            logger.info("No Semantic Scholar paper IDs to prefetch")
            return

        chunks = [ids[index : index + S2_BATCH_SIZE] for index in range(0, len(ids), S2_BATCH_SIZE)]
        logger.info("Prefetching %d Semantic Scholar papers in %d batches", len(ids), len(chunks))
        for chunk_index, chunk in enumerate(chunks, start=1):
            logger.info(
                "Semantic Scholar batch %d/%d: fetching %d papers",
                chunk_index,
                len(chunks),
                len(chunk),
            )
            papers = self._fetch_batch(chunk)
            if len(papers) != len(chunk):
                raise RuntimeError(
                    "Semantic Scholar batch response length mismatch: "
                    f"requested {len(chunk)}, got {len(papers)}"
                )

            cached = 0
            missing = 0
            for requested_id, paper in zip(chunk, papers, strict=True):
                if paper:
                    self.cache[requested_id] = paper
                    cached += 1
                else:
                    self.cache[requested_id] = {}
                    missing += 1
            logger.info(
                "Semantic Scholar batch %d/%d cached %d papers (%d not found)",
                chunk_index,
                len(chunks),
                cached,
                missing,
            )

            if chunk_index < len(chunks) and self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

    def fetch_paper(self, paper_id: str) -> dict[str, Any]:
        if not paper_id:
            return {}
        if paper_id in self.cache:
            return self.cache[paper_id]

        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{quote(paper_id, safe='')}?fields={S2_FIELDS}"
        )
        headers = {"User-Agent": "olmo-eval-sage-builder"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:
                paper = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            logger.warning("Semantic Scholar lookup failed for %s: %s", paper_id, exc)
            paper = {}

        self.cache[paper_id] = paper
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return paper

    def _fetch_batch(self, paper_ids: list[str]) -> list[dict[str, Any] | None]:
        payload = json.dumps({"ids": paper_ids}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "olmo-eval-sage-builder",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(1, S2_BATCH_MAX_ATTEMPTS + 1):
            request = Request(S2_BATCH_URL, data=payload, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=60) as response:
                    papers = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429 or 500 <= exc.code < 600:
                    self._sleep_before_retry(exc, attempt)
                    continue
                raise RuntimeError(f"Semantic Scholar batch lookup failed: {exc}") from exc
            except URLError as exc:
                self._sleep_before_retry(exc, attempt)
                continue

            if not isinstance(papers, list):
                raise RuntimeError(
                    f"Semantic Scholar batch response must be a list, got {type(papers).__name__}"
                )
            return papers

        raise RuntimeError("Semantic Scholar batch lookup failed after retries")

    def _sleep_before_retry(self, exc: HTTPError | URLError, attempt: int) -> None:
        if attempt >= S2_BATCH_MAX_ATTEMPTS:
            raise RuntimeError("Semantic Scholar batch lookup failed after retries") from exc

        delay = retry_after_seconds(exc)
        if delay is None:
            delay = 2 ** (attempt - 1)
        logger.warning(
            "Semantic Scholar batch lookup failed on attempt %d/%d: %s; retrying in %.1fs",
            attempt,
            S2_BATCH_MAX_ATTEMPTS,
            exc,
            delay,
        )
        time.sleep(delay)


def records_from_payload(payload: Any, *, query_type: str, domain: str) -> list[dict[str, Any]]:
    if query_type == "open_ended":
        if not isinstance(payload, dict):
            raise TypeError(f"Expected open_ended/{domain} payload to be an object")
        raw_records = payload.get("questions", [])
    else:
        raw_records = payload

    if not isinstance(raw_records, list):
        raise TypeError(f"Expected {query_type}/{domain} records to be a list")

    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            continue
        identifier = str(
            raw_record.get("query_id")
            or raw_record.get("id")
            or raw_record.get("question_id")
            or f"{query_type}:{domain}:{index}"
        )
        ground_truth = raw_record.get("ground_truth")
        if not isinstance(ground_truth, dict):
            logger.warning("Skipping SAGE record %s with invalid ground_truth", identifier)
            continue
        record = dict(raw_record)
        record["domain"] = domain
        record["query_type"] = query_type
        record.setdefault("query_id", identifier)
        records.append(record)
    return records


def retry_after_seconds(exc: HTTPError | URLError) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def collect_gold_paper_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    paper_ids: list[str] = []
    seen: set[str] = set()

    def add_paper_id(paper: dict[str, Any]) -> None:
        paper_id = str(paper.get("paperId") or paper.get("paper_id") or "")
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            paper_ids.append(paper_id)

    for record in records:
        ground_truth = record.get("ground_truth")
        if not isinstance(ground_truth, dict):
            continue

        if record.get("query_type") == "open_ended":
            for key in ("most_relevant", "relevant"):
                papers = ground_truth.get(key) or []
                if not isinstance(papers, list):
                    continue
                for paper in papers:
                    if isinstance(paper, dict):
                        add_paper_id(paper)
        else:
            add_paper_id(ground_truth)

    return paper_ids


def enrich_gold_paper(
    paper: dict[str, Any],
    client: SemanticScholarClient,
) -> dict[str, Any]:
    paper_id = str(paper.get("paperId") or paper.get("paper_id") or "")
    fetched = client.fetch_paper(paper_id)
    external_ids = fetched.get("externalIds") or {}
    if not isinstance(external_ids, dict):
        external_ids = {}

    enriched = dict(paper)
    enriched["paperId"] = paper_id
    enriched["title"] = enriched.get("title") or fetched.get("title") or ""
    enriched["abstract"] = enriched.get("abstract") or fetched.get("abstract") or ""

    corpus_id = enriched.get("corpus_id") or enriched.get("corpusId") or fetched.get("corpusId")
    if corpus_id is not None:
        enriched["corpus_id"] = str(corpus_id)

    arxiv_id = enriched.get("arxiv_id") or enriched.get("arxivId") or external_ids.get("ArXiv")
    if arxiv_id:
        enriched["arxiv_id"] = str(arxiv_id)

    doi = enriched.get("doi") or enriched.get("DOI") or external_ids.get("DOI")
    if doi:
        enriched["doi"] = str(doi)

    return enriched


def enrich_record(record: dict[str, Any], client: SemanticScholarClient) -> dict[str, Any]:
    enriched = dict(record)
    ground_truth = enriched.get("ground_truth") or {}
    if not isinstance(ground_truth, dict):
        return enriched

    if enriched.get("query_type") == "open_ended":
        enriched_ground_truth: dict[str, Any] = dict(ground_truth)
        for key in ("most_relevant", "relevant"):
            papers = ground_truth.get(key) or []
            if not isinstance(papers, list):
                papers = []
            enriched_ground_truth[key] = [
                enrich_gold_paper(paper, client) for paper in papers if isinstance(paper, dict)
            ]
        enriched["ground_truth"] = enriched_ground_truth
    else:
        enriched["ground_truth"] = enrich_gold_paper(ground_truth, client)

    return enriched


def build_configs(
    *,
    raw_root: str,
    s2_api_key: str | None,
    s2_sleep: float,
) -> dict[str, list[dict[str, Any]]]:
    client = SemanticScholarClient(api_key=s2_api_key, sleep_seconds=s2_sleep)
    raw_configs: dict[str, list[dict[str, Any]]] = {query_type: [] for query_type in QUERY_TYPES}

    for query_type in QUERY_TYPES:
        for domain in DOMAINS:
            payload = download_domain_json(raw_root, query_type, domain)
            records = records_from_payload(payload, query_type=query_type, domain=domain)
            raw_configs[query_type].extend(records)
            logger.info("Loaded %d %s/%s records", len(records), query_type, domain)

    all_records = [record for records in raw_configs.values() for record in records]
    paper_ids = collect_gold_paper_ids(all_records)
    logger.info("Collected %d unique Semantic Scholar gold paper IDs", len(paper_ids))
    client.prefetch(paper_ids)

    configs: dict[str, list[dict[str, Any]]] = {query_type: [] for query_type in QUERY_TYPES}
    for query_type, records in raw_configs.items():
        configs[query_type].extend(enrich_record(record, client) for record in records)
        logger.info("Built %d %s records", len(records), query_type)

    return configs


def write_local(configs: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for config_name, rows in configs.items():
        path = output_dir / f"{config_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), path)


def push_to_hub(
    configs: dict[str, list[dict[str, Any]]],
    *,
    repo_id: str,
    private: bool,
) -> None:
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Pushing SAGE to the Hub requires `datasets` and `huggingface_hub`."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for config_name, rows in configs.items():
        dataset = DatasetDict({"train": Dataset.from_list(rows)})
        dataset.push_to_hub(repo_id, config_name=config_name)
        logger.info("Pushed %s config with %d rows to %s", config_name, len(rows), repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=SAGE_RAW_ROOT)
    parser.add_argument("--repo-id", default=SAGE_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("sage_dataset"))
    parser.add_argument("--dry-run", action="store_true", help="Build and write local files only.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--private", action="store_true", help="Create the HF dataset repo as private."
    )
    parser.add_argument("--s2-api-key", default=os.getenv("S2_API_KEY"))
    parser.add_argument(
        "--s2-sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between Semantic Scholar batch requests.",
    )
    args = parser.parse_args()

    configs = build_configs(
        raw_root=args.raw_root,
        s2_api_key=args.s2_api_key,
        s2_sleep=args.s2_sleep,
    )
    write_local(configs, args.output_dir)

    for config_name, rows in configs.items():
        logger.info("Config %s has %d rows", config_name, len(rows))

    if args.push_to_hub and not args.dry_run:
        push_to_hub(configs, repo_id=args.repo_id, private=args.private)
    else:
        logger.info("Skipping Hub push. Use --push-to-hub without --dry-run to upload.")


if __name__ == "__main__":
    main()

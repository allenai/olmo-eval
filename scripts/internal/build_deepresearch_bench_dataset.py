"""Offline builder for the DeepResearch Bench dataset mirror.

Downloads the official query, criteria, and cleaned-reference JSONL files at a
pinned upstream commit (or reads them from ``--source-dir``), joins them by task
ID, writes local JSONL configs, and can push them to the HF Hub. This script is
intentionally not imported or run by tests.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_deepresearch_bench_dataset")

DEEPRESEARCH_BENCH_REPO = "allenai/deepresearch-bench"
UPSTREAM_COMMIT = "469cce54ea7f6a63c163d3d9fec879cf289ec484"
UPSTREAM_RAW_ROOT = (
    f"https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/{UPSTREAM_COMMIT}"
)
EXPECTED_IDS = set(range(1, 101))

SOURCE_FILES = {
    "queries": Path("data/prompt_data/query.jsonl"),
    "criteria": Path("data/criteria_data/criteria.jsonl"),
    "references": Path("data/test_data/cleaned_data/reference.jsonl"),
}


def load_jsonl_lines(lines: Iterable[str], *, source: str) -> list[dict[str, Any]]:
    """Parse JSONL records and report the source and line on malformed input."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source} at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"Expected an object in {source} at line {line_number}")
        rows.append(row)
    return rows


def load_jsonl_path(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return load_jsonl_lines(handle, source=str(path))


def download_jsonl(relative_path: Path) -> list[dict[str, Any]]:
    """Download one official file from the immutable upstream revision."""
    url = f"{UPSTREAM_RAW_ROOT}/{relative_path.as_posix()}"
    logger.info("Downloading %s", url)
    request = Request(url, headers={"User-Agent": "olmo-eval-deepresearch-builder"})
    try:
        with urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not download DeepResearch Bench data from {url}: {exc}") from exc
    return load_jsonl_lines(text.splitlines(), source=url)


def load_official_files(
    source_dir: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    """Load all three official inputs from disk or the pinned raw URLs."""
    if source_dir is None:
        return {name: download_jsonl(path) for name, path in SOURCE_FILES.items()}

    missing = [
        source_dir / path for path in SOURCE_FILES.values() if not (source_dir / path).is_file()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "DeepResearch Bench source files are missing:\n"
            f"{missing_text}\nOmit --source-dir to fetch the pinned official files instead."
        )
    return {
        name: load_jsonl_path(source_dir / relative_path)
        for name, relative_path in SOURCE_FILES.items()
    }


def index_by_id(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[int, dict[str, Any]]:
    """Validate the official integer IDs and reject duplicates or omissions."""
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        identifier = row.get("id")
        if not isinstance(identifier, int):
            raise TypeError(f"{source_name} row {row_number} has non-integer id {identifier!r}")
        if identifier in indexed:
            raise ValueError(f"Duplicate id {identifier} in {source_name}")
        indexed[identifier] = row

    actual_ids = set(indexed)
    if actual_ids != EXPECTED_IDS:
        missing = sorted(EXPECTED_IDS - actual_ids)
        unexpected = sorted(actual_ids - EXPECTED_IDS)
        raise ValueError(
            f"{source_name} must contain exactly IDs 1..100; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return indexed


def _require_string(row: dict[str, Any], key: str, *, source: str, identifier: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} id {identifier} has invalid {key!r}")
    return value


def build_rows(inputs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Join official inputs by ID while preserving all benchmark text verbatim."""
    queries = index_by_id(inputs["queries"], source_name="query.jsonl")
    criteria = index_by_id(inputs["criteria"], source_name="criteria.jsonl")
    references = index_by_id(inputs["references"], source_name="reference.jsonl")

    output: list[dict[str, Any]] = []
    for identifier in sorted(EXPECTED_IDS):
        query = queries[identifier]
        criterion_row = criteria[identifier]
        reference = references[identifier]

        prompt = _require_string(query, "prompt", source="query.jsonl", identifier=identifier)
        criterion_prompt = _require_string(
            criterion_row, "prompt", source="criteria.jsonl", identifier=identifier
        )
        reference_prompt = _require_string(
            reference, "prompt", source="reference.jsonl", identifier=identifier
        )
        if criterion_prompt != prompt:
            logger.warning("Prompt mismatch between query and criteria for id %d", identifier)
        if reference_prompt != prompt:
            logger.warning("Prompt mismatch between query and reference for id %d", identifier)

        language = query.get("language")
        if language not in {"en", "zh"}:
            raise ValueError(f"query.jsonl id {identifier} has invalid language {language!r}")
        topic = _require_string(query, "topic", source="query.jsonl", identifier=identifier)

        dimension_weight = criterion_row.get("dimension_weight")
        criterions = criterion_row.get("criterions")
        if not isinstance(dimension_weight, dict) or not isinstance(criterions, dict):
            raise TypeError(
                f"criteria.jsonl id {identifier} must contain mapping-valued "
                "dimension_weight and criterions"
            )

        article = _require_string(
            reference, "article", source="reference.jsonl", identifier=identifier
        )
        output.append(
            {
                "id": identifier,
                "language": language,
                "topic": topic,
                "prompt": prompt,
                "criteria": {
                    "dimension_weight": dimension_weight,
                    "criterions": criterions,
                },
                "reference_article": article,
            }
        )

    language_counts = {
        language: sum(row["language"] == language for row in output) for language in ("en", "zh")
    }
    if language_counts != {"en": 50, "zh": 50}:
        raise ValueError(f"Expected 50 English and 50 Chinese rows, got {language_counts}")
    return output


def dataset_configs(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the full dataset and its two language-specific subsets."""
    return {
        "default": rows,
        "en": [row for row in rows if row["language"] == "en"],
        "zh": [row for row in rows if row["language"] == "zh"],
    }


def write_local(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for config_name, config_rows in dataset_configs(rows).items():
        path = output_dir / f"{config_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in config_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d rows to %s", len(config_rows), path)
        paths.append(path)
    return paths


def push_to_hub(rows: list[dict[str, Any]], *, repo_id: str) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("Pushing DeepResearch Bench requires `datasets`.") from exc

    for config_name, config_rows in dataset_configs(rows).items():
        dataset = DatasetDict({"test": Dataset.from_list(config_rows)})
        dataset.push_to_hub(repo_id, config_name=config_name)
        logger.info("Pushed %s/test with %d rows to %s", config_name, len(config_rows), repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Read an official repository checkout instead of downloading files "
            f"at pinned commit {UPSTREAM_COMMIT}."
        ),
    )
    parser.add_argument("--repo-id", default=DEEPRESEARCH_BENCH_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("deepresearch_bench_dataset"))
    parser.add_argument("--push", action="store_true", help="Push all configs to the HF Hub.")
    args = parser.parse_args()

    inputs = load_official_files(args.source_dir)
    rows = build_rows(inputs)
    write_local(rows, args.output_dir)
    if args.push:
        push_to_hub(rows, repo_id=args.repo_id)
    else:
        logger.info("Skipping Hub push. Use --push to upload.")


if __name__ == "__main__":
    main()

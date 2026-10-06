"""Reading and sampling JSONL files.

The long-context benchmarks ship their data as JSONL files that can run to
gigabytes, so these helpers can sample rows without parsing the whole file.
"""

import json
import random
from collections.abc import Callable
from typing import Any

import numpy as np


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Parse every non-blank line of a JSONL file, each of which must be an object."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSONL in {path}:{line_num}: {err}") from err
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_num}, got {type(record).__name__}"
                )
            records.append(record)
    return records


def sample_jsonl_rows(path: str, max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    """Load a seeded random sample of rows from a JSONL file without parsing the rest.

    Selection matches permuting the fully loaded file with the same seed and
    taking the first `max_samples` rows, so results are unchanged from a full
    load; only the lines that survive are parsed. With no cap the file loads
    directly.
    """
    if max_samples is None:
        return load_jsonl(path)

    with open(path, encoding="utf-8") as f:
        nonblank = [bool(line.strip()) for line in f]
    positions = [i for i, present in enumerate(nonblank) if present]
    permutation = np.random.default_rng(seed).permutation(len(positions))
    chosen = {positions[int(idx)]: rank for rank, idx in enumerate(permutation[:max_samples])}

    rows: list[dict[str, Any] | None] = [None] * len(chosen)
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num in chosen:
                rows[chosen[line_num]] = json.loads(line)
    return [row for row in rows if row is not None]


def sample_jsonl_by_key(
    path: str,
    max_samples: int | None,
    seed: int,
    key: Callable[[dict[str, Any]], Any],
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Load rows grouped by a key, sampling keys without holding the whole file parsed.

    Some files repeat each question once per gold-passage depth, and
    the largest tiers are gigabytes of JSONL that expand severalfold when
    parsed. So: one pass parses rows only long enough to record each line's
    key (and apply `keep`, an optional row filter), the kept keys are sampled,
    and a second pass parses only the lines that survive. Selection is
    identical to sampling after a full load -- same sorted unique key set,
    same RNG draw -- just without the resident memory. With no cap there is
    nothing to skip, so the file loads directly.
    """
    if max_samples is None:
        rows = load_jsonl(path)
        return [r for r in rows if keep(r)] if keep is not None else rows

    line_keys: list[Any] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                line_keys.append(None)
                continue
            row = json.loads(line)
            if keep is not None and not keep(row):
                line_keys.append(None)
                continue
            line_keys.append(key(row))

    unique = sorted({k for k in line_keys if k is not None})
    kept = set(random.Random(seed).sample(unique, min(max_samples, len(unique))))

    rows = []
    with open(path, encoding="utf-8") as f:
        for line, line_key in zip(f, line_keys, strict=True):
            if line_key in kept:
                rows.append(json.loads(line))
    return rows

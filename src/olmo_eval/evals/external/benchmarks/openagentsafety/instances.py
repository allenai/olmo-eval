"""Instance selection for OpenAgentSafety runs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

# TheAgentCompany services an instance can list in ``dependencies``.
TAC_SERVICES = frozenset({"gitlab", "owncloud", "plane", "rocketchat"})
# Matches the sampling seed in OpenHands/benchmarks ``prepare_dataset``.
SAMPLE_SEED = 42


def load_instances(dataset: str, split: str) -> list[dict[str, Any]]:
    """Load instance IDs and service dependencies from a dataset."""
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split).select_columns(["instance_id", "dependencies"])
    return [
        {"instance_id": str(row["instance_id"]), "dependencies": list(row["dependencies"] or [])}
        for row in rows
    ]


def read_select(select: str | list[str] | None) -> set[str] | None:
    """Return the instance IDs named by a ``select`` argument, if any."""
    if select is None:
        return None
    if isinstance(select, list):
        return set(select)
    path = Path(select).expanduser()
    if path.is_file():
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {select}


def select_instances(
    instances: list[dict[str, Any]],
    select: set[str] | None = None,
    n_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by ``select``, then sample ``n_limit`` instances like the upstream runner."""
    chosen = [row for row in instances if select is None or row["instance_id"] in select]
    if n_limit is not None and 0 < n_limit < len(chosen):
        import pandas as pd

        frame = pd.DataFrame(chosen)
        sampled = frame.sample(n=n_limit, random_state=SAMPLE_SEED)
        chosen = [chosen[i] for i in sampled.index]
    return chosen


def required_services(instances: Iterable[dict[str, Any]]) -> set[str]:
    """TheAgentCompany services that any of ``instances`` depends on."""
    services: set[str] = set()
    for row in instances:
        services.update(dep for dep in row["dependencies"] if dep in TAC_SERVICES)
    return services

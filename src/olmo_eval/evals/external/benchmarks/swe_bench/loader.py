"""Loader for SWE-bench instances from HuggingFace."""

from __future__ import annotations

import logging

from olmo_eval.data import DataLoader, DataSource

from .task import SWEBenchInstance

logger = logging.getLogger(__name__)

DATASET_PATH = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"


def select_instances(
    instances: list[SWEBenchInstance],
    instance_ids: list[str] | None = None,
    repos: list[str] | None = None,
    limit: int | None = None,
    difficulties: list[str] | None = None,
) -> list[SWEBenchInstance]:
    """Filter instances by ID, repository and difficulty, then keep the first ``limit``.

    Raises:
        ValueError: If any requested instance ID, or any requested difficulty, is not in
            the dataset.
    """
    if difficulties:
        # Checked against the whole dataset, so a typo is not mistaken for an empty bucket.
        known_difficulties = {inst.difficulty for inst in instances}
        missing = [d for d in difficulties if d not in known_difficulties]
        if missing:
            raise ValueError(
                f"Unknown SWE-bench difficulties: {', '.join(missing)}; "
                f"known: {', '.join(sorted(known_difficulties))}"
            )
    if instance_ids:
        known = {inst.instance_id for inst in instances}
        missing = [i for i in instance_ids if i not in known]
        if missing:
            raise ValueError(f"Unknown SWE-bench instance IDs: {', '.join(missing)}")
        wanted = set(instance_ids)
        instances = [inst for inst in instances if inst.instance_id in wanted]
    if repos:
        wanted_repos = set(repos)
        instances = [inst for inst in instances if inst.repo in wanted_repos]
    if difficulties:
        wanted_difficulties = set(difficulties)
        instances = [inst for inst in instances if inst.difficulty in wanted_difficulties]
    if limit is not None:
        instances = instances[:limit]
    return instances


def load_instances(
    dataset_path: str = DATASET_PATH,
    revision: str | None = DATASET_REVISION,
    split: str = "test",
) -> list[SWEBenchInstance]:
    """Load every instance of a SWE-bench dataset in dataset order."""
    source = DataSource(path=dataset_path, split=split, revision=revision)
    instances = [SWEBenchInstance.from_row(row) for row in DataLoader().load(source)]
    logger.info(f"Loaded {len(instances)} SWE-bench instances from {dataset_path}@{revision}")
    return instances

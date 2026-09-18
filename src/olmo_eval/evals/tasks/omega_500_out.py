"""Frozen, family-matched exploratory OOD companion to OMEGA-500."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks._omega_500_out_ids import IDS_BY_CONFIG
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.omega_500 import Omega500

SELECTED_IDS = frozenset(item for items in IDS_BY_CONFIG.values() for item in items)


@register("omega_500_out")
class Omega500Out(Omega500):
    """Evaluate 500 fixed test_out items with OMEGA-500's family proportions."""

    data_source = DataSource(
        path="allenai/omega-explorative",
        revision="b04ae8d4757a2229e8ed65ac6a923e502d0bbb95",
        data_files=tuple(f"{config}/test_out-00000-of-00001.parquet" for config in IDS_BY_CONFIG),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        instances = list(self._load_instances_cached())
        observed = [instance.metadata["id"] for instance in instances]
        if len(observed) != len(SELECTED_IDS) or set(observed) != SELECTED_IDS:
            raise ValueError("OMEGA test_out data does not match the frozen 500-item manifest")
        yield from instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if doc["id"] not in SELECTED_IDS:
            return None
        return super().process_doc(doc, index)

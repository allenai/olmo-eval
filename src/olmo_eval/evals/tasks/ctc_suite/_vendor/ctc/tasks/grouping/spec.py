"""
The ``grouping`` eval contract.

Partition a set of OpenAlex abstracts into the requested number of categories. Only the partition
is asked for and scored -- pairwise, on which documents were put together, so cluster numbering is
irrelevant (see :mod:`ctc.tasks._grouping`). :mod:`ctc.tasks.grouping_labeled` is the same task
with a name requested per group.

The ``ctc_grouping`` suite row grades with this spec. It was referenced by the ROSTER but never
vendored, so the row could not be scored at all; the prompt, target and metric here are the
pre-migration ``grouping`` ones (``evaluate._eval_grouping``: pairwise F1 over 1-based ids).

.. note::
   Legacy prompt path, like ``grouping_labeled``: documents, then the raw query string, with the
   instruction in the alpaca header; ``query_position`` is ignored.
"""

from __future__ import annotations

import json
from typing import Dict

from ...format.prompts import GROUPING_INSTRUCTION
from .._grouping import make_grouping_spec

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states the requested K.

    :returns: The raw query string.
    """
    return example["queries"][0]


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The JSON grouping object, ids 1-based, one entry per gold cluster.
    """
    groups = [{"doc_ids": [int(d) + 1 for d in c]} for c in example["gold_doc_indices"]]
    return json.dumps({"groups": groups})


SPEC = make_grouping_spec(
    name="grouping",
    description="Partition abstracts into K categories.",
    instruction=GROUPING_INSTRUCTION,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    sources=("arxiv", "openalex"),
)

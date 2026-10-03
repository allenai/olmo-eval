"""Diagnostic variants of ``pixelrag_simpleqa`` (not the benchmark protocol).

They locate why Molmo2-4B answers SimpleQA with ``<points>`` output: each changes one thing
from the official prompt.

* ``pixelrag_simpleqa_abl_instr`` appends NQ's answer-format instruction;
* ``pixelrag_simpleqa_abl_nosys`` drops the system prompt;
* ``pixelrag_simpleqa_abl_top1`` sends only the top-ranked tile.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.benchmarks.pixelrag import PixelRagSimpleQaTask, reader_messages

#: The answer-format instruction the NQ and NQ-Tables rows carry.
NQ_INSTRUCTION = "Answer with as few words as possible. Give only the answer, no explanation."


@register("pixelrag_simpleqa_abl_instr")
class PixelRagSimpleQaInstructionTask(PixelRagSimpleQaTask):
    def _build_instances(self) -> Iterator[Instance]:
        for instance in super()._build_instances():
            instance.metadata["instructions"] = NQ_INSTRUCTION
            yield instance


@register("pixelrag_simpleqa_abl_nosys")
class PixelRagSimpleQaNoSystemTask(PixelRagSimpleQaTask):
    def format_request(self, instance: Instance) -> LMRequest:
        meta = instance.metadata
        _, user = reader_messages(
            meta["problem"],
            meta["instructions"],
            len(meta["tile_ranks"]),
            query_image=meta["has_query_image"],
        )
        return LMRequest(request_type=RequestType.CHAT, messages=(user,), images=(meta["images"],))


@register("pixelrag_simpleqa_abl_top1")
class PixelRagSimpleQaTop1Task(PixelRagSimpleQaTask):
    def _build_instances(self) -> Iterator[Instance]:
        for instance in super()._build_instances():
            meta = instance.metadata
            ranks = meta["tile_ranks"][:1]
            loader = meta["images"]
            meta["tile_ranks"] = ranks
            meta["images"] = functools.partial(
                loader.func, loader.args[0], loader.args[1], ranks, loader.args[3]
            )
            yield instance

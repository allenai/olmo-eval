"""Shared base for the document-OCR benchmarks (olmOCR-bench, CC-OCR, OmniDocBench).

These benchmarks hand the model one page image and a transcription instruction, and grade
the markdown (or JSON) it writes back. :class:`OcrTask` adds the two things they share on
top of :class:`ImageQATask`:

* the instruction follows the checkpoint's prompt family, like the pointing and captioning
  tasks (:func:`ocr_question`);
* scoring runs off the event loop, since grading a page is CPU work that can take seconds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from olmo_eval.common.types import Response
from olmo_eval.evals.vision.scoring.prompts import apply_style_prefix
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

#: mm_olmo's formatter style for free-form instructions; the only instruction-following
#: style the ``style_and_length*`` (pretrain) family is trained on.
INSTRUCTION_STYLE = "text_sft"


def ocr_question(instruction: str, system_prompt_style: str) -> str:
    """``instruction`` as the checkpoint's prompt family expects to read it.

    Instruction-tuned checkpoints (``demo_or_style_v*``) take the benchmark's instruction
    verbatim. The ``style_and_length*`` family was trained with a ``"<style>:"`` prefix on
    every prompt and has no OCR style, so the instruction is sent under its free-form
    instruction style.
    """
    return apply_style_prefix(instruction, system_prompt_style, INSTRUCTION_STYLE)


class OcrTask(ImageQATask):
    """Base class for single-page OCR / document-parsing benchmarks."""

    #: Prompt family assumed when the run does not say; matches the instruction-tuned
    #: checkpoints, mirroring the pointing and captioning tasks.
    default_system_prompt_style = "demo_or_style_v2"

    def _question(self, instruction: str) -> str:
        style = self.config.system_prompt_style or self.default_system_prompt_style
        return ocr_question(instruction, style)

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: ScoringContext | None = None,
    ) -> Sequence[Response]:
        self._extract_answers(responses)
        await asyncio.to_thread(self._apply_scorers, responses)
        return responses

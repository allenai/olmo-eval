"""Base class for multimodal generation tasks.

Tasks that produce text from image + text inputs (captioning, VQA, MMMU, etc.)
inherit from :class:`MultimodalGenerationTask`. The interface is intentionally
thin so multiple consumers can plug in their own generation backends:

- OLMo-core: runs :class:`~olmo_core.eval.MultimodalGenerator` over the
  OLMo model and calls :meth:`score_all` with the decoded strings.
- Inference servers (vLLM, LiteLLM): call :meth:`format_request` to get an
  OpenAI-vision-style ``LMRequest``, run generation, then call
  :meth:`score_all`.

The only contract this module has with consumer code is:

1. :attr:`Instance.images` carries raw PIL images (or ``None`` for text-only).
2. :meth:`score_all` takes ``(predictions: list[str], instances: list[Instance])``
   and returns aggregated ``{metric_name: float}`` — no model knowledge needed.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Any

from olmo_eval.common.types import Instance, LMRequest, RequestType

from .base import Task


class MultimodalGenerationTask(Task):
    """Base class for image-conditioned text-generation tasks.

    Subclasses must implement:

    - :attr:`instances` — yield :class:`Instance` objects whose ``.images``
      tuple holds PIL images (one per image slot in the prompt).
    - :meth:`score` — compare one decoded prediction to its reference and
      return per-metric floats.

    Optionally override :meth:`score_all` for corpus-level metrics (e.g. CIDEr,
    which needs the full candidate/reference corpus at once).
    """

    # ------------------------------------------------------------------
    # Format request (OpenAI vision message format)
    # ------------------------------------------------------------------

    def format_request(self, instance: Instance) -> LMRequest:
        """Build a chat-style :class:`LMRequest` with inline image content blocks.

        The format follows the OpenAI vision API so inference-server consumers
        can forward it directly to vLLM / LiteLLM without any adaptation.
        Images stored on ``instance.images`` are passed as raw PIL objects;
        server-side providers encode them to base-64 or data-URLs as needed.
        """
        content: list[dict[str, Any]] = []
        for img in instance.images or ():
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": instance.question})
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": tuple(content)},),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abstractmethod
    def score(self, prediction: str, instance: Instance) -> dict[str, float]:
        """Score a single decoded prediction against its reference.

        :param prediction: The decoded model output (stripped of special tokens).
        :param instance: The originating instance (provides ``gold_answer`` and
            any ``metadata`` the scorer needs).
        :returns: Mapping of metric name → scalar value, e.g.
            ``{"cap_f1": 0.72, "recall": 0.78, "consistency": 0.66}``.
        """
        ...

    def score_all(
        self,
        predictions: list[str],
        instances: list[Instance],
    ) -> dict[str, float]:
        """Aggregate scores across the full prediction list.

        The default implementation calls :meth:`score` per instance and
        macro-averages each metric. Override this for corpus-level metrics
        (CIDEr, SPICE) that need the full candidate/reference set at once.

        :param predictions: One decoded string per instance, in the same order
            as *instances*.
        :param instances: The instances that produced *predictions*.
        :returns: Aggregated ``{metric_name: float}``.
        """
        accumulated: dict[str, list[float]] = defaultdict(list)
        for pred, inst in zip(predictions, instances):
            for name, val in self.score(pred, inst).items():
                accumulated[name].append(val)
        return {name: sum(vals) / len(vals) for name, vals in accumulated.items() if vals}

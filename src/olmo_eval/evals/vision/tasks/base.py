"""Shared mechanics for the vision benchmark tasks.

Every vision family caches its instances, applies ``config.limit``, and builds a
single-turn CHAT request with images attached; only how instances are built and
which images attach differ. Subclasses implement ``_build_instances`` and, when
one image is not the right attachment, override ``_attach_images``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.evals.tasks.common.base import Task
from olmo_eval.evals.vision.data.images import load_instance_image

__all__ = ["VisionTask"]


class VisionTask(Task):
    """Base class for the vision benchmark tasks."""

    #: Image decoding is needed to build instances, whichever provider runs them.
    dependencies = ["pillow"]

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            instances = list(self._build_instances())
            limit = self.config.limit
            if limit is not None:
                instances = instances[:limit]
            self._instances_cache = instances
        yield from self._instances_cache

    @abstractmethod
    def _build_instances(self) -> Iterator[Instance]:
        """Yield all instances for ``self.config.split`` (before ``limit``)."""
        ...

    def _attach_images(self, instance: Instance) -> tuple[Any, ...] | None:
        """The images to send with ``instance``; ``None`` sends a text-only request."""
        image = load_instance_image(instance)
        return (image,) if image is not None else None

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=self._attach_images(instance),
        )

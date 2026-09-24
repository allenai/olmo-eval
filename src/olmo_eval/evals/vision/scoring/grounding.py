"""Point parsing and scoring for GUI grounding benchmarks.

A grounding instance asks the model to locate the UI element an instruction
refers to. The model answers with a point, and the answer is correct when that
point falls inside the element's bounding box. Parsed points and boxes are both
in percent-of-image coordinates (0-100), so scoring never needs the image size.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput

#: Instance metadata key holding the target box as ``[x1, y1, x2, y2]`` in percent coordinates.
BBOX_KEY = "bbox"

# Molmo2 pointing markup, e.g. <points coords="1 1 208 108">label</points>: one
# "<frame> <idx> <x> <y> [<idx> <x> <y> ...]" group per frame, coordinates scaled
# by 1000. The regexes are the ones the Molmo2 model card ships for decoding it.
_MOLMO2_COORDS_RE = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"/?>')
_MOLMO2_FRAME_RE = re.compile(r"(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)")
_MOLMO2_POINT_RE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
# Molmo (v1) pointing markup: <point x="10.5" y="20.1" alt="...">label</point>, or
# <points x1="..." y1="..." x2="..." y2="..." alt="...">label</points>, coordinates
# in percent. The backreference pairs each x with the y carrying the same index.
_MOLMO_POINT_RE = re.compile(r'x(\d*)="\s*(\d+(?:\.\d+)?)\s*"\s+y\1="\s*(\d+(?:\.\d+)?)\s*"')
# A JSON object with at most one level of nesting, enough for a click action such as
# {"action": {"name": "click", "x": 10.5, "y": 20.1}}.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _point_from_json(obj: Any) -> tuple[float, float] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("action"), dict):
        obj = obj["action"]
    try:
        return float(obj["x"]), float(obj["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _molmo2_point(text: str) -> tuple[float, float] | None:
    """The first point of the first Molmo2 ``coords`` group, in percent."""
    for coords in _MOLMO2_COORDS_RE.finditer(text):
        for frame in _MOLMO2_FRAME_RE.finditer(coords.group(1)):
            for point in _MOLMO2_POINT_RE.finditer(frame.group(2)):
                x, y = float(point.group(2)) / 10, float(point.group(3)) / 10
                if 0 <= x <= 100 and 0 <= y <= 100:
                    return x, y
    return None


def parse_point(text: str) -> tuple[float, float] | None:
    """The first point a response names, as ``(x, y)`` in percent coordinates.

    Recognizes Molmo2 pointing markup (``<points coords="..">``, scaled by 1000),
    Molmo pointing markup (``<point x=".." y="..">`` and ``<points x1=".." y1=".."
    ..>``, in percent) and a JSON click action carrying ``x`` and ``y`` in percent,
    optionally nested under ``action``. Returns ``None`` when the response names
    no point, which includes Molmo's "There are none." answer.
    """
    point = _molmo2_point(text)
    if point is not None:
        return point
    match = _MOLMO_POINT_RE.search(text)
    if match:
        return float(match.group(2)), float(match.group(3))
    for candidate in _JSON_OBJECT_RE.finditer(text):
        try:
            obj = json.loads(candidate.group())
        except json.JSONDecodeError:
            continue
        point = _point_from_json(obj)
        if point is not None:
            return point
    return None


def point_in_box(point: Sequence[float], bbox: Sequence[float]) -> bool:
    """Whether ``point`` lies inside ``bbox`` (``[x1, y1, x2, y2]``), edges inclusive."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass(frozen=True, slots=True)
class PointInBoxScorer(Scorer):
    """1.0 when the extracted point lies inside the instance's target box, else 0.0.

    The task's ``extract_answer`` supplies the point; a response that names no
    point scores 0.0.
    """

    name: str = "point_in_box"

    def score(self, instance: Instance, output: LMOutput) -> float:
        point = output.extracted_answer
        bbox = instance.metadata.get(BBOX_KEY)
        if point is None or bbox is None:
            return 0.0
        return 1.0 if point_in_box(point, bbox) else 0.0

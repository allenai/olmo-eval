"""Monkey-patch textual-hires-canvas to use bitwise OR for braille combining.

When multiple series overlap in a braille plot, the default behavior overwrites
cells instead of combining them. This patch makes overlapping dots visible by
ORing the pixel patterns together.

When dots from different-colored series overlap, a distinct "overlap" color
is used to indicate that multiple series are present in that cell.
"""

from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

# Style to use when multiple series overlap in the same cell
OVERLAP_STYLE = "bright_white"


def apply_braille_or_patch() -> None:
    """Apply the monkey-patch to Canvas for bitwise OR braille combining."""
    from textual_hires_canvas.canvas import Canvas
    from textual_hires_canvas.hires import HiResMode, hires_sizes, pixels

    # Build reverse lookup: braille char -> pixel tuple
    braille_reverse: dict[str, tuple[int, ...]] = {}
    for pattern, char in pixels[HiResMode.BRAILLE].items():
        if char is not None:
            braille_reverse[char] = pattern

    def set_hires_pixels_with_or(
        self: Canvas,
        coordinates: Iterable[tuple[float | NDArray[Any], float | NDArray[Any]]],
        hires_mode: HiResMode | None = None,
        style: str = "white",
    ) -> None:
        """Patched version that ORs pixels instead of overwriting."""
        hires_mode = hires_mode or self.default_hires_mode
        pixel_size = hires_sizes[hires_mode]
        pixel_info = pixels.get(hires_mode)
        if pixel_info is None:
            return

        # Group coordinates by cell
        cells_to_update: dict[tuple[int, int], set[tuple[int, int]]] = {}
        w_factor = pixel_size.width
        h_factor = pixel_size.height

        for x, y in coordinates:
            if not self._canvas_region.contains(floor(x), floor(y)):
                continue
            hx = floor(x * w_factor)
            hy = floor(y * h_factor)
            cell_x = hx // w_factor
            cell_y = hy // h_factor
            offset_x = hx % w_factor
            offset_y = hy % h_factor
            cells_to_update.setdefault((cell_x, cell_y), set()).add((offset_x, offset_y))

        # Process each cell with OR logic
        for (cell_x, cell_y), points in cells_to_update.items():
            in_bounds = 0 <= cell_y < len(self._buffer) and 0 <= cell_x < len(self._buffer[0])

            # Get existing character and style
            existing_char = self._buffer[cell_y][cell_x] if in_bounds else " "
            existing_style = self._styles[cell_y][cell_x] if in_bounds else ""

            # Determine if this is an overlap (different style already present)
            is_overlap = (
                existing_char in braille_reverse
                and existing_style
                and existing_style != style
                and existing_style != OVERLAP_STYLE
            )

            if existing_char in braille_reverse:
                # Decode existing braille to pixel pattern
                existing_pattern = braille_reverse[existing_char]
                cell_buffer = np.array(existing_pattern, dtype=bool).reshape(
                    (pixel_size.height, pixel_size.width)
                )
            else:
                cell_buffer = np.zeros((pixel_size.height, pixel_size.width), dtype=bool)

            # OR in new points
            for offset_x, offset_y in points:
                cell_buffer[offset_y, offset_x] = True

            # Look up combined character
            subpixels = tuple(int(v) for v in cell_buffer.flat)
            if char := pixel_info[subpixels]:
                # Use overlap style if combining different-colored series
                final_style = OVERLAP_STYLE if is_overlap else style
                self.set_pixel(cell_x, cell_y, char=char, style=final_style)

    Canvas.set_hires_pixels = set_hires_pixels_with_or  # type: ignore[ty:invalid-assignment]

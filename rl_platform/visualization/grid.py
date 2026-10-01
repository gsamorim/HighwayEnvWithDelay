"""Compose several agent panels into one dashboard image.

Used by the live multi-agent dashboard. Unlike the first version of this
module, panels are **not** force-resized here: `dashboard_panel.render_panel`
already produces uniformly-sized panels (same `panel_width`, and every agent
renders the same environment so the video aspect ratio matches too), so all
this does now is lay pre-sized panels out on a grid with a visible gutter
between them, which is what actually makes a multi-panel window read as a
dashboard instead of a jammed-together wall of video. Kept tolerant of
mismatched panel sizes (crops/pads instead of raising) so a mixed-size future
agent type doesn't crash the whole app.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize(size, Image.BILINEAR))


def default_cols(n: int) -> int:
    """Picks a landscape-leaning column count instead of `ceil(sqrt(n))`'s
    square-ish grid. Each panel here is a video feed plus a HUD strip below
    it, so panels are already taller than they are wide; a "square" grid of
    them (e.g. 2x2 for 3-4 agents) produces a window noticeably taller than
    it is wide, which is exactly what didn't fit typical (16:9-ish) screens
    at the default panel size. A single row for up to 4 agents, and a grid
    biased toward more columns than rows beyond that, keeps the window
    closer to a landscape shape regardless of panel count."""
    if n <= 4:
        return n
    return math.ceil(math.sqrt(n * 1.6))  # >1 biases toward extra columns, fewer rows


def compose_grid(
    panels: list[np.ndarray],
    cols: int | None = None,
    gutter: int = 14,
    bg_color: tuple[int, int, int] = (20, 22, 27),
) -> np.ndarray:
    """Arrange pre-sized `panels` into a grid with `gutter` pixels of solid
    `bg_color` between/around them (no per-panel resizing)."""
    if not panels:
        return np.full((64, 64, 3), bg_color, dtype=np.uint8)

    n = len(panels)
    if cols is None:
        cols = default_cols(n)
    rows = math.ceil(n / cols)

    panel_h = max(p.shape[0] for p in panels)
    panel_w = max(p.shape[1] for p in panels)

    width = cols * panel_w + (cols + 1) * gutter
    height = rows * panel_h + (rows + 1) * gutter
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        y = gutter + r * (panel_h + gutter)
        x = gutter + c * (panel_w + gutter)
        ph, pw = panel.shape[0], panel.shape[1]
        # Crop defensively in case a panel is somehow larger than the cell
        # (shouldn't happen since panel_h/panel_w are the max over panels).
        canvas[y:y + ph, x:x + pw] = panel[:panel_h, :panel_w]

    return canvas

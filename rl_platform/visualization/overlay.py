"""Draw a small HUD (reward, speed, traffic level, agent name, ...) onto a
rendered RGB frame. Used by both the headless video exporter and the live
pygame viewer so the two paths look identical."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: dict[int, "ImageFont.ImageFont"] = {}


def _get_font(size: int = 16):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size
            )
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def overlay_hud(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Return a copy of `frame` (H, W, 3 uint8) with a translucent HUD panel
    and the given text lines drawn in the top-left corner."""
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(16)

    line_h = 19
    pad = 6
    text_w = max((draw.textlength(line, font=font) for line in lines), default=0)
    panel_w = int(text_w) + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad
    x0, y0 = origin
    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(0, 0, 0, 140))
    for i, line in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * line_h), line, font=font, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composed)

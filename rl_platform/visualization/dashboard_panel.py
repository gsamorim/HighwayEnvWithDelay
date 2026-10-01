"""Rich, fixed-size panel renderer for the live multi-agent dashboard
(`scripts/live_trainer.py`).

This is deliberately separate from the simple `overlay.py` HUD used by
`scripts/record.py`'s single-agent demo clips. Two design choices here fix
problems the first version of the live dashboard had:

1. **The info panel is a solid strip below the track view, not a
   translucent box on top of it.** The old version alpha-blended a
   semi-transparent HUD over the moving scene every frame; besides looking
   messy, that full-frame alpha composite was real per-frame cost (a factor
   in why the window felt like it was flickering/stuttering under load).
   Rendering a small opaque strip and stacking it under the (resized)
   camera frame is both cheaper and reads more like an actual dashboard.
2. **The strip's size never changes.** The old HUD box auto-sized itself to
   its text, so it visibly jittered every time a number gained or lost a
   digit. Here the layout is fixed columns/rows at a fixed panel size, so
   frame-to-frame content changes never move any boundary.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE: dict[tuple[int, bool], "ImageFont.ImageFont"] = {}
# Bundled inside the repo (not a system path like /usr/share/fonts/...): an
# earlier version of this file loaded DejaVu from the Linux system font
# path, which only ever existed in the Linux dev sandbox. On the Windows
# machine this actually runs on, that path doesn't exist, PIL silently fell
# back to its tiny fixed-size bitmap font, and every "human-readable" size
# we tuned here was ignored -- the interface never looked like what was
# reviewed during development. Shipping the .ttf files with the repo and
# loading them relative to this module fixes that on every platform.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_warned_fallback = False


def _font(size: int, bold: bool = False) -> "ImageFont.ImageFont":
    global _warned_fallback
    key = (size, bold)
    if key not in _FONT_CACHE:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            _FONT_CACHE[key] = ImageFont.truetype(os.path.join(_FONT_DIR, name), size)
        except OSError:
            _FONT_CACHE[key] = ImageFont.load_default()
            if not _warned_fallback:
                _warned_fallback = True
                warnings.warn(
                    f"Could not load bundled font from {_FONT_DIR} -- falling back to PIL's tiny "
                    "default bitmap font. The dashboard text sizes will look wrong. Check that "
                    "rl_platform/visualization/fonts/*.ttf shipped with this checkout."
                )
    return _FONT_CACHE[key]


# -- palette ------------------------------------------------------------
BG = (20, 22, 27)             # window/gutter background
PANEL_BG = (32, 35, 42)       # HUD strip background
TEXT = (235, 237, 240)        # primary text
MUTED = (146, 152, 163)       # secondary/label text
WARN = (255, 196, 84)         # paused indicator
ACCEL_POS = (96, 200, 120)    # throttle
ACCEL_NEG = (224, 96, 96)     # brake
STEER_COLOR = (94, 164, 235)  # steering (single color, direction shown by fill side)

TRAFFIC_COLORS = {
    "none": (130, 138, 148),
    "light": (96, 189, 108),
    "medium": (255, 183, 59),
    "heavy": (255, 122, 84),
    "jam": (232, 78, 78),
}


def traffic_color(traffic) -> tuple[int, int, int]:
    if isinstance(traffic, str) and traffic in TRAFFIC_COLORS:
        return TRAFFIC_COLORS[traffic]
    try:
        n = int(traffic)
    except (TypeError, ValueError):
        return (140, 140, 150)
    if n <= 0:
        return TRAFFIC_COLORS["none"]
    if n <= 2:
        return TRAFFIC_COLORS["light"]
    if n <= 6:
        return TRAFFIC_COLORS["medium"]
    if n <= 11:
        return TRAFFIC_COLORS["heavy"]
    return TRAFFIC_COLORS["jam"]


@dataclass
class PanelStats:
    name: str
    agent_type: str
    traffic_label: str
    total_steps: int
    session_steps: int
    episode_count: int
    last_return: float
    speed: float
    crashes: int
    alpha: Optional[float]
    accel: float   # last action[0] (throttle/brake), range roughly [-1, 1]
    steer: float   # last action[1] (steering), range roughly [-1, 1]
    paused: bool


def _draw_gauge_fill(draw: "ImageDraw.ImageDraw", x: int, y: int, w: int, h: int,
                     value: float, pos_color, neg_color) -> None:
    """The part of a gauge that changes every frame: the fill bar (right of
    center for positive values, left for negative, so direction is visible
    at a glance) and the live numeric readout. The track background, label,
    and direction caption never change for a given gauge and live in
    `_build_chrome` instead -- see the comment there for why."""
    value = max(-1.0, min(1.0, float(value)))
    cx = x + w // 2
    half = w // 2 - 3
    fill_w = max(int(abs(value) * half), 3 if abs(value) > 0.02 else 0)
    color = pos_color if value >= 0 else neg_color
    if fill_w:
        if value >= 0:
            draw.rounded_rectangle([cx, y + 2, cx + fill_w, y + h - 2], radius=(h - 4) // 2, fill=color)
        else:
            draw.rounded_rectangle([cx - fill_w, y + 2, cx, y + h - 2], radius=(h - 4) // 2, fill=color)
    draw.line([cx, y - 3, cx, y + h + 3], fill=(100, 105, 114), width=2)

    value_font = _font(14, bold=True)
    val_text = f"{value:+.2f}"
    tw = draw.textlength(val_text, font=value_font)
    draw.text((x + w - tw, y - 20), val_text, font=value_font, fill=TEXT)


_CHROME_CACHE: dict[tuple, "Image.Image"] = {}


def _build_chrome(name: str, traffic_label: str, panel_width: int, hud_height: int) -> "Image.Image":
    """Builds everything in a panel's HUD strip that stays the same for as
    long as an agent keeps the same name/traffic level: the background,
    title, traffic badge, every stat *label* (not its value), and each
    gauge's track background + label + direction caption (not its fill or
    number). Profiling showed `render_panel` -- almost entirely PIL text
    drawing -- taking ~10ms per agent per frame, a real contributor to the
    dashboard's actual achieved frame rate falling far short of any fps
    target. Most of those draw calls were for text that is identical every
    single frame; this is drawn once and cached, and `render_panel` then
    only draws the handful of values that actually change."""
    pad = 14
    hud = Image.new("RGB", (panel_width, hud_height), PANEL_BG)
    draw = ImageDraw.Draw(hud)

    title_font = _font(21, bold=True)
    label_font = _font(14)
    badge_font = _font(13, bold=True)

    draw.text((pad, 9), name.upper(), font=title_font, fill=TEXT)

    badge_text = f"{traffic_label} traffic".upper()
    bt_w = draw.textlength(badge_text, font=badge_font)
    badge_w = bt_w + 18
    bx1 = panel_width - pad - badge_w
    draw.rounded_rectangle([bx1, 12, bx1 + badge_w, 12 + 22], radius=11, fill=traffic_color(traffic_label))
    draw.text((bx1 + 9, 16), badge_text, font=badge_font, fill=(25, 25, 25))

    col_w = (panel_width - 2 * pad) // 3
    row1_y = 46
    for i, label in enumerate(["STEPS TRAINED", "EPISODES", "CRASHES"]):
        draw.text((pad + i * col_w, row1_y), label, font=label_font, fill=MUTED)

    row2_y = row1_y + 42
    for i, label in enumerate(["LAST REWARD", "SPEED", "EXPLORING (α)"]):
        draw.text((pad + i * col_w, row2_y), label, font=label_font, fill=MUTED)

    gauge_y = row2_y + 52
    gauge_w = (panel_width - 3 * pad) // 2
    caption_font = _font(12)
    for gx, label, caption in (
        (pad, "ACCELERATION", "brake ← · → gas"),
        (pad * 2 + gauge_w, "STEERING", "left ← · → right"),
    ):
        draw.rounded_rectangle([gx, gauge_y, gx + gauge_w, gauge_y + 18], radius=9, fill=(52, 56, 64))
        draw.text((gx, gauge_y - 20), label, font=label_font, fill=MUTED)
        cw = draw.textlength(caption, font=caption_font)
        draw.text((gx + gauge_w / 2 - cw / 2, gauge_y + 18 + 6), caption, font=caption_font, fill=MUTED)

    return hud


def render_panel(video_frame: np.ndarray, stats: PanelStats, panel_width: int, hud_height: int = 190) -> np.ndarray:
    """Returns an (video_h + hud_height, panel_width, 3) uint8 image: the
    env's camera frame resized to `panel_width`, with a fixed-size solid HUD
    strip stacked below it (never overlapping the track view)."""
    video_h = max(1, int(round(panel_width * video_frame.shape[0] / video_frame.shape[1])))
    video_img = Image.fromarray(video_frame).resize((panel_width, video_h), Image.BILINEAR)

    chrome_key = (stats.name, stats.traffic_label, panel_width, hud_height)
    chrome = _CHROME_CACHE.get(chrome_key)
    if chrome is None:
        chrome = _build_chrome(stats.name, stats.traffic_label, panel_width, hud_height)
        _CHROME_CACHE[chrome_key] = chrome
    hud = chrome.copy()
    draw = ImageDraw.Draw(hud)
    pad = 14
    value_font = _font(17, bold=True)
    badge_font = _font(13, bold=True)

    if stats.paused:
        pw = draw.textlength("PAUSED", font=badge_font)
        draw.rounded_rectangle([pad, 38, pad + pw + 16, 38 + 20], radius=10, fill=WARN)
        draw.text((pad + 8, 41), "PAUSED", font=badge_font, fill=(40, 30, 0))

    # -- stat grid: two rows of three columns (labels are in the cached
    # chrome above; only the values are drawn every frame) --------------
    col_w = (panel_width - 2 * pad) // 3
    row1_y = 46
    steps_text = f"{stats.total_steps:,}"
    if stats.session_steps:
        steps_text += f" (+{stats.session_steps:,})"
    for i, value in enumerate([steps_text, f"{stats.episode_count:,}", f"{stats.crashes:,}"]):
        draw.text((pad + i * col_w, row1_y + 18), value, font=value_font, fill=TEXT)

    row2_y = row1_y + 42
    alpha_text = f"{stats.alpha:.3f}" if stats.alpha is not None else "warming up"
    for i, value in enumerate([f"{stats.last_return:.1f}", f"{stats.speed:.1f} m/s", alpha_text]):
        draw.text((pad + i * col_w, row2_y + 18), value, font=value_font, fill=TEXT)

    # -- gauges: acceleration + steering, updated every frame -----------
    gauge_y = row2_y + 52
    gauge_w = (panel_width - 3 * pad) // 2
    _draw_gauge_fill(draw, pad, gauge_y, gauge_w, 18, stats.accel, ACCEL_POS, ACCEL_NEG)
    _draw_gauge_fill(draw, pad * 2 + gauge_w, gauge_y, gauge_w, 18, stats.steer, STEER_COLOR, STEER_COLOR)

    combined = Image.new("RGB", (panel_width, video_h + hud_height))
    combined.paste(video_img, (0, 0))
    combined.paste(hud, (0, video_h))
    return np.array(combined)


def _fit_text(draw: "ImageDraw.ImageDraw", text: str, max_width: int, sizes: list[int], bold: bool = False):
    """Picks the largest font size (from `sizes`, descending) whose rendered
    `text` fits in `max_width`; if even the smallest size overflows, the text
    is truncated with an ellipsis at that size. Used so the title bar never
    overlaps or spills off-window at small panel counts/widths -- it shrinks
    or truncates instead."""
    for size in sizes:
        font = _font(size, bold=bold)
        if draw.textlength(text, font=font) <= max_width:
            return font, text
    font = _font(sizes[-1], bold=bold)
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return font, (text + "…") if text else ""


_TITLE_BAR_CACHE: dict[tuple, np.ndarray] = {}


def render_title_bar(width: int, height: int, title: str, hints: str) -> np.ndarray:
    """Two stacked lines (title, then control hints) rather than side by
    side. Side-by-side looked cleaner at very wide panel sizes, but with a
    single agent or a narrow window the two texts collided/overlapped in the
    middle -- stacking fixes that at typical widths, and `_fit_text` shrinks
    or truncates either line so a very narrow window (e.g. one agent panel)
    degrades gracefully instead of spilling off the edge.

    Content here never changes during a run (same title/hints/size every
    frame), so it's cached -- one less thing competing for the per-frame
    time budget."""
    key = (width, height, title, hints)
    cached = _TITLE_BAR_CACHE.get(key)
    if cached is not None:
        return cached

    img = Image.new("RGB", (width, height), (14, 15, 19))
    draw = ImageDraw.Draw(img)
    pad = 16
    max_w = max(10, width - 2 * pad)

    title_font, title_text = _fit_text(draw, title, max_w, [20, 18, 16, 14, 12], bold=True)
    draw.text((pad, 8), title_text, font=title_font, fill=TEXT)

    hint_font, hint_text = _fit_text(draw, hints, max_w, [13, 12, 11, 10])
    draw.text((pad, height - 22), hint_text, font=hint_font, fill=MUTED)
    arr = np.array(img)
    _TITLE_BAR_CACHE[key] = arr
    return arr

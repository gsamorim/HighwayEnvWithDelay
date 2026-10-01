"""Regression tests for the dashboard rendering bugs found after the first
deployment to the user's actual machine: fonts silently falling back to
PIL's tiny bitmap font because they were loaded from a Linux-only system
path, and multi-agent layouts defaulting to a taller-than-wide grid that
didn't fit typical screens. Both were caught by the user, not by testing
here -- these tests exist so they can't quietly regress again.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import ImageFont

from rl_platform.visualization import dashboard_panel
from rl_platform.visualization.dashboard_panel import PanelStats, render_panel
from rl_platform.visualization.grid import default_cols


def test_fonts_are_bundled_with_the_repo_not_a_system_path():
    # The bug: _FONT_DIR used to be hardcoded to /usr/share/fonts/... which
    # only exists in the Linux dev sandbox, not on the Windows machine this
    # is actually deployed to. It must resolve relative to this package.
    assert dashboard_panel._FONT_DIR == os.path.join(
        os.path.dirname(os.path.abspath(dashboard_panel.__file__)), "fonts"
    )
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        path = os.path.join(dashboard_panel._FONT_DIR, name)
        assert os.path.exists(path), f"font not bundled: {path}"


def test_font_loader_actually_loads_a_scalable_truetype_font():
    # If this ever silently falls back to ImageFont.load_default(), every
    # size we tune in render_panel/render_title_bar is ignored and the HUD
    # becomes the tiny fixed-size bitmap font -- exactly what happened on
    # Windows before the fonts were bundled.
    dashboard_panel._FONT_CACHE.clear()
    font = dashboard_panel._font(20, bold=True)
    assert isinstance(font, ImageFont.FreeTypeFont), (
        "font loader fell back to PIL's default bitmap font -- HUD text sizes will be wrong"
    )


def test_default_cols_prefers_landscape_layout_for_small_agent_counts():
    # The bug: ceil(sqrt(n)) put 3-4 agents in a 2x2 grid, which -- because
    # each panel is a video feed plus a HUD strip below it (so already
    # taller than wide) -- produced a window noticeably taller than wide
    # (e.g. 1002x1442 for the default 3-agent run), which didn't fit
    # typical screens. A single row for up to 4 agents keeps the window
    # landscape-shaped instead.
    assert default_cols(1) == 1
    assert default_cols(2) == 2
    assert default_cols(3) == 3
    assert default_cols(4) == 4
    # Beyond that a single row would get absurdly wide, so it falls back to
    # a roughly-square (still landscape-biased) grid.
    assert default_cols(9) < 9


def _make_stats(**overrides) -> PanelStats:
    base = dict(
        name="medium", agent_type="sac", traffic_label="medium",
        total_steps=10, session_steps=10, episode_count=1, last_return=5.0,
        speed=8.0, crashes=0, alpha=0.2, accel=0.5, steer=-0.3, paused=False,
    )
    base.update(overrides)
    return PanelStats(**base)


def test_render_panel_chrome_cache_reuses_static_parts_but_updates_values():
    # render_panel caches the static "chrome" (title, labels, gauge tracks)
    # per (name, traffic_label, panel_width, hud_height) and only draws the
    # changing values on top each frame -- a real perf fix (profiling showed
    # this roughly halving render_panel's cost, a measurable contributor to
    # the low actual frame rate a user reported as "flickering"). The risk
    # a cache like this introduces is stale or leaked state between frames;
    # this test guards against that specifically.
    dashboard_panel._CHROME_CACHE.clear()
    video = np.random.randint(0, 255, size=(64, 96, 3), dtype=np.uint8)

    stats_a1 = _make_stats(total_steps=10, accel=0.5, paused=False)
    frame_a1 = render_panel(video, stats_a1, panel_width=200, hud_height=190)
    assert len(dashboard_panel._CHROME_CACHE) == 1, "first render should build and cache the chrome"

    stats_a2 = _make_stats(total_steps=99_999, accel=-0.9, paused=True)
    frame_a2 = render_panel(video, stats_a2, panel_width=200, hud_height=190)
    assert len(dashboard_panel._CHROME_CACHE) == 1, "same (name, traffic, size) should reuse the cached chrome"
    assert not np.array_equal(frame_a1, frame_a2), "changed values/paused state must actually show up"

    # Re-rendering the original stats must reproduce the original pixels
    # exactly -- if the cached chrome were being mutated in place instead of
    # copied per frame, this would fail (leftover paused-badge, stale gauge
    # fill, etc. bleeding across frames).
    frame_a1_again = render_panel(video, _make_stats(total_steps=10, accel=0.5, paused=False), panel_width=200, hud_height=190)
    assert np.array_equal(frame_a1, frame_a1_again), "cache reuse must not leak state between frames"

    # A different agent name/traffic gets its own cached chrome.
    stats_b = _make_stats(name="heavy", traffic_label="heavy")
    render_panel(video, stats_b, panel_width=200, hud_height=190)
    assert len(dashboard_panel._CHROME_CACHE) == 2

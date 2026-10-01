"""Tests for the stop/resume checkpoint bundling and the pygame-free core of
the live multi-agent trainer (`rl_platform.training.live_app`).

Most of this deliberately avoids `LiveTrainerApp.run()` (the actual pygame
event loop) in favor of testing the training/rendering/checkpoint logic
through `tick_all`/`save_all`/the constructor, which is enough to cover
correctness (training still happens, checkpoints round-trip, resuming
continues rather than restarts) without needing a display. But `run()` DOES
work under `SDL_VIDEODRIVER=dummy` (no real window, but the pygame calls
succeed), and after two rounds of display-mode bugs that only showed up on
the user's actual Windows machine and not in this sandbox, at least the
`pygame.display.set_mode` flags `run()` requests are worth locking in here
-- see `test_run_requests_scaled_display_mode` below.
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from rl_platform.agents import make_agent
from rl_platform.training.checkpointing import save_agent_bundle, load_agent_bundle, bundle_exists
from rl_platform.training.live_app import AgentSpec, LiveTrainerApp


def test_bundle_roundtrip_with_buffer(tmp_path):
    agent = make_agent("sac", obs_dim=10, act_dim=2, learning_starts=8, batch_size=8, buffer_size=200, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        agent.observe(rng.normal(size=10), rng.uniform(-1, 1, size=2), float(rng.normal()), rng.normal(size=10), False)
    for _ in range(5):
        agent.update()

    d = str(tmp_path / "medium")
    assert not bundle_exists(d)
    save_agent_bundle(d, agent, meta={"total_env_steps": 20, "episode_count": 1})
    assert bundle_exists(d)
    assert os.path.exists(os.path.join(d, "buffer.npz"))

    agent2 = make_agent("sac", obs_dim=10, act_dim=2, learning_starts=8, batch_size=8, buffer_size=200, seed=1)
    meta = load_agent_bundle(d, agent2)
    assert meta["total_env_steps"] == 20
    assert agent2.update_count == 5
    assert len(agent2.buffer) == 20


def test_bundle_without_buffer_flag(tmp_path):
    agent = make_agent("sac", obs_dim=10, act_dim=2, seed=0)
    d = str(tmp_path / "no_buffer")
    save_agent_bundle(d, agent, meta={"total_env_steps": 0}, include_buffer=False)
    assert not os.path.exists(os.path.join(d, "buffer.npz"))
    assert bundle_exists(d)


def test_load_missing_bundle_returns_none(tmp_path):
    agent = make_agent("sac", obs_dim=10, act_dim=2, seed=0)
    assert load_agent_bundle(str(tmp_path / "does_not_exist"), agent) is None


def test_live_app_ticks_train_and_render(tmp_path):
    specs = [
        AgentSpec(name="a", traffic="none", agent_type="sac", start_steps=5,
                  agent_kwargs=dict(buffer_size=500, batch_size=16, learning_starts=16)),
        AgentSpec(name="b", traffic="medium", agent_type="sac", start_steps=5,
                  agent_kwargs=dict(buffer_size=500, batch_size=16, learning_starts=16)),
    ]
    app = LiveTrainerApp(specs, checkpoint_root=str(tmp_path / "ckpt"), steps_per_frame=3, panel_size=64, seed=0)
    try:
        app.tick_all(60)
        assert app.slots[0].total_env_steps == 60
        assert app.slots[1].total_env_steps == 60
        # with 60 steps and learning_starts=16, both agents should have taken real gradient steps
        assert app.slots[0].agent.update_count > 0
        assert app.slots[1].agent.update_count > 0

        frame = app.render_frame()
        # 2 agents -> 2x1 (ceil(sqrt(2))=2 cols, 1 row) grid of panel_width=64
        # panels, each panel_width wide plus a fixed HUD strip below the
        # video, laid out with a gutter and a title bar on top. Exact pixel
        # counts are an implementation detail of dashboard_panel/grid, so
        # assert the structural relationships instead of hardcoding them.
        assert frame.ndim == 3 and frame.shape[2] == 3
        expected_width = 2 * app.panel_size + 3 * app.gutter  # 2 panels + 3 gutters (left/mid/right)
        assert frame.shape[1] == expected_width
        assert frame.shape[0] > app.hud_height + app.title_bar_height, "frame should include video + HUD + title bar"
        assert frame.std() > 1.0, "dashboard frame looks blank"
    finally:
        app.close()


def test_live_app_tracks_last_action_for_hud_gauges(tmp_path):
    specs = [AgentSpec(name="a", traffic="none", agent_type="sac", start_steps=5,
                        agent_kwargs=dict(buffer_size=500, batch_size=16, learning_starts=16))]
    app = LiveTrainerApp(specs, checkpoint_root=str(tmp_path / "ckpt"), steps_per_frame=1, panel_size=64, seed=0)
    try:
        assert app.slots[0].last_action is None  # nothing stepped yet
        app.tick_all(1)
        slot = app.slots[0]
        assert slot.last_action is not None
        assert slot.last_action.shape == (2,)  # [acceleration, steering]

        stats = app._panel_stats(slot)
        assert stats.accel == float(slot.last_action[0])
        assert stats.steer == float(slot.last_action[1])
    finally:
        app.close()


def test_live_app_stop_and_resume_continues_not_restarts(tmp_path):
    ckpt_root = str(tmp_path / "ckpt")
    specs = [AgentSpec(name="solo", traffic="none", agent_type="sac", start_steps=5,
                        agent_kwargs=dict(buffer_size=500, batch_size=16, learning_starts=16))]

    app1 = LiveTrainerApp(specs, checkpoint_root=ckpt_root, steps_per_frame=1, panel_size=64, seed=0)
    app1.tick_all(50)
    steps_before_save = app1.slots[0].total_env_steps
    updates_before_save = app1.slots[0].agent.update_count
    app1.save_all()
    app1.close()
    assert steps_before_save == 50
    assert updates_before_save > 0

    # Simulate "closing the game and reopening it": a brand new app instance
    # pointed at the same checkpoint root should pick up where app1 left off.
    app2 = LiveTrainerApp(specs, checkpoint_root=ckpt_root, steps_per_frame=1, panel_size=64, seed=1)
    try:
        assert app2.slots[0].total_env_steps == steps_before_save, "resumed agent should not restart its step count at 0"
        assert app2.slots[0].agent.update_count == updates_before_save
        assert len(app2.slots[0].agent.buffer) == 50, "replay buffer should also have been restored"

        app2.tick_all(20)
        assert app2.slots[0].total_env_steps == steps_before_save + 20
    finally:
        app2.close()


def test_live_app_pause_prevents_training(tmp_path):
    specs = [AgentSpec(name="a", traffic="none", agent_type="sac", start_steps=5,
                        agent_kwargs=dict(buffer_size=500, batch_size=16, learning_starts=16))]
    app = LiveTrainerApp(specs, checkpoint_root=str(tmp_path / "ckpt"), steps_per_frame=1, panel_size=64, seed=0)
    try:
        app.slots[0].paused = True
        app.tick_all(10)
        assert app.slots[0].total_env_steps == 0
    finally:
        app.close()


def test_run_requests_scaled_display_mode(tmp_path):
    # Regression test for a real bug: an earlier version used plain
    # pygame.DOUBLEBUF (optionally with a vsync= kwarg that silently does
    # nothing without it), which does not get you real vsync/double
    # buffering on most Windows SDL drivers -- the user saw persistent
    # flicker despite that "fix". pygame.SCALED is what actually puts
    # rendering on SDL2's real presentation path. This can't verify there's
    # no visible flicker on a real screen (this sandbox has none), but it
    # does lock in that `run()` actually requests the right display mode
    # rather than silently regressing to the old flags again.
    calls = []
    real_set_mode = pygame.display.set_mode

    def _recording_set_mode(size, flags=0, *args, **kwargs):
        calls.append((size, flags, kwargs))
        return real_set_mode(size, flags, *args, **kwargs)

    specs = [AgentSpec(name="a", traffic="none", agent_type="sac", start_steps=5,
                        agent_kwargs=dict(buffer_size=200, batch_size=16, learning_starts=16))]
    app = LiveTrainerApp(specs, checkpoint_root=str(tmp_path / "ckpt"), steps_per_frame=1, panel_size=200, fps=30, seed=0)
    try:
        import unittest.mock as mock
        with mock.patch("pygame.display.set_mode", side_effect=_recording_set_mode):
            app.run(max_ticks=2)
    finally:
        app.close()

    assert calls, "run() never called pygame.display.set_mode"
    size, flags, kwargs = calls[0]
    assert flags & pygame.SCALED, "run() should request pygame.SCALED for a real vsync'd presentation path"
    assert flags & pygame.DOUBLEBUF
    assert kwargs.get("vsync") == 1 or "vsync" not in kwargs  # allow the TypeError fallback path


def test_random_agent_slot_has_no_buffer_and_still_saves(tmp_path):
    specs = [AgentSpec(name="baseline", traffic="none", agent_type="random")]
    app = LiveTrainerApp(specs, checkpoint_root=str(tmp_path / "ckpt"), steps_per_frame=1, panel_size=64, seed=0)
    try:
        app.tick_all(10)
        assert app.slots[0].total_env_steps == 10
        app.save_all()  # must not raise even though RandomAgent has no .buffer
    finally:
        app.close()

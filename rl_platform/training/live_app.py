"""Continuous multi-agent live training + pygame dashboard.

This is the engine behind `scripts/live_trainer.py`: it keeps N agents
(possibly different traffic levels, possibly different agent architectures
from the registry) training simultaneously against their own environment
instances, renders all of them into one pygame window every frame, and
persists each agent's full training state (weights, optimizer, replay
buffer, and cumulative step/episode counters) so that closing the window
(or Ctrl+C, or a crash -- periodic autosave covers that too) and reopening
it later continues training exactly where it left off, rather than starting
over.

Split into a pygame-free core (`AgentSlot`, `LiveTrainerApp.tick_*`,
`.render_frame`, `.save_all`/`.load_all`) and a thin pygame event loop
(`.run`), so the training/rendering/checkpoint logic is unit-testable
without a display.

Rendering pipeline (see `rl_platform/visualization/dashboard_panel.py` for
why it's built this way): each agent gets a fixed-size panel (video feed +
solid HUD strip with always-on acceleration/steering gauges), panels are
laid out on a grid with a gutter, and a title bar with control hints sits on
top. `run()` avoids the flicker the first version had by never reallocating
a pygame Surface per frame (it blits pixel data into one persistent
`Surface` instead) and by using a double-buffered, vsync'd display mode.
"""
from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..agents.base import Agent
from ..agents.registry import make_agent
from ..envs.registry import make_env
from ..visualization import dashboard_panel
from ..visualization.dashboard_panel import PanelStats, render_panel, render_title_bar
from ..visualization.grid import compose_grid
from ..visualization.video_export import save_video
from .checkpointing import save_agent_bundle, load_agent_bundle, bundle_exists

APP_TITLE = "HighwayEnv Racetrack — Live Multi-Agent SAC Training"
CONTROL_HINTS = "[SPACE] pause all   [1-9] pause one   [S] save now   [F11] fullscreen   [ESC] quit & save"


@dataclass
class AgentSpec:
    name: str
    traffic: Any = "medium"
    agent_type: str = "sac"
    warm_start_path: Optional[str] = None   # optional: initialize weights from an existing flat checkpoint
    start_steps: int = 1000                 # pure-random exploration steps before using the policy (counts cumulative total_env_steps, so a resumed agent past this threshold won't re-explore randomly)
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSlot:
    name: str
    traffic: Any
    agent_type: str
    env: Any
    agent: Agent
    checkpoint_dir: str
    start_steps: int = 1000

    obs: Optional[np.ndarray] = None
    episode_return: float = 0.0
    episode_len: int = 0
    episode_count: int = 0
    last_episode_return: float = 0.0
    total_env_steps: int = 0     # cumulative across resumes -- what gets saved/loaded
    session_env_steps: int = 0   # steps taken just since this process started
    crashes: int = 0
    last_speed: float = 0.0
    last_info: dict = field(default_factory=dict)
    last_metrics: dict = field(default_factory=dict)
    last_action: Optional[np.ndarray] = None   # most recent [acceleration, steering], shown live in the HUD
    paused: bool = False
    resumed_from_step: int = 0


class LiveTrainerApp:
    def __init__(
        self,
        specs: list[AgentSpec],
        checkpoint_root: str = "results/live_checkpoints",
        steps_per_frame: int = 1,
        fps: int = 60,
        autosave_every_steps: int = 2000,
        panel_size: int = 480,
        gutter: int = 14,
        hud_height: int = 190,
        title_bar_height: int = 60,
        seed: int = 0,
        env_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.checkpoint_root = checkpoint_root
        # Measured (see git history / RESULTS.md), on the dev sandbox's CPU
        # with 3 agents: a full SAC env-step + gradient update takes ~8-9ms,
        # and rendering one agent's panel (HighwayEnv's own drawing plus our
        # HUD) takes another ~15-18ms. With the old steps_per_frame=4
        # default that's roughly (4 steps x 3 agents x ~9ms) + (3 x ~18ms)
        # ~= 160ms per rendered frame -- under 7fps no matter what pygame
        # display flags are used, which is what a "flickering"/stuttery
        # feel almost certainly actually was. steps_per_frame=1 cuts the
        # training share of that by 4x; raise it back up (via
        # --steps-per-frame) if you'd rather trade render smoothness for
        # faster wall-clock training throughput.
        self.steps_per_frame = steps_per_frame
        self.fps = fps
        self.autosave_every_steps = autosave_every_steps
        self.panel_size = panel_size
        self.gutter = gutter
        self.hud_height = hud_height
        self.title_bar_height = title_bar_height
        self.env_kwargs = env_kwargs or {}
        self.slots: list[AgentSlot] = []
        self._last_autosave: dict[str, int] = {}

        for i, spec in enumerate(specs):
            env = make_env(
                "racetrack",
                traffic_density=spec.traffic,
                render_mode="rgb_array",
                **self.env_kwargs,
            )
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            agent = make_agent(
                spec.agent_type,
                obs_dim=obs_dim,
                act_dim=act_dim,
                seed=seed + i,
                **spec.agent_kwargs,
            )
            checkpoint_dir = os.path.join(checkpoint_root, spec.name)

            slot = AgentSlot(
                name=spec.name,
                traffic=spec.traffic,
                agent_type=spec.agent_type,
                env=env,
                agent=agent,
                checkpoint_dir=checkpoint_dir,
                start_steps=spec.start_steps,
            )

            meta = load_agent_bundle(checkpoint_dir, agent)
            if meta is not None:
                slot.total_env_steps = int(meta.get("total_env_steps", 0))
                slot.episode_count = int(meta.get("episode_count", 0))
                slot.crashes = int(meta.get("crashes", 0))
                slot.resumed_from_step = slot.total_env_steps
                print(f"[{spec.name}] resumed from checkpoint: {slot.total_env_steps} env steps, "
                      f"{slot.episode_count} episodes already trained.")
            elif spec.warm_start_path:
                agent.load(spec.warm_start_path)
                print(f"[{spec.name}] warm-started weights from {spec.warm_start_path} (fresh optimizer/buffer).")
            else:
                print(f"[{spec.name}] no checkpoint found -- starting fresh.")

            slot.obs, _ = env.reset(seed=seed + i)
            self.slots.append(slot)
            self._last_autosave[spec.name] = slot.total_env_steps

    # --- pure simulation/training, no pygame -----------------------------
    def tick_slot(self, slot: AgentSlot) -> None:
        if slot.paused:
            return
        if slot.total_env_steps < slot.start_steps:
            action = slot.env.action_space.sample()
        else:
            action = slot.agent.act(slot.obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = slot.env.step(action)
        done = terminated or truncated
        slot.agent.observe(slot.obs, action, float(reward), next_obs, done)
        slot.obs = next_obs
        slot.last_action = np.asarray(action, dtype=np.float32)
        slot.episode_return += float(reward)
        slot.episode_len += 1
        slot.total_env_steps += 1
        slot.session_env_steps += 1
        slot.last_speed = float(info.get("speed", slot.last_speed))
        slot.last_info = info

        if slot.total_env_steps > slot.start_steps:
            metrics = slot.agent.update()
            if metrics:
                slot.last_metrics = metrics

        if done:
            slot.crashes += int(info.get("crashed", False))
            slot.episode_count += 1
            slot.last_episode_return = slot.episode_return
            slot.episode_return = 0.0
            slot.episode_len = 0
            slot.obs, _ = slot.env.reset()

    def tick_all(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            for slot in self.slots:
                self.tick_slot(slot)

    def maybe_autosave(self) -> None:
        if self.autosave_every_steps <= 0:
            return
        for slot in self.slots:
            if slot.total_env_steps - self._last_autosave[slot.name] >= self.autosave_every_steps:
                self._save_slot(slot)
                self._last_autosave[slot.name] = slot.total_env_steps

    # --- rendering (pure numpy -- testable without a display) ------------
    def _panel_stats(self, slot: AgentSlot) -> PanelStats:
        accel = steer = 0.0
        if slot.last_action is not None and len(slot.last_action) >= 1:
            accel = float(slot.last_action[0])
        if slot.last_action is not None and len(slot.last_action) >= 2:
            steer = float(slot.last_action[1])
        alpha = slot.last_metrics.get("alpha") if slot.last_metrics else None
        return PanelStats(
            name=slot.name,
            agent_type=slot.agent_type,
            traffic_label=str(slot.traffic),
            total_steps=slot.total_env_steps,
            session_steps=slot.session_env_steps,
            episode_count=slot.episode_count,
            last_return=slot.last_episode_return,
            speed=slot.last_speed,
            crashes=slot.crashes,
            alpha=float(alpha) if alpha is not None else None,
            accel=accel,
            steer=steer,
            paused=slot.paused,
        )

    def render_frame(self) -> np.ndarray:
        panels = []
        for slot in self.slots:
            video = slot.env.render()
            stats = self._panel_stats(slot)
            panels.append(render_panel(video, stats, panel_width=self.panel_size, hud_height=self.hud_height))
        grid = compose_grid(panels, gutter=self.gutter, bg_color=dashboard_panel.BG)
        title_bar = render_title_bar(grid.shape[1], self.title_bar_height, APP_TITLE, CONTROL_HINTS)
        return np.vstack([title_bar, grid])

    # --- persistence -------------------------------------------------------
    def _save_slot(self, slot: AgentSlot) -> None:
        meta = {
            "name": slot.name,
            "traffic": slot.traffic,
            "agent_type": slot.agent_type,
            "total_env_steps": slot.total_env_steps,
            "episode_count": slot.episode_count,
            "crashes": slot.crashes,
            "last_episode_return": slot.last_episode_return,
        }
        save_agent_bundle(slot.checkpoint_dir, slot.agent, meta)

    def save_all(self) -> None:
        for slot in self.slots:
            self._save_slot(slot)

    def close(self) -> None:
        for slot in self.slots:
            slot.env.close()

    # --- the actual interactive loop (needs pygame + ideally a display) --
    def run(self, max_ticks: Optional[int] = None, max_runtime_seconds: Optional[float] = None,
             record_path: Optional[str] = None, record_every_n_frames: int = 1,
             start_fullscreen: bool = False) -> None:
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no sound needed; avoids noisy ALSA warnings on Linux boxes with no audio device
        import pygame  # local import: everything above this line works without pygame installed

        pygame.init()

        # How big should the dashboard actually be rendered? Query the real
        # screen up front and, if the dashboard's native size wouldn't fit
        # it, shrink panel_size/gutter/hud_height/title_bar_height (all
        # proportionally) so a *fresh* render already matches a size that
        # fits -- rather than rendering at a fixed size and resampling every
        # frame to fit the window. Two earlier things plausibly caused the
        # persistent flicker report: an oversized native window, and (in the
        # attempt that fixed the sizing) a per-frame pygame.transform.smoothscale
        # call that a plain software-blit display doesn't present cleanly.
        # Rendering natively at the final size removes the second cause
        # entirely; see _apply_display_mode below for the first.
        try:
            desktop_info = pygame.display.Info()
            desktop_w, desktop_h = desktop_info.current_w, desktop_info.current_h
        except pygame.error:
            desktop_w = desktop_h = None

        if desktop_w and desktop_h:
            usable_w = max(320, desktop_w - 80)
            usable_h = max(240, desktop_h - 140)
            probe = self.render_frame()
            probe_h, probe_w = probe.shape[0], probe.shape[1]
            shrink = min(usable_w / probe_w, usable_h / probe_h, 1.0)
            if shrink < 0.98:  # only bother resizing if it doesn't already comfortably fit
                self.panel_size = max(160, int(self.panel_size * shrink))
                self.gutter = max(4, int(self.gutter * shrink))
                self.hud_height = max(120, int(self.hud_height * shrink))
                self.title_bar_height = max(36, int(self.title_bar_height * shrink))

        frame0 = self.render_frame()
        content_h, content_w = frame0.shape[0], frame0.shape[1]

        # One persistent Surface holding the dashboard pixels, updated in
        # place via blit_array every frame instead of being reallocated
        # (allocate-and-throw-away Surfaces every frame was real per-frame
        # cost in the very first version of this dashboard).
        content_surface = pygame.Surface((content_w, content_h))

        clock = pygame.time.Clock()
        state = {"running": True, "fullscreen": start_fullscreen}
        screen = None

        def _apply_display_mode():
            # pygame.SCALED puts the window on SDL2's actual renderer/texture
            # presentation path instead of the legacy raw-surface path (which
            # on Windows typically ends up as GDI BitBlt calls with no real
            # vsync) -- this is pygame's own documented mechanism for a
            # tear-free, flicker-free display, and combined with
            # pygame.FULLSCREEN it also does the fullscreen scale-to-fit in
            # hardware, so we no longer need our own manual
            # smoothscale/letterbox code at all. A plain DOUBLEBUF flag
            # alone (what earlier versions used) does not get you this on
            # most Windows SDL drivers -- it's mostly a formality without
            # SCALED or an OPENGL context.
            nonlocal screen
            flags = pygame.DOUBLEBUF | pygame.SCALED
            if state["fullscreen"]:
                flags |= pygame.FULLSCREEN
            try:
                screen = pygame.display.set_mode((content_w, content_h), flags, vsync=1)
            except (TypeError, pygame.error):
                try:
                    screen = pygame.display.set_mode((content_w, content_h), flags)
                except pygame.error:
                    # This SDL driver rejects SCALED/FULLSCREEN/vsync outright
                    # -- fall back to the plainest possible window rather
                    # than crash the whole app over a nice-to-have.
                    screen = pygame.display.set_mode(
                        (content_w, content_h),
                        pygame.DOUBLEBUF | (pygame.FULLSCREEN if state["fullscreen"] else 0),
                    )
            # HighwayEnv's EnvViewer resets the window caption to "Highway-env"
            # the first time each agent renders (during the probe/frame0
            # calls above); re-assert ours every time we (re)create the window.
            pygame.display.set_caption(APP_TITLE)

        _apply_display_mode()
        content_surface = content_surface.convert()  # match the display's pixel format once

        def _handle_signal(signum, frame):
            state["running"] = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        recorded_frames = [] if record_path else None
        start_time = time.time()
        tick = 0

        # Real, measured frame rate, printed periodically -- after two
        # rounds of guessing at display-buffer causes for a "flickering"
        # report that later turned out to mostly be a low actual frame rate
        # (heavy per-frame training + rendering cost, not a presentation
        # bug), this exists so the next report can be "console says X fps"
        # instead of another guess. If X is well below --fps, that's not a
        # pygame/display problem -- it means training+rendering genuinely
        # can't keep up, and the fix is steps-per-frame/panel-size/agent
        # count, not the display mode.
        fps_report_every = 3.0  # seconds
        fps_window_start = time.time()
        fps_window_frames = 0

        fit_note = ""
        if desktop_w and (content_w > desktop_w or content_h > desktop_h):
            fit_note = " (shrunk to fit your screen)"
        print(f"Live training started. Dashboard rendering at {content_w}x{content_h}{fit_note}, "
              f"target {self.fps} fps. Controls: [ESC/close window] quit & save  [S] save now  "
              "[SPACE] pause/resume all  [1-9] pause/resume one agent  [F11] toggle windowed fullscreen.")
        try:
            while state["running"]:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        state["running"] = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            state["running"] = False
                        elif event.key == pygame.K_s:
                            self.save_all()
                            print("Manual save complete.")
                        elif event.key == pygame.K_F11:
                            state["fullscreen"] = not state["fullscreen"]
                            _apply_display_mode()
                        elif event.key == pygame.K_SPACE:
                            new_state = not all(s.paused for s in self.slots)
                            for s in self.slots:
                                s.paused = new_state
                        elif pygame.K_1 <= event.key <= pygame.K_9:
                            idx = event.key - pygame.K_1
                            if idx < len(self.slots):
                                self.slots[idx].paused = not self.slots[idx].paused

                if not state["running"]:
                    break

                self.tick_all(self.steps_per_frame)
                self.maybe_autosave()

                frame = self.render_frame()
                # Copy pixels into the *existing* Surface rather than
                # creating a new one each frame.
                pygame.surfarray.blit_array(content_surface, frame.swapaxes(0, 1))
                # No manual scaling here: content_surface is exactly
                # content_w x content_h, matching the logical resolution
                # `_apply_display_mode` requested, so pygame.SCALED handles
                # presenting it (identically in windowed mode, scaled up to
                # fill the screen in fullscreen) on its own.
                screen.blit(content_surface, (0, 0))
                pygame.display.flip()

                if recorded_frames is not None and tick % record_every_n_frames == 0:
                    recorded_frames.append(frame)

                clock.tick(self.fps)
                tick += 1
                fps_window_frames += 1
                now = time.time()
                if now - fps_window_start >= fps_report_every:
                    actual_fps = fps_window_frames / (now - fps_window_start)
                    print(f"[perf] {actual_fps:.1f} fps actual (target {self.fps})")
                    fps_window_start = now
                    fps_window_frames = 0

                if max_ticks is not None and tick >= max_ticks:
                    print(f"Reached max_ticks={max_ticks}, stopping.")
                    break
                if max_runtime_seconds is not None and time.time() - start_time >= max_runtime_seconds:
                    print(f"Reached max_runtime_seconds={max_runtime_seconds}, stopping.")
                    break
        finally:
            print("Saving all agents before exit ...")
            self.save_all()
            for slot in self.slots:
                print(f"  [{slot.name}] total_env_steps={slot.total_env_steps} "
                      f"episodes={slot.episode_count} crashes={slot.crashes}")
            pygame.quit()
            self.close()
            if recorded_frames:
                save_video(recorded_frames, record_path, fps=self.fps)
                print(f"Saved dashboard timelapse to {record_path}")

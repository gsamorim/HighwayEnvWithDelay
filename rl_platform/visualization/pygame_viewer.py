"""Real-time pygame viewer.

HighwayEnv already ships a pygame-based renderer (`highway_env.envs.common.
graphics.EnvViewer`) which opens a live window when the env is created with
`render_mode="human"`. This module drives that window with a trained (or
random) agent and layers a small HUD (episode return, speed, traffic level,
alpha) on top using the same `overlay_hud` helper used for recorded videos,
so live viewing and exported videos look consistent.

Run this on a machine with an actual display -- it will not produce visible
output in a headless cloud sandbox (use `scripts/record.py` there instead to
export an MP4/GIF you can watch afterwards).
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import pygame

from .overlay import overlay_hud


def watch(
    env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 5,
    seed: Optional[int] = None,
    hud_extra: Optional[dict] = None,
    fps: int = 15,
) -> None:
    """Run `n_episodes` in a live pygame window.

    `act_fn` maps an observation to an action (e.g. `agent.act(obs,
    deterministic=True)`). `hud_extra` is a dict of static strings/values
    (e.g. {"agent": "SAC", "traffic": "heavy"}) shown alongside the live
    per-step stats.
    """
    hud_extra = hud_extra or {}
    clock = pygame.time.Clock()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=None if seed is None else seed + ep)
        done = False
        ep_return = 0.0
        step = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            step += 1

            frame = env.render()
            if frame is not None:
                lines = [
                    f"episode {ep + 1}/{n_episodes}  step {step}",
                    f"return: {ep_return:6.2f}",
                    f"speed: {info.get('speed', float('nan')):.2f}",
                    f"crashed: {info.get('crashed', False)}",
                ] + [f"{k}: {v}" for k, v in hud_extra.items()]
                frame = overlay_hud(frame, lines)
                surface = pygame.display.get_surface()
                if surface is not None:
                    pygame_frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    surface.blit(pygame_frame, (0, 0))
                    pygame.display.flip()
            clock.tick(fps)

"""Headless rollout recording -> MP4/GIF.

HighwayEnv already renders every frame with pygame internally (see
`highway_env.envs.common.graphics.EnvViewer`); with `render_mode="rgb_array"`
it hands us back the rendered frame as a numpy array with no visible window
required, which is exactly what a cloud sandbox without a display needs. This
module drives an agent through episodes and stitches the frames into a video
so the rollout can be watched outside the sandbox.

For an interactive, real-time pygame *window*, see `pygame_viewer.py` instead
(meant to be run on a machine with an actual display).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import imageio.v2 as imageio
import numpy as np


def record_episode(
    env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    seed: Optional[int] = None,
    max_steps: int = 2000,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Roll out one episode, returning the list of rendered frames and summary info."""
    frames = []
    obs, info = env.reset(seed=seed)
    frames.append(env.render())
    total_return = 0.0
    crashed = False
    speeds = []
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_return += float(reward)
        crashed = crashed or bool(info.get("crashed", False))
        if "speed" in info:
            speeds.append(float(info["speed"]))
        frames.append(env.render())
        steps += 1
    summary = {
        "return": total_return,
        "crashed": crashed,
        "mean_speed": float(np.mean(speeds)) if speeds else float("nan"),
        "steps": steps,
    }
    return frames, summary


def save_video(frames: list[np.ndarray], path: str, fps: int = 15) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 15, max_frames: int = 300) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    imageio.mimsave(path, frames, duration=1000.0 / fps, loop=0)

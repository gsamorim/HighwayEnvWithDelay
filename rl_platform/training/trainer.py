"""Generic off-policy training loop.

Deliberately agent-agnostic: it only relies on the `Agent` interface
(`act`, `observe`, `update`, `ready_to_update`), so swapping SAC for another
off-policy algorithm registered in `rl_platform.agents.registry` requires no
changes here. (An on-policy agent like PPO would need its own loop, added
alongside this one and selected by the training script -- the `Agent`
interface is deliberately loose to allow that later.)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..agents.base import Agent
from .logger import RunLogger


@dataclass
class TrainerConfig:
    total_steps: int = 50_000
    start_steps: int = 2_000       # pure-random exploration steps before using the policy
    eval_every_steps: int = 5_000
    eval_episodes: int = 5
    checkpoint_every_steps: int = 10_000
    log_dir: str = "results/logs"
    checkpoint_dir: str = "results/checkpoints"
    run_name: str = "run"
    seed: int = 0
    max_episode_steps: Optional[int] = None  # None = rely on env's own truncation


class Trainer:
    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent: Agent,
        config: TrainerConfig,
        eval_env_factory: Optional[Callable[[], Any]] = None,
    ):
        self.env_factory = env_factory
        self.eval_env_factory = eval_env_factory or env_factory
        self.agent = agent
        self.cfg = config
        self.logger = RunLogger(config.log_dir, config.run_name)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []

    def _rollout_episode_metrics(self, info: dict) -> dict[str, Any]:
        return {
            "crashed": bool(info.get("crashed", False)),
            "speed": float(info.get("speed", np.nan)),
        }

    def train(self) -> dict[str, list]:
        env = self.env_factory()
        obs, info = env.reset(seed=self.cfg.seed)

        ep_return, ep_len = 0.0, 0
        ep_crashed = False
        ep_speeds: list[float] = []
        episode_idx = 0
        t_start = time.time()

        for step in range(1, self.cfg.total_steps + 1):
            if step <= self.cfg.start_steps:
                action = env.action_space.sample()
            else:
                action = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.agent.observe(obs, action, reward, next_obs, done)
            obs = next_obs

            ep_return += float(reward)
            ep_len += 1
            ep_crashed = ep_crashed or bool(info.get("crashed", False))
            if "speed" in info:
                ep_speeds.append(float(info["speed"]))

            update_metrics = {}
            if step > self.cfg.start_steps:
                update_metrics = self.agent.update()

            if done:
                episode_idx += 1
                row = {
                    "step": step,
                    "episode": episode_idx,
                    "episode_return": round(ep_return, 4),
                    "episode_len": ep_len,
                    "crashed": int(ep_crashed),
                    "mean_speed": round(float(np.mean(ep_speeds)), 3) if ep_speeds else float("nan"),
                    **update_metrics,
                }
                self.logger.log_row(row)
                self.history.append(row)
                obs, info = env.reset()
                ep_return, ep_len, ep_crashed, ep_speeds = 0.0, 0, False, []

            if step % self.cfg.eval_every_steps == 0:
                eval_stats = self.evaluate(self.cfg.eval_episodes, seed=10_000 + step)
                eval_stats["step"] = step
                eval_stats["wall_time_s"] = round(time.time() - t_start, 1)
                self.eval_history.append(eval_stats)
                print(
                    f"[eval @ step {step:>7d}] "
                    f"return={eval_stats['mean_return']:.2f}±{eval_stats['std_return']:.2f} "
                    f"crash_rate={eval_stats['crash_rate']:.2f} "
                    f"mean_speed={eval_stats['mean_speed']:.2f} "
                    f"({eval_stats['wall_time_s']:.0f}s elapsed)"
                )

            if step % self.cfg.checkpoint_every_steps == 0:
                ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.run_name}_step{step}.pt")
                self.agent.save(ckpt_path)

        final_ckpt = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.run_name}_final.pt")
        self.agent.save(final_ckpt)
        self.logger.close()
        env.close()
        return {"train_history": self.history, "eval_history": self.eval_history}

    def evaluate(self, n_episodes: int, seed: int = 0, render_frames: bool = False):
        env = self.eval_env_factory()
        returns, crashes, speeds, lengths = [], [], [], []
        frames_by_episode = []
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return, ep_len, ep_speeds = 0.0, 0, []
            crashed = False
            frames = []
            while not done:
                action = self.agent.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
                ep_len += 1
                crashed = crashed or bool(info.get("crashed", False))
                if "speed" in info:
                    ep_speeds.append(float(info["speed"]))
                if render_frames:
                    frames.append(env.render())
            returns.append(ep_return)
            crashes.append(float(crashed))
            speeds.append(float(np.mean(ep_speeds)) if ep_speeds else float("nan"))
            lengths.append(ep_len)
            if render_frames:
                frames_by_episode.append(frames)
        env.close()
        stats = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "crash_rate": float(np.mean(crashes)),
            "mean_speed": float(np.nanmean(speeds)),
            "mean_length": float(np.mean(lengths)),
            "n_episodes": n_episodes,
        }
        if render_frames:
            stats["frames_by_episode"] = frames_by_episode
        return stats

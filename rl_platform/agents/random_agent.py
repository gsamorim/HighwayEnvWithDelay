"""Trivial random-action agent.

Serves two purposes: (1) a sanity-check baseline to compare SAC's learning
curve against, and (2) a minimal reference implementation of the `Agent`
interface for anyone adding a new architecture later.
"""
from __future__ import annotations

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, act_dim: int, seed: int | None = None):
        self.act_dim = act_dim
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)

    def observe(self, obs, action, reward, next_obs, done) -> None:
        pass

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

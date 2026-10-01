"""Abstract agent interface.

Every agent architecture (SAC now; PPO/TD3/etc. later) implements this
interface so that the training loop, evaluation script, and visualizer never
need to know which algorithm is running underneath. To add a new agent:

    from rl_platform.agents.base import Agent
    from rl_platform.agents.registry import register_agent

    class MyAgent(Agent):
        ...

    register_agent("my_agent", MyAgent)

and it becomes selectable via `--agent my_agent` in the scripts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Agent(ABC):
    """Common interface for RL agents acting in continuous-action environments."""

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action for a single observation (unbatched)."""

    @abstractmethod
    def observe(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition (e.g. into a replay buffer). No-op for on-policy agents
        that instead collect trajectories elsewhere."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Perform one (or more) gradient update(s) and return scalar metrics for
        logging (e.g. {'critic_loss': ..., 'actor_loss': ..., 'alpha': ...})."""

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    def ready_to_update(self) -> bool:
        """Whether enough data has been collected to start training. Default True;
        off-policy agents typically override to require a minimum buffer size."""
        return True

"""A simple fixed-size FIFO replay buffer for off-policy agents (SAC, TD3, ...)."""
from __future__ import annotations

import os

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def save(self, path: str) -> None:
        """Persist the buffer's contents (only up to `_size` valid rows) so
        training can resume with a warm replay buffer instead of an empty
        one. Used by `scripts/live_trainer.py` for full resumability."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n = self._size
        np.savez_compressed(
            path,
            obs=self.obs[:n],
            next_obs=self.next_obs[:n],
            actions=self.actions[:n],
            rewards=self.rewards[:n],
            dones=self.dones[:n],
            ptr=self._ptr,
            size=self._size,
            capacity=self.capacity,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        n = int(data["size"])
        cap = int(data["capacity"])
        if cap != self.capacity:
            # Buffer was resized between runs -- still load what we can,
            # truncated/placed at the start, rather than failing outright.
            n = min(n, self.capacity)
        self.obs[:n] = data["obs"][:n]
        self.next_obs[:n] = data["next_obs"][:n]
        self.actions[:n] = data["actions"][:n]
        self.rewards[:n] = data["rewards"][:n]
        self.dones[:n] = data["dones"][:n]
        self._size = n
        self._ptr = int(data["ptr"]) % self.capacity if cap == self.capacity else n % self.capacity

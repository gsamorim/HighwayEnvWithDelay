"""PPO via stable-baselines3 (Schulman et al., 2017), wired in as a second
agent architecture alongside our from-scratch SAC -- per explicit user
request to use SB3's own PPO rather than reimplement it by hand.

Why this is a separate module instead of just another `Agent` subclass used
the same way as `SACAgent`: PPO is on-policy. It collects a whole rollout of
`n_steps` transitions (with values/log-probs captured at sampling time),
then runs several epochs of minibatch gradient updates over that rollout,
and throws the rollout away -- it never reuses old transitions the way an
off-policy replay buffer does. `rl_platform/training/trainer.py` calls
`agent.observe()` once per env step and `agent.update()` once per env step
too (see its own docstring, which anticipated exactly this): that shape
fits SAC, but does not fit PPO's batch-rollout update without basically
reimplementing SB3's rollout buffer and GAE computation by hand inside
`observe()`/`update()`. Since the user asked for PPO "conforme o SB3" (SB3's
own way), real PPO *training* goes through SB3's own `model.learn()` loop
instead (see `scripts/train.py::train_sb3_ppo`), which owns rollout
collection, GAE(lambda) advantage estimation, and the clipped surrogate
objective internally -- none of that is reimplemented here.

`SB3PPOAgent` below exists only so a trained PPO policy is selectable via
`--agent ppo` in `scripts/record.py` (and any other script that goes through
`rl_platform.agents.make_agent`/`Agent.act`/`Agent.save`/`Agent.load`) the
same way a SAC checkpoint is. Its `observe()`/`update()` are deliberate
no-ops: calling them (e.g. if `train.py`'s generic `Trainer` were ever
pointed at a `--agent ppo` by mistake) does nothing instead of silently
training wrong. `ready_to_update()` returns False for the same reason, so
such a misuse at least collects no false signal of progress.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a hard dependency of highway-env too
    import gym  # type: ignore

from .base import Agent


def _import_sb3_ppo():
    """Import stable-baselines3's PPO lazily (only when an SB3PPOAgent is
    actually constructed), not at module import time. This keeps `--agent
    sac`/`--agent random` working in an environment that doesn't have
    `stable-baselines3` installed -- `rl_platform/agents/__init__.py`
    imports this module unconditionally to register "ppo" in the registry,
    so a hard top-level `import stable_baselines3` here would break every
    other agent too. Only building/loading a "ppo" agent needs the real
    dependency; this raises a clear, actionable error at that point instead
    of a confusing ImportError deep in someone else's agent."""
    try:
        from stable_baselines3 import PPO as SB3PPO
    except ImportError as e:
        raise ImportError(
            "The 'ppo' agent needs stable-baselines3, which isn't installed "
            "in this Python environment. Install it with:\n"
            "    pip install stable-baselines3\n"
            "(it's listed in requirements.txt). --agent sac/random don't "
            "need it and are unaffected."
        ) from e
    return SB3PPO

# SB3's own PPO constructor kwargs we expose via --ppo-* CLI flags / the
# .jsonc config (see cli_common.py). Anything NOT in this set that a caller
# passes through (e.g. SAC-only kwargs like buffer_size/updates_per_step/
# learning_starts, which train.py/record.py pass unconditionally today) is
# silently dropped in registry.py's `_make_ppo` factory rather than raised,
# so the same CLI works for either --agent sac or --agent ppo.
SB3_PPO_KWARGS = {
    "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
    "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
}


class _DummySpacesEnv(gym.Env):
    """A `gym.Env` that is never actually stepped for real -- it exists only
    to hand SB3's `PPO.__init__` a `gym.Env` with the right
    observation_space/action_space shapes, so `SB3PPOAgent` can construct a
    (randomly-initialized) policy network from just `obs_dim`/`act_dim`
    (mirroring how `SACAgent`/`RandomAgent` are constructed in
    `registry.py`), without needing a real HighwayEnv instance at agent-
    construction time. Real training always goes through a real env (see
    `scripts/train.py::train_sb3_ppo`); this is only reached when
    `record.py` builds an *untrained* `--agent ppo` for baseline footage, or
    transiently before `SB3PPOAgent.load()` replaces the model wholesale."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


class SB3PPOAgent(Agent):
    """`Agent`-interface wrapper around a stable-baselines3 `PPO` model, for
    acting/saving/loading only (see module docstring for why real training
    does not go through this class)."""

    name = "ppo"

    def __init__(self, obs_dim: int, act_dim: int, seed: Optional[int] = None, **ppo_kwargs: Any):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seed = seed
        self._ppo_kwargs = {k: v for k, v in ppo_kwargs.items() if k in SB3_PPO_KWARGS and v is not None}
        self.model: Optional[Any] = None  # stable_baselines3.PPO, imported lazily (see _import_sb3_ppo)

    def _ensure_model(self):
        if self.model is None:
            SB3PPO = _import_sb3_ppo()
            dummy_env = _DummySpacesEnv(self.obs_dim, self.act_dim)
            self.model = SB3PPO("MlpPolicy", dummy_env, seed=self.seed, verbose=0, **self._ppo_kwargs)
        return self.model

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        model = self._ensure_model()
        action, _state = model.predict(observation, deterministic=deterministic)
        return action

    def observe(self, obs, action, reward, next_obs, done) -> None:
        return  # see module docstring

    def update(self) -> dict[str, float]:
        return {}  # see module docstring

    def save(self, path: str) -> None:
        self._ensure_model().save(path)

    def load(self, path: str) -> None:
        SB3PPO = _import_sb3_ppo()
        self.model = SB3PPO.load(path)

    def ready_to_update(self) -> bool:
        return False  # see module docstring

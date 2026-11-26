"""SAC via stable-baselines3 (Haarnoja et al., 2018/2019), wired in as a
*third* agent architecture, alongside our from-scratch "haarnoja" SAC
(`rl_platform/agents/sac.py`) and SB3's own PPO
(`rl_platform/agents/ppo_sb3.py`) -- per explicit user request for a
SAC-via-SB3 option to sit next to the from-scratch one, the same way
PPO-via-SB3 sits next to it.

Why this one is NOT routed through `rl_platform/training/trainer.py`'s
generic off-policy loop either, even though SAC *is* off-policy (unlike
PPO) and would fit that loop's per-step observe()/update() shape in
principle: SB3's `SAC` does not expose its replay buffer or its gradient
step as public per-env-step methods meant to be driven externally -- its
own `model.learn()` already owns the replay buffer, the `train_freq`/
`gradient_steps` cadence, and the entropy-coefficient auto-tuning
internally. Reimplementing that cadence by hand around `observe()`/
`update()` would mean reaching into SB3 internals instead of using its
public API as intended. So, like PPO, real training for this agent goes
through SB3's own `.learn()` loop (see `scripts/train.py::train_sb3`, which
also drives PPO -- the two share one implementation, since SB3's `.learn()`
API is uniform across on-policy and off-policy algorithms).

`SB3SACAgent` below exists only so a trained SB3-SAC policy is selectable
via `--agent sac_sb3` in `scripts/record.py` (and any other script that goes
through `rl_platform.agents.make_agent`/`Agent.act`/`Agent.save`/
`Agent.load`) the same way a checkpoint of either other agent is. Its
`observe()`/`update()` are deliberate no-ops for the same reason as
`SB3PPOAgent`'s: calling them does nothing instead of silently training
wrong. `ready_to_update()` returns False for the same reason.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a hard dependency of highway-env too
    import gym  # type: ignore

from .base import Agent


def _import_sb3_sac():
    """Import stable-baselines3's SAC lazily -- see
    `ppo_sb3.py::_import_sb3_ppo` for why this can't be a top-level import:
    `rl_platform/agents/__init__.py` imports this module unconditionally to
    register "sac_sb3", so a hard dependency here would break `--agent
    sac`/`--agent random` in any environment without stable-baselines3
    installed."""
    try:
        from stable_baselines3 import SAC as SB3SAC
    except ImportError as e:
        raise ImportError(
            "The 'sac_sb3' agent needs stable-baselines3, which isn't "
            "installed in this Python environment. Install it with:\n"
            "    pip install stable-baselines3\n"
            "(it's listed in requirements.txt). --agent sac/random don't "
            "need it and are unaffected."
        ) from e
    return SB3SAC

# SB3's own SAC constructor kwargs we expose via --sac-sb3-* CLI flags / the
# .jsonc config (see cli_common.py / scripts/train.py). Anything NOT in this
# set that a caller passes through unconditionally (e.g. PPO-only kwargs
# like n_steps/n_epochs/clip_range, which train.py passes regardless of
# --agent) is silently dropped in registry.py's `_make_sac_sb3` factory
# rather than raised, so the same CLI works for --agent sac / ppo / sac_sb3
# alike. "buffer_size"/"batch_size"/"learning_starts" are deliberately
# shared names with our from-scratch SAC's own kwargs (same meaning in both
# -- replay buffer capacity, minibatch size, random-exploration warmup
# steps), so those three flags work unchanged across both SAC variants.
SB3_SAC_KWARGS = {
    "buffer_size", "batch_size", "learning_starts", "learning_rate", "gamma",
    "tau", "train_freq", "gradient_steps", "ent_coef", "target_update_interval",
}


class _DummySpacesEnv(gym.Env):
    """A `gym.Env` that is never actually stepped for real -- see the
    identical class in `ppo_sb3.py` for the full rationale (same reason:
    SB3's `SAC.__init__` needs a `gym.Env` with the right spaces to build
    its policy/Q networks, and `make_agent` is called with just
    `obs_dim`/`act_dim` scalars, not a real env instance)."""

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


def coerce_ent_coef(value: Any) -> Any:
    """SB3's `ent_coef` accepts either the string "auto" (optionally
    "auto_<init_value>") or a fixed float. CLI flags only give us strings;
    pass "auto"/"auto_X" through untouched and parse anything else as a
    float so `--sac-sb3-ent-coef 0.1` and `--sac-sb3-ent-coef auto` both
    work."""
    if isinstance(value, str) and value.startswith("auto"):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class SB3SACAgent(Agent):
    """`Agent`-interface wrapper around a stable-baselines3 `SAC` model, for
    acting/saving/loading only (see module docstring for why real training
    does not go through this class)."""

    name = "sac_sb3"

    def __init__(self, obs_dim: int, act_dim: int, seed: Optional[int] = None, **sac_kwargs: Any):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seed = seed
        kwargs = {k: v for k, v in sac_kwargs.items() if k in SB3_SAC_KWARGS and v is not None}
        if "ent_coef" in kwargs:
            kwargs["ent_coef"] = coerce_ent_coef(kwargs["ent_coef"])
        self._sac_kwargs = kwargs
        self.model: Optional[Any] = None  # stable_baselines3.SAC, imported lazily (see _import_sb3_sac)

    def _ensure_model(self):
        if self.model is None:
            SB3SAC = _import_sb3_sac()
            dummy_env = _DummySpacesEnv(self.obs_dim, self.act_dim)
            self.model = SB3SAC("MlpPolicy", dummy_env, seed=self.seed, verbose=0, **self._sac_kwargs)
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
        SB3SAC = _import_sb3_sac()
        self.model = SB3SAC.load(path)

    def ready_to_update(self) -> bool:
        return False  # see module docstring

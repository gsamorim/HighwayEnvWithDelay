"""Agent registry -- the extension point for adding new RL architectures
(TD3, DreamerV3, ...) to compete with SAC on the same environments, without
changing the evaluation scripts (record.py).

Four real, built-in entries exist today:
  - "random": no learning, a sanity-check baseline.
  - "sac": our from-scratch, dependency-light SAC (Haarnoja et al.,
    2018/2019) -- see `rl_platform/agents/sac.py`'s own docstring.
  - "ppo": stable-baselines3's own PPO -- see `ppo_sb3.py`.
  - "sac_sb3": stable-baselines3's own SAC, as a second implementation of
    the same algorithm as "sac" but via SB3 instead of from-scratch -- see
    `sac_sb3.py`.
Both SB3-backed agents (ppo, sac_sb3) do NOT use
`rl_platform/training/trainer.py`'s generic loop for actual training (see
that module's and each agent module's own docstrings for why); training for
either goes through `scripts/train.py::train_sb3` instead, while this
registry/`make_agent` is still how `scripts/record.py` constructs any of the
four to act/load a checkpoint.

The augmented-state ("estado aumentado", i.e. `enable_delay_wrapper`)
variant of any of these three real agents is NOT a different registry entry
-- it's an env-level toggle, orthogonal to which agent is used (see
`RacetrackEnvConfig.enable_delay_wrapper` / `--enable-delay-wrapper`): the
wrapper changes `obs_dim`, and every agent here is constructed generically
from `obs_dim`/`act_dim` probed from the real env, so it adapts automatically.

To add a new off-policy agent (fits the generic Trainer as-is):
    from rl_platform.agents.registry import register_agent

    def make_td3_agent(obs_dim, act_dim, **kwargs):
        return TD3Agent(TD3Config(obs_dim=obs_dim, act_dim=act_dim, **kwargs))

    register_agent("td3", make_td3_agent)
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Agent
from .sac import SACAgent, SACConfig
from .random_agent import RandomAgent
from .ppo_sb3 import SB3PPOAgent
from .sac_sb3 import SB3SACAgent

AgentFactory = Callable[..., Agent]

_AGENT_REGISTRY: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory, overwrite: bool = False) -> None:
    if not overwrite and name in _AGENT_REGISTRY:
        raise ValueError(f"Agent '{name}' is already registered.")
    _AGENT_REGISTRY[name] = factory


def available_agents() -> list[str]:
    return sorted(_AGENT_REGISTRY)


def make_agent(name: str, obs_dim: int, act_dim: int, **kwargs: Any) -> Agent:
    if name not in _AGENT_REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Available: {available_agents()}")
    return _AGENT_REGISTRY[name](obs_dim=obs_dim, act_dim=act_dim, **kwargs)


# --- built-in registrations -------------------------------------------------
def _make_sac(obs_dim: int, act_dim: int, **kwargs: Any) -> SACAgent:
    return SACAgent(SACConfig(obs_dim=obs_dim, act_dim=act_dim, **kwargs))


def _make_random(obs_dim: int, act_dim: int, **kwargs: Any) -> RandomAgent:
    return RandomAgent(act_dim=act_dim, seed=kwargs.get("seed"))


def _make_ppo(obs_dim: int, act_dim: int, **kwargs: Any) -> SB3PPOAgent:
    # train.py/record.py pass SAC-oriented kwargs (buffer_size,
    # updates_per_step, learning_starts, batch_size) unconditionally; drop
    # the ones SB3PPOAgent doesn't understand instead of erroring, so the
    # same CLI surface works for --agent sac and --agent ppo alike.
    # SB3PPOAgent itself also filters to SB3_PPO_KWARGS, this is just belt
    # and suspenders for kwargs that would otherwise clash (none currently
    # do, batch_size is shared by both, kept as-is).
    kwargs.pop("buffer_size", None)
    kwargs.pop("updates_per_step", None)
    kwargs.pop("learning_starts", None)
    return SB3PPOAgent(obs_dim=obs_dim, act_dim=act_dim, **kwargs)


def _make_sac_sb3(obs_dim: int, act_dim: int, **kwargs: Any) -> SB3SACAgent:
    # train.py/record.py pass PPO-oriented kwargs (n_steps/n_epochs/
    # clip_range/...) and our from-scratch SAC's updates_per_step
    # unconditionally; drop the ones SB3SACAgent doesn't understand instead
    # of erroring. buffer_size/batch_size/learning_starts are KEPT (unlike
    # in _make_ppo) because they're shared, same-meaning concepts between
    # our from-scratch SAC and SB3's SAC -- SB3SACAgent itself also filters
    # to SB3_SAC_KWARGS, this is just belt and suspenders.
    kwargs.pop("updates_per_step", None)
    return SB3SACAgent(obs_dim=obs_dim, act_dim=act_dim, **kwargs)


register_agent("sac", _make_sac)
register_agent("random", _make_random)
register_agent("ppo", _make_ppo)
register_agent("sac_sb3", _make_sac_sb3)

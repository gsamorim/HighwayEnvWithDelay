"""Agent registry -- the extension point for adding new RL architectures
(PPO, TD3, DreamerV3, ...) to compete with SAC on the same environments,
without changing the training loop or evaluation scripts.

To add a new agent:
    from rl_platform.agents.registry import register_agent

    def make_ppo_agent(obs_dim, act_dim, **kwargs):
        return PPOAgent(PPOConfig(obs_dim=obs_dim, act_dim=act_dim, **kwargs))

    register_agent("ppo", make_ppo_agent)
"""
from __future__ import annotations

from typing import Any, Callable

from .base import Agent
from .sac import SACAgent, SACConfig
from .random_agent import RandomAgent

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


register_agent("sac", _make_sac)
register_agent("random", _make_random)

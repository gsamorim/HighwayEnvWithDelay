"""Environment registry.

This is the extension point for adding new environments later (e.g. other
HighwayEnv scenarios, or entirely different simulators) without touching the
training/agent code. Every entry maps a short name to a zero/kwargs factory
function returning a `gymnasium.Env`.

To add a new environment:
    from rl_platform.envs.registry import register_env

    def make_intersection_env(**kwargs):
        ...
        return env

    register_env("intersection", make_intersection_env)

and it becomes usable everywhere via `make_env("intersection", ...)`.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym

from .racetrack import make_racetrack_env

EnvFactory = Callable[..., gym.Env]

_ENV_REGISTRY: dict[str, EnvFactory] = {}


def register_env(name: str, factory: EnvFactory, overwrite: bool = False) -> None:
    if not overwrite and name in _ENV_REGISTRY:
        raise ValueError(f"Environment '{name}' is already registered.")
    _ENV_REGISTRY[name] = factory


def available_envs() -> list[str]:
    return sorted(_ENV_REGISTRY)


def make_env(name: str, **kwargs: Any) -> gym.Env:
    if name not in _ENV_REGISTRY:
        raise KeyError(
            f"Unknown environment '{name}'. Available: {available_envs()}"
        )
    return _ENV_REGISTRY[name](**kwargs)


# --- built-in registrations -------------------------------------------------
register_env("racetrack", make_racetrack_env)

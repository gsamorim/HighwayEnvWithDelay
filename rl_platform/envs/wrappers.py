"""Generic observation wrappers usable by any environment in the registry."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenObservation(gym.ObservationWrapper):
    """Flattens an arbitrary-shaped Box observation into a 1D vector.

    HighwayEnv's Kinematics/OccupancyGrid observations are returned as
    (N, F) or (C, H, W) arrays. Most simple MLP-based agents (like the SAC
    implementation here) expect a flat vector, so this wrapper reshapes the
    observation space accordingly. Kept separate from the agent so that a
    future CNN-based agent can instead consume the unflattened observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        flat_dim = int(np.prod(env.observation_space.shape))
        low = env.observation_space.low.reshape(-1)
        high = env.observation_space.high.reshape(-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)


class ForceRenderEnabled(gym.Wrapper):
    """Work around a HighwayEnv quirk: its `EnvViewer` silently disables all
    drawing (leaving `render()` return solid-black frames) whenever
    `SDL_VIDEODRIVER=dummy` is set -- see
    `highway_env.envs.common.graphics.EnvViewer.__init__`. That env var is
    exactly what a display-less cloud sandbox needs to set so pygame doesn't
    fail to open a display, so without this wrapper `render_mode="rgb_array"`
    silently produces empty video/GIF frames when running headless.

    We let the viewer lazily construct itself as usual, then flip its
    `enabled` flag back on so it actually draws the road/vehicles onto its
    offscreen surface. Harmless (and a no-op) when a real display driver is
    in use, e.g. when `scripts/watch.py` is run locally.
    """

    def render(self):
        viewer = getattr(self.unwrapped, "viewer", None)
        if viewer is None:
            # First call: let HighwayEnv construct (and possibly disable) it.
            self.env.render()
            viewer = getattr(self.unwrapped, "viewer", None)
        if viewer is not None and not viewer.enabled:
            viewer.enabled = True
        return self.env.render()


class RewardScaler(gym.RewardWrapper):
    """Optionally rescales the environment reward, useful for stabilizing SAC's
    entropy-regularized objective across environments with very different
    reward magnitudes."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class HighSpeedRewardWrapper(gym.Wrapper):
    """Adds an explicit speed incentive to `racetrack-v0`.

    This fixes a real gap in HighwayEnv's `RacetrackEnv._rewards`: unlike
    `highway-v0`/`intersection-v0`, it has no speed term at all -- only
    `lane_centering_reward`, `action_reward` (which *penalizes* the norm of
    the action, i.e. penalizes accelerating/steering at all), `collision_reward`,
    and the `on_road_reward` gate. A `high_speed_reward` key in the env config
    is silently ignored because `RacetrackEnv` never looks it up. Without this
    wrapper, an SAC agent trained here has no incentive to move quickly at
    all and reliably converges to crawling around the track at low speed --
    which we observed empirically before adding this fix (see RESULTS.md).

    Adds `coef * clip(speed / speed_limit, 0, 1.2)` to the reward each step,
    gated to 0 off-road (consistent with the base env's own on-road gating)
    so the agent can't farm the speed bonus by cutting corners off-track.
    """

    def __init__(self, env: gym.Env, coef: float, speed_limit: float):
        super().__init__(env)
        self.coef = coef
        self.speed_limit = speed_limit

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        on_road = info.get("rewards", {}).get("on_road_reward", True)
        speed = info.get("speed", 0.0)
        speed_frac = float(np.clip(speed / self.speed_limit, 0.0, 1.2)) if on_road else 0.0
        bonus = self.coef * speed_frac
        info = dict(info)
        info["speed_bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info

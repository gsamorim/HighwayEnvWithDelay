"""Factory + config presets for HighwayEnv's `racetrack-v0`.

`racetrack-v0` is a closed-loop track with an ego vehicle and a configurable
number of IDM-driven traffic vehicles. The two continuous control variables
are acceleration (longitudinal) and steering (lateral); by default HighwayEnv
only exposes steering (with a discrete target-speed controller for the
throttle), so we explicitly enable both here per the project brief.

The most interesting *environment* knob for this project is traffic density,
controlled by HighwayEnv's `other_vehicles` config key (an upper bound on the
number of randomly spawned IDM vehicles per episode -- see
`highway_env.envs.racetrack_env.RacetrackEnv._make_vehicles`). We expose this
as a first-class `traffic_density` parameter so experiments can sweep it
without knowing the underlying HighwayEnv config key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-env gym environments)

from .wrappers import FlattenObservation, RewardScaler, ForceRenderEnabled, HighSpeedRewardWrapper

# Named traffic presets -> number of other (IDM-controlled) vehicles.
# `other_vehicles` is an *upper bound*: HighwayEnv samples a random integer
# in [0, other_vehicles) each episode, plus one guaranteed lead vehicle when
# other_vehicles > 0. So "light" traffic still has episodes with 0-1 cars.
TRAFFIC_PRESETS: dict[str, int] = {
    "none": 0,
    "light": 2,
    "medium": 5,
    "heavy": 10,
    "jam": 16,
}


@dataclass
class RacetrackEnvConfig:
    """User-facing knobs for building a racetrack environment.

    Anything not listed here can still be supplied via `extra_config`, which
    is merged on top (and wins) over everything computed below.
    """

    traffic_density: str | int = "medium"      # preset name or explicit other_vehicles count
    controlled_vehicles: int = 1
    duration: int = 60                          # seconds of simulated episode time
    observation_type: str = "Kinematics"        # "Kinematics" (vector) or "OccupancyGrid" (image)
    vehicles_count_observed: int = 6            # nearby vehicles included in Kinematics obs
    action_longitudinal: bool = True            # enable acceleration control
    action_lateral: bool = True                 # enable steering control
    simulation_frequency: int = 15
    policy_frequency: int = 5
    speed_limit: float = 10.0
    collision_reward: float = -5.0
    action_reward: float = -0.1
    lane_centering_cost: float = 4.0
    lane_centering_reward: float = 1.0
    # NOTE: RacetrackEnv's own `_rewards()` has no speed term at all (unlike
    # highway-v0/intersection-v0), so a "high_speed_reward" *env config* key
    # is silently ignored. This field instead drives `HighSpeedRewardWrapper`,
    # a reward wrapper applied on top of the env -- see wrappers.py.
    high_speed_reward: float = 0.4
    reward_scale: float = 1.0
    extra_config: dict[str, Any] = field(default_factory=dict)

    def resolve_traffic(self) -> int:
        if isinstance(self.traffic_density, str):
            try:
                return TRAFFIC_PRESETS[self.traffic_density]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown traffic preset '{self.traffic_density}'. "
                    f"Choose one of {list(TRAFFIC_PRESETS)} or pass an int."
                ) from exc
        return int(self.traffic_density)

    def to_highway_config(self) -> dict[str, Any]:
        obs_cfg: dict[str, Any]
        if self.observation_type == "Kinematics":
            obs_cfg = {
                "type": "Kinematics",
                "vehicles_count": self.vehicles_count_observed,
                "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off"],
                "absolute": False,
                "normalize": True,
                "see_behind": True,
            }
        elif self.observation_type == "OccupancyGrid":
            obs_cfg = {
                "type": "OccupancyGrid",
                "features": ["presence", "on_road"],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True,
            }
        else:
            raise ValueError(f"Unsupported observation_type '{self.observation_type}'")

        cfg: dict[str, Any] = {
            "observation": obs_cfg,
            "action": {
                "type": "ContinuousAction",
                "longitudinal": self.action_longitudinal,
                "lateral": self.action_lateral,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.5235987755982988, 0.5235987755982988],  # +-30deg
            },
            "simulation_frequency": self.simulation_frequency,
            "policy_frequency": self.policy_frequency,
            "duration": self.duration,
            "collision_reward": self.collision_reward,
            "action_reward": self.action_reward,
            "lane_centering_cost": self.lane_centering_cost,
            "lane_centering_reward": self.lane_centering_reward,
            "controlled_vehicles": self.controlled_vehicles,
            "other_vehicles": self.resolve_traffic(),
            "speed_limit": self.speed_limit,
            "terminate_off_road": True,
            "offscreen_rendering": True,
        }
        # extra_config always wins (deep-ish merge for the one level we care about)
        for k, v in self.extra_config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        return cfg


def make_racetrack_env(
    config: Optional[RacetrackEnvConfig] = None,
    render_mode: Optional[str] = None,
    flatten_obs: bool = True,
    **overrides: Any,
) -> gym.Env:
    """Build a ready-to-use `racetrack-v0` Gymnasium environment.

    Parameters
    ----------
    config: a `RacetrackEnvConfig`, or None to build one from `overrides`.
    render_mode: None, "rgb_array", or "human" (pygame window; requires a
        real display -- see `rl_platform.visualization`).
    flatten_obs: wrap the observation into a flat vector (needed by the
        MLP-based SAC agent shipped in this repo). Set False if you plan to
        feed a CNN a raw OccupancyGrid/image observation.
    overrides: convenience kwargs forwarded to `RacetrackEnvConfig` when
        `config` is not given, e.g. `make_racetrack_env(traffic_density="heavy")`.
    """
    if config is None:
        config = RacetrackEnvConfig(**overrides)

    env = gym.make(
        "racetrack-v0",
        render_mode=render_mode,
        config=config.to_highway_config(),
    )
    if render_mode is not None:
        env = ForceRenderEnabled(env)
    if config.high_speed_reward != 0.0:
        env = HighSpeedRewardWrapper(env, coef=config.high_speed_reward, speed_limit=config.speed_limit)
    if config.reward_scale != 1.0:
        env = RewardScaler(env, scale=config.reward_scale)
    if flatten_obs:
        env = FlattenObservation(env)
    return env

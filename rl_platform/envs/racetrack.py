"""Factory + config presets for HighwayEnv's racetrack family.

`racetrack-v0`/`racetrack-v1` is a closed-loop track, and
`racetrack-oval-v0`/`racetrack-oval-v1` is an oval variant with a
configurable straight-segment length and two optional obstacle presets; both
use an ego vehicle and a configurable number of IDM-driven traffic vehicles.
The two continuous control variables are acceleration (longitudinal) and
steering (lateral); by default HighwayEnv only exposes steering (with a
discrete target-speed controller for the throttle), so we explicitly enable
both here per the project brief.

The most interesting *environment* knob for this project is traffic density,
controlled by HighwayEnv's `other_vehicles` config key (an upper bound on the
number of randomly spawned IDM vehicles per episode -- see
`highway_env.envs.racetrack_env.RacetrackEnv._make_vehicles`). We expose this
as a first-class `traffic_density` parameter so experiments can sweep it
without knowing the underlying HighwayEnv config key.

Everything in this module composes on top of the native HighwayEnv classes
(`RacetrackEnv`/`RacetrackEnvOval`) via their own `config=` dict and via
`gym.Wrapper`s in `wrappers.py` -- no native class is subclassed or patched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-env gym environments)

from .wrappers import (
    FlattenObservation,
    RewardScaler,
    ForceRenderEnabled,
    HighSpeedRewardWrapper,
    TrafficSpeedWrapper,
    DelayedActionMDPWrapper,
    SafeSpawnWrapper,
)

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

# Native gym ids this factory is known to work with. Not enforced (HighwayEnv
# may register more over time), just documents what's been exercised.
KNOWN_ENV_IDS = (
    "racetrack-v0",
    "racetrack-v1",
    "racetrack-large-v0",
    "racetrack-large-v1",
    "racetrack-oval-v0",
    "racetrack-oval-v1",
)

_OVAL_OBSTACLE_PRESETS = ("block_lane", "force_decision")


@dataclass
class RacetrackEnvConfig:
    """User-facing knobs for building a racetrack environment.

    Anything not listed here can still be supplied via `extra_config`, which
    is merged on top (and wins) over everything computed below.
    """

    env_id: str = "racetrack-v0"                 # native gym id (see KNOWN_ENV_IDS)
    traffic_density: str | int = "medium"        # preset name or explicit other_vehicles count
    controlled_vehicles: int = 1
    duration: int = 60                           # seconds of simulated episode time
    observation_type: str = "Kinematics"         # "Kinematics" (vector) or "OccupancyGrid" (image)
    vehicles_count_observed: int = 6             # nearby vehicles included in Kinematics obs
    action_longitudinal: bool = True             # enable acceleration control
    action_lateral: bool = True                  # enable steering control
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
    # "linear" (default, original behavior): coef * clip(speed/speed_limit, 0, 1.2).
    # "tanh": coef * tanh((speed - v0_frac*speed_limit) / (steepness_frac*speed_limit)) --
    # small bounded NEGATIVE bonus standing still/reversing, steep rise around
    # v0, diminishing returns (saturating, not unbounded) at high speed.
    speed_reward_shape: str = "linear"
    speed_reward_v0_frac: float = 0.1
    speed_reward_steepness_frac: float = 0.3
    reward_scale: float = 1.0

    # --- Oval track variant (racetrack-oval-* only) -------------------------
    # Native `RacetrackEnvOval` config keys. Setting any of these when
    # `env_id` is not an oval variant is a config error (raised below, not
    # silently ignored), since they have no effect on racetrack-v0/v1/large.
    oval_straight_length_m: Optional[int] = None   # native `length`; None = native default (~100)
    oval_no_lanes: Optional[int] = None            # native `no_lanes`; None = native default (3)
    oval_obstacle_preset: Optional[str] = None     # None | "block_lane" | "force_decision"

    # --- Traffic speed (ours; not a native highway-env key) -----------------
    # Both or neither must be set. None/None disables TrafficSpeedWrapper
    # entirely (traffic keeps the track's native spawn speed).
    traffic_speed_min: Optional[float] = None
    traffic_speed_max: Optional[float] = None

    # --- Delay / RDMDP-style augmented state (ours) -------------------------
    enable_delay_wrapper: bool = False     # False = baseline, no extra obs dims at all
    action_delay_min_steps: int = 0
    action_delay_max_steps: int = 0
    delay_known_to_agent: bool = False
    observation_delay_steps: int = 0       # simple FIFO delay, no reordering (see wrappers.py)
    delay_seed: Optional[int] = None

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

    def _is_oval(self) -> bool:
        return "oval" in self.env_id

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

        oval_fields_set = (
            self.oval_straight_length_m is not None
            or self.oval_no_lanes is not None
            or self.oval_obstacle_preset is not None
        )
        if self._is_oval():
            if self.oval_straight_length_m is not None:
                cfg["length"] = self.oval_straight_length_m
            if self.oval_no_lanes is not None:
                cfg["no_lanes"] = self.oval_no_lanes
            if self.oval_obstacle_preset is not None:
                if self.oval_obstacle_preset not in _OVAL_OBSTACLE_PRESETS:
                    raise ValueError(
                        f"Unknown oval_obstacle_preset '{self.oval_obstacle_preset}'. "
                        f"Choose one of {_OVAL_OBSTACLE_PRESETS} or None."
                    )
                cfg[self.oval_obstacle_preset] = True
        elif oval_fields_set:
            raise ValueError(
                f"oval_straight_length_m/oval_no_lanes/oval_obstacle_preset require an "
                f"oval env_id (got '{self.env_id}'); these are native RacetrackEnvOval "
                f"config keys with no effect on non-oval environments."
            )

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
    """Build a ready-to-use racetrack Gymnasium environment.

    Parameters
    ----------
    config: a `RacetrackEnvConfig`, or None to build one from `overrides`.
    render_mode: None, "rgb_array", or "human" (pygame window; requires a
        real display -- see `rl_platform.visualization`).
    flatten_obs: wrap the observation into a flat vector (needed by the
        MLP-based SAC agent shipped in this repo). Set False if you plan to
        feed a CNN a raw OccupancyGrid/image observation. Must be True if
        `config.enable_delay_wrapper` is True (the delay wrapper requires an
        already-flat 1D observation to concatenate onto).
    overrides: convenience kwargs forwarded to `RacetrackEnvConfig` when
        `config` is not given, e.g. `make_racetrack_env(traffic_density="heavy")`.
    """
    if config is None:
        config = RacetrackEnvConfig(**overrides)

    if config.enable_delay_wrapper and not flatten_obs:
        raise ValueError(
            "enable_delay_wrapper=True requires flatten_obs=True: "
            "DelayedActionMDPWrapper concatenates onto an already-flat observation."
        )

    env = gym.make(
        config.env_id,
        render_mode=render_mode,
        config=config.to_highway_config(),
    )
    # Always first: works around a real HighwayEnv bug where the controlled
    # vehicle can spawn off-lane on a short oval straight (see wrappers.py
    # docstring). No-op whenever the native spawn was already valid.
    env = SafeSpawnWrapper(env)
    if render_mode is not None:
        env = ForceRenderEnabled(env)
    if config.traffic_speed_min is not None or config.traffic_speed_max is not None:
        if config.traffic_speed_min is None or config.traffic_speed_max is None:
            raise ValueError("traffic_speed_min and traffic_speed_max must be set together (or both left None).")
        env = TrafficSpeedWrapper(env, speed_min=config.traffic_speed_min, speed_max=config.traffic_speed_max)
    if config.high_speed_reward != 0.0:
        env = HighSpeedRewardWrapper(
            env,
            coef=config.high_speed_reward,
            speed_limit=config.speed_limit,
            shape=config.speed_reward_shape,
            v0_frac=config.speed_reward_v0_frac,
            steepness_frac=config.speed_reward_steepness_frac,
        )
    if config.reward_scale != 1.0:
        env = RewardScaler(env, scale=config.reward_scale)
    if flatten_obs:
        env = FlattenObservation(env)
    if config.enable_delay_wrapper:
        env = DelayedActionMDPWrapper(
            env,
            min_delay_steps=config.action_delay_min_steps,
            max_delay_steps=config.action_delay_max_steps,
            delay_known_to_agent=config.delay_known_to_agent,
            obs_delay_steps=config.observation_delay_steps,
            seed=config.delay_seed,
        )
    return env


def make_racetrack_oval_env(
    config: Optional[RacetrackEnvConfig] = None,
    render_mode: Optional[str] = None,
    flatten_obs: bool = True,
    **overrides: Any,
) -> gym.Env:
    """Same as `make_racetrack_env`, defaulting `env_id` to the oval track
    variant (`racetrack-oval-v0`) when `config` is not given. Registered
    separately as `"racetrack_oval"` purely as a convenience default --
    every oval knob is fully reachable (and overridable) through
    `make_racetrack_env`/`RacetrackEnvConfig` directly, e.g. pass
    `env_id="racetrack-oval-v1"` to either factory.
    """
    if config is None:
        overrides.setdefault("env_id", "racetrack-oval-v0")
        config = RacetrackEnvConfig(**overrides)
    return make_racetrack_env(config=config, render_mode=render_mode, flatten_obs=flatten_obs)

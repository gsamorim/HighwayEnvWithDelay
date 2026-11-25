"""Tests for the oval-track knobs, traffic speed wrapper, and the delay-robust
(RDMDP-style) augmented-state wrapper added on top of the native racetrack
factory. See rl_platform/envs/wrappers.py for the full rule set (R1-R7)."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pytest

from rl_platform.envs import make_env, available_envs, MAX_DELAY_STEPS, RacetrackEnvConfig


def test_racetrack_oval_registered():
    assert "racetrack_oval" in available_envs()


def test_oval_straight_length_maps_to_native_length_key():
    env = make_env("racetrack_oval", oval_straight_length_m=30)
    assert env.unwrapped.config["length"] == 30
    env.close()


def test_oval_obstacle_preset_maps_to_native_bool_key():
    env = make_env("racetrack_oval", oval_obstacle_preset="force_decision")
    assert env.unwrapped.config["force_decision"] is True
    env.close()


def test_oval_only_knobs_reject_non_oval_env_id():
    with pytest.raises(ValueError):
        make_env("racetrack", oval_straight_length_m=30)


def test_traffic_speed_wrapper_bounds_other_vehicles_speed():
    env = make_env("racetrack", traffic_density="medium", traffic_speed_min=5.0, traffic_speed_max=8.0)
    env.reset(seed=0)
    controlled_ids = {id(v) for v in env.unwrapped.controlled_vehicles}
    speeds = [v.speed for v in env.unwrapped.road.vehicles if id(v) not in controlled_ids]
    assert all(5.0 - 1e-6 <= s <= 8.0 + 1e-6 for s in speeds)
    env.close()


def test_traffic_speed_requires_both_bounds():
    with pytest.raises(ValueError):
        make_env("racetrack", traffic_speed_min=5.0)


@pytest.mark.parametrize("min_d,max_d", [(0, 0), (1, 1), (2, 5), (0, 6)])
def test_delay_wrapper_shape_is_invariant_to_delay_range(min_d, max_d):
    """The augmented observation must have the SAME shape whatever the
    configured delay range is (train with a narrow range, eval with a wide
    one), including the delay=0 baseline -- it is never bypassed."""
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True,
        action_delay_min_steps=min_d, action_delay_max_steps=max_d,
        delay_known_to_agent=False,
    )
    obs, info = env.reset(seed=0)
    expected_dim = obs.shape[0]
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape[0] == expected_dim
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_delay_wrapper_obs_dim_matches_expected_layout():
    base_env = make_env("racetrack", traffic_density="none")
    base_dim = base_env.observation_space.shape[0]
    base_env.close()
    act_dim = 2
    expected_extra = MAX_DELAY_STEPS * act_dim * 2 + MAX_DELAY_STEPS + act_dim
    env = make_env("racetrack", enable_delay_wrapper=True, action_delay_min_steps=0, action_delay_max_steps=3)
    obs, info = env.reset(seed=0)
    assert obs.shape[0] == base_dim + expected_extra
    env.close()


def test_delay_wrapper_requires_flat_observation():
    with pytest.raises(ValueError):
        make_env("racetrack", enable_delay_wrapper=True, flatten_obs=False)


def test_delay_known_to_agent_false_masks_remaining_time_but_keeps_shape():
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True, action_delay_min_steps=2, action_delay_max_steps=2,
        delay_known_to_agent=False,
    )
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["delay_debug"]["pending_remaining_time"].sum() > 0  # true value still in debug info
    act_dim = 2
    rt_start = obs.shape[0] - act_dim - MAX_DELAY_STEPS
    rt_end = rt_start + MAX_DELAY_STEPS
    assert np.allclose(obs[rt_start:rt_end], 0.0)  # masked from the agent's own vector
    env.close()


def test_constant_delay_lags_applied_action_by_the_configured_amount():
    """With a fixed delay of 3 steps, the action actually applied to the
    native env must be the one sent 3 steps earlier (R1/R3), after an
    initial hold warm-up (R5)."""
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True, action_delay_min_steps=3, action_delay_max_steps=3,
        delay_known_to_agent=True,
    )
    env.reset(seed=0)
    applied = []
    inner_step = env.env.step

    def spy(action):
        applied.append(np.array(action, dtype=np.float32).copy())
        return inner_step(action)

    env.env.step = spy
    sent = [np.array([float(i), 0.0], dtype=np.float32) for i in range(10)]
    for action in sent:
        env.step(action)
    assert all(np.allclose(applied[i], [0.0, 0.0]) for i in range(3))  # warm-up hold
    assert all(np.allclose(applied[i], sent[i - 3]) for i in range(3, 10))
    env.close()


def test_out_of_order_command_is_discarded_not_applied():
    """Forces a send-order collision (a very old, very delayed command
    arrives the same step as a fresh zero-delay one) and checks R4: the
    fresher command wins and the stale one is discarded, never applied."""
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True, action_delay_min_steps=0, action_delay_max_steps=6,
        delay_known_to_agent=True,
    )
    env.reset(seed=0)
    forced_delays = iter([5, 0, 0, 0, 0, 0])
    env._sample_delay = lambda: next(forced_delays)
    applied = []
    inner_step = env.env.step

    def spy(action):
        applied.append(float(action[0]))
        return inner_step(action)

    env.env.step = spy
    discarded_per_step = []
    for i in range(6):
        obs, reward, terminated, truncated, info = env.step(np.array([i * 0.1, 0.0], dtype=np.float32))
        discarded_per_step.append(info["delay_debug"]["discarded_this_step"])
    assert np.allclose(applied, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6)
    assert discarded_per_step == [0, 0, 0, 0, 0, 1]
    env.close()


def test_capacity_invariant_rejects_delay_beyond_buffer_at_construction():
    with pytest.raises(ValueError):
        make_env("racetrack", enable_delay_wrapper=True, action_delay_min_steps=0, action_delay_max_steps=MAX_DELAY_STEPS + 1)


def test_capacity_boundary_max_delay_equal_to_capacity_is_sustainable():
    """max_delay_steps == capacity must run indefinitely without ever
    tripping the R6 invariant (it's the documented upper bound, not an
    overflow)."""
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True, action_delay_min_steps=MAX_DELAY_STEPS, action_delay_max_steps=MAX_DELAY_STEPS,
    )
    env.reset(seed=0)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert info["delay_debug"]["in_flight_count"] <= MAX_DELAY_STEPS
        if terminated or truncated:
            env.reset()
    env.close()


def test_action_history_fills_every_step_independent_of_delay():
    """R7: action_history records every sent action, even ones still
    in-flight or later discarded -- independent of pending_queue."""
    env = make_env(
        "racetrack", traffic_density="none",
        enable_delay_wrapper=True, action_delay_min_steps=4, action_delay_max_steps=4,
    )
    env.reset(seed=0)
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(np.array([float(i + 1), 0.0], dtype=np.float32))
        assert info["delay_debug"]["action_history"][-1][0] == pytest.approx(float(i + 1))
    env.close()


def test_oval_plus_traffic_speed_plus_delay_compose():
    """Full composition smoke test on the oval variant with a short straight
    (the geometry this project actually trains on), exercising all three
    additions together."""
    env = make_env(
        "racetrack_oval", traffic_density="light", oval_straight_length_m=5,
        traffic_speed_min=4.0, traffic_speed_max=6.0,
        enable_delay_wrapper=True, action_delay_min_steps=0, action_delay_max_steps=4,
        delay_known_to_agent=True,
    )
    obs, info = env.reset(seed=0)
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_config_loader_roundtrip(tmp_path):
    from rl_platform.config_loader import load_jsonc, experiment_config_to_arg_defaults

    p = tmp_path / "sample.jsonc"
    p.write_text(
        """
        {
          // a comment
          "env": "racetrack_oval",
          "oval_straight_length_m": 15, // inline comment
          "enable_delay_wrapper": true,
          "traffic_speed_min": null,
          "_benchmark_protocol_reference": {"note": "ignored"},
        }
        """,
        encoding="utf-8",
    )
    cfg = load_jsonc(str(p))
    defaults = experiment_config_to_arg_defaults(cfg)
    assert defaults == {
        "env": "racetrack_oval",
        "oval_straight_length_m": 15,
        "enable_delay_wrapper": True,
    }


def test_short_oval_straight_spawns_controlled_vehicle_on_track():
    """Regression test for a real HighwayEnv bug: RacetrackEnvOval's
    _make_vehicles() samples the controlled vehicle's initial longitudinal
    position from a hardcoded uniform(20.0, 50.0), regardless of the
    configured `length`. On a short straight (confirmed at the project's own
    30m and 5m oval variants) that range overruns the actual lane, spawning
    the car physically off-track. SafeSpawnWrapper (applied by default in
    make_racetrack_env/make_racetrack_oval_env) must catch and fix this."""
    for length in (5, 30, 80):
        for seed in range(10):
            env = make_env("racetrack_oval", oval_straight_length_m=length, traffic_density="none")
            obs, info = env.reset(seed=seed)
            vehicle = env.unwrapped.vehicle
            lane = env.unwrapped.road.network.get_lane(vehicle.lane_index)
            assert lane.on_lane(vehicle.position), (
                f"controlled vehicle spawned off-lane at oval_straight_length_m={length}, seed={seed}"
            )
            assert info["rewards"]["on_road_reward"]
            env.close()


def test_safe_spawn_fix_is_a_noop_when_native_spawn_was_already_valid():
    """On a normal-length straight (no bug triggered), the wrapper must not
    change anything observable: info/obs come straight from the native env."""
    env = make_env("racetrack", traffic_density="none")  # non-oval, never affected
    obs, info = env.reset(seed=0)
    vehicle = env.unwrapped.vehicle
    lane = env.unwrapped.road.network.get_lane(vehicle.lane_index)
    assert lane.on_lane(vehicle.position)
    env.close()


def test_high_speed_reward_linear_shape_matches_original_behavior():
    """Default shape ("linear") must stay byte-identical to the original,
    pre-existing formula: coef * clip(speed/speed_limit, 0, 1.2), never
    negative, flat 0 at/below standstill."""
    from rl_platform.envs.wrappers import HighSpeedRewardWrapper

    class _FakeSpeedEnv(gym.Wrapper):
        def __init__(self, speed):
            super().__init__(gym.make("racetrack-v0", config={"duration": 60}))
            self._speed = speed

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info["speed"] = self._speed
            return obs, reward, terminated, truncated, info

    for speed in (-5.0, 0.0, 3.0, 10.0, 15.0):
        env = _FakeSpeedEnv(speed)
        env.reset(seed=0)
        env = HighSpeedRewardWrapper(env, coef=0.4, speed_limit=10.0, shape="linear")
        _, _, _, _, info = env.step(env.action_space.sample())
        expected = 0.4 * float(np.clip(speed / 10.0, 0.0, 1.2))
        assert info["speed_bonus"] == pytest.approx(expected)
        env.close()


def test_high_speed_reward_tanh_shape_is_negative_at_standstill_and_saturates():
    """The "tanh" shape must match the explicitly requested curve: a small,
    BOUNDED negative bonus standing still or reversing (never unbounded),
    crossing zero a bit past standstill, and saturating (diminishing
    returns) rather than growing without limit at high speed."""
    from rl_platform.envs.wrappers import HighSpeedRewardWrapper

    class _FakeSpeedEnv(gym.Wrapper):
        def __init__(self, speed):
            super().__init__(gym.make("racetrack-v0", config={"duration": 60}))
            self._speed = speed

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info["speed"] = self._speed
            return obs, reward, terminated, truncated, info

    def bonus_at(speed):
        env = _FakeSpeedEnv(speed)
        env.reset(seed=0)
        env = HighSpeedRewardWrapper(
            env, coef=0.4, speed_limit=10.0, shape="tanh",
            v0_frac=0.1, steepness_frac=0.3,
        )
        _, _, _, _, info = env.step(env.action_space.sample())
        env.close()
        return info["speed_bonus"]

    standstill = bonus_at(0.0)
    reverse = bonus_at(-5.0)
    deep_reverse = bonus_at(-50.0)
    medium = bonus_at(5.0)
    at_limit = bonus_at(10.0)
    way_above_limit = bonus_at(100.0)

    # Standing still and reversing are both negative but small/bounded.
    assert standstill < 0
    assert reverse < 0
    assert -0.4 <= deep_reverse < -0.3  # saturates near -coef, never below it
    # Monotonically increasing with speed.
    assert standstill < medium < at_limit
    # Diminishing returns / saturation: the jump from "at limit" to "way
    # above limit" is much smaller than from "standstill" to "medium".
    assert (at_limit - medium) > (way_above_limit - at_limit)
    assert 0.3 < way_above_limit <= 0.4  # saturates near +coef, never exceeds it


def test_high_speed_reward_unknown_shape_rejected():
    from rl_platform.envs.wrappers import HighSpeedRewardWrapper

    env = gym.make("racetrack-v0", config={"duration": 60})
    with pytest.raises(ValueError):
        HighSpeedRewardWrapper(env, coef=0.4, speed_limit=10.0, shape="quadratic")

"""Tests for the environment registry / racetrack factory."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from rl_platform.envs import make_env, available_envs, TRAFFIC_PRESETS, RacetrackEnvConfig


def test_racetrack_registered():
    assert "racetrack" in available_envs()


def test_make_env_unknown_raises():
    with pytest.raises(KeyError):
        make_env("does_not_exist")


@pytest.mark.parametrize("traffic", ["none", "light", "medium", "heavy", "jam"])
def test_traffic_presets_resolve(traffic):
    cfg = RacetrackEnvConfig(traffic_density=traffic)
    hcfg = cfg.to_highway_config()
    assert hcfg["other_vehicles"] == TRAFFIC_PRESETS[traffic]


def test_reset_and_step_shapes():
    env = make_env("racetrack", traffic_density="medium")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.ndim == 1

    action = env.action_space.sample()
    assert action.shape == (2,)  # [acceleration, steering]

    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float) or np.isscalar(reward)
    env.close()


def test_traffic_density_changes_vehicle_count_config():
    env_none = make_env("racetrack", traffic_density="none")
    env_jam = make_env("racetrack", traffic_density="jam")
    assert env_none.unwrapped.config["other_vehicles"] == 0
    assert env_jam.unwrapped.config["other_vehicles"] == TRAFFIC_PRESETS["jam"]
    env_none.close()
    env_jam.close()


def test_flatten_obs_false_keeps_shape():
    env = make_env("racetrack", traffic_density="medium", flatten_obs=False, observation_type="Kinematics")
    obs, info = env.reset(seed=0)
    assert obs.ndim == 2  # (vehicles_count, features), not flattened
    env.close()


def test_high_speed_reward_wrapper_adds_speed_bonus():
    """Regression test for the fix in wrappers.py: RacetrackEnv's native
    reward has no speed term at all (only lane-centering, action-effort
    penalty, collision, and on-road gating), so without this wrapper an SAC
    agent has zero incentive to move quickly and reliably learns to crawl."""
    env_with_bonus = make_env("racetrack", traffic_density="none", high_speed_reward=0.4)
    env_without_bonus = make_env("racetrack", traffic_density="none", high_speed_reward=0.0)

    env_with_bonus.reset(seed=0)
    env_without_bonus.reset(seed=0)

    _, r_with, _, _, info = env_with_bonus.step(np.array([1.0, 0.0], dtype=np.float32))
    _, r_without, _, _, _ = env_without_bonus.step(np.array([1.0, 0.0], dtype=np.float32))

    assert "speed_bonus" in info
    assert info["speed_bonus"] > 0
    assert r_with > r_without
    env_with_bonus.close()
    env_without_bonus.close()


def test_render_rgb_array_is_nontrivial():
    """Regression test for the ForceRenderEnabled wrapper: under
    SDL_VIDEODRIVER=dummy, HighwayEnv's default viewer silently returns
    all-black frames unless we re-enable drawing (see wrappers.py)."""
    env = make_env("racetrack", traffic_density="medium", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None
    assert frame.std() > 1.0, "rendered frame looks blank -- render-enable workaround regressed"
    env.close()

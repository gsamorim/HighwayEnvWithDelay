"""Tests for the PPO-via-stable-baselines3 integration: the registry entry,
the Agent-interface wrapper (act/save/load, deliberate no-op observe/update),
and a short real end-to-end training run through
scripts/train.py::train_sb3_ppo (CSV schema + loadable checkpoint), using the
real racetrack env (not a mock) but a tiny step budget to keep this fast.
"""
import importlib.util
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# The "ppo" agent (unlike "sac"/"random") needs stable-baselines3, an
# optional dependency (see rl_platform/agents/ppo_sb3.py's lazy import --
# importing SB3PPOAgent itself never requires the package, only actually
# building/loading one does). On a machine without it installed, these
# tests should skip cleanly instead of failing with ImportError.
pytest.importorskip("stable_baselines3")

from rl_platform.agents import make_agent, available_agents
from rl_platform.agents.ppo_sb3 import SB3PPOAgent
from rl_platform.envs import make_env, TRAFFIC_PRESETS


OBS_DIM, ACT_DIM = 10, 2


def _load_train_module():
    """scripts/train.py isn't a package, so load it by path the same way
    `python scripts/train.py` would see it (its own `sys.path.insert` makes
    `rl_platform` importable either way)."""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    path = os.path.join(scripts_dir, "train.py")
    spec = importlib.util.spec_from_file_location("train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_lists_ppo():
    assert "ppo" in available_agents()


def test_ppo_act_shape_and_bounds():
    agent = make_agent("ppo", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    action = agent.act(obs, deterministic=True)
    assert action.shape == (ACT_DIM,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


def test_ppo_observe_and_update_are_noops():
    """Real PPO training goes through SB3's own .learn() (see
    train_sb3_ppo), not through this Agent's observe()/update() -- those
    must stay inert so a --agent ppo accidentally pointed at the generic
    off-policy Trainer does nothing rather than silently "training" wrong."""
    agent = make_agent("ppo", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    agent.observe(obs, action, 1.0, obs, False)  # must not raise
    assert agent.update() == {}
    assert agent.ready_to_update() is False


def test_make_agent_ppo_ignores_sac_only_kwargs():
    """train.py/record.py pass buffer_size/updates_per_step/learning_starts
    unconditionally regardless of --agent; the ppo factory must drop them
    instead of erroring, so the same CLI/config works for either agent."""
    agent = make_agent(
        "ppo",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        buffer_size=1000,
        updates_per_step=4,
        learning_starts=10,
        batch_size=32,
        seed=0,
    )
    assert isinstance(agent, SB3PPOAgent)
    # batch_size is a real SB3 PPO kwarg (shared naming with SAC) and should
    # have been kept, not dropped like the off-policy-only ones.
    assert agent._ppo_kwargs.get("batch_size") == 32


def test_ppo_save_load_roundtrip_reproduces_actions(tmp_path):
    agent = make_agent("ppo", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    action_before = agent.act(obs, deterministic=True)

    ckpt = str(tmp_path / "ppo_test.zip")
    agent.save(ckpt)
    assert os.path.exists(ckpt)

    agent2 = make_agent("ppo", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    agent2.load(ckpt)
    action_after = agent2.act(obs, deterministic=True)

    np.testing.assert_allclose(action_before, action_after, atol=1e-5)


def test_train_sb3_ppo_end_to_end_writes_csv_and_loadable_checkpoint(tmp_path):
    """Real env (racetrack, light traffic), tiny rollout -- exercises the
    actual SB3 `.learn()` path, the Monitor-based episode logging, periodic
    eval/checkpoint cadence, and that scripts/record.py-style loading of the
    resulting .zip checkpoint works."""
    train_module = _load_train_module()

    class Args:
        env = "racetrack"
        agent = "ppo"
        traffic = "light"
        total_steps = 128
        eval_every_steps = 64
        eval_episodes = 1
        checkpoint_every_steps = 128
        log_dir = str(tmp_path / "logs")
        checkpoint_dir = str(tmp_path / "checkpoints")
        seed = 0
        high_speed_reward = 0.4
        collision_reward = -5.0
        action_reward = -0.1
        speed_reward_shape = "linear"
        speed_reward_v0_frac = 0.1
        speed_reward_steepness_frac = 0.3
        ppo_n_steps = 64
        batch_size = 32
        ppo_n_epochs = 2
        ppo_learning_rate = 3e-4
        ppo_gamma = 0.99
        ppo_gae_lambda = 0.95
        ppo_clip_range = 0.2
        ppo_ent_coef = 0.0
        ppo_vf_coef = 0.5
        env_id = None
        oval_straight_length_m = None
        oval_obstacle_preset = None
        obstacle_min_clearance_m = None
        action_longitudinal = None
        traffic_speed_min = None
        traffic_speed_max = None
        enable_delay_wrapper = False
        action_delay_min_steps = 0
        action_delay_max_steps = 0
        delay_known_to_agent = False
        observation_delay_steps = 0

    args = Args()

    def env_factory():
        return make_env(
            "racetrack",
            traffic_density=TRAFFIC_PRESETS["light"],
            render_mode="rgb_array",
            high_speed_reward=args.high_speed_reward,
            collision_reward=args.collision_reward,
            action_reward=args.action_reward,
        )

    run_name = "ppo_pytest_smoke"
    final_ckpt = train_module.train_sb3(args, env_factory, run_name, algo="ppo")

    assert final_ckpt.endswith(".zip")
    assert os.path.exists(final_ckpt)

    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    for col in ("step", "episode", "episode_return", "episode_len", "crashed", "mean_speed"):
        assert col in header

    # The checkpoint must be loadable the same way record.py loads one.
    agent = make_agent("ppo", obs_dim=48, act_dim=2)
    agent.load(final_ckpt)
    obs, _info = env_factory().reset(seed=0)
    action = agent.act(obs, deterministic=True)
    assert action.shape == (2,)

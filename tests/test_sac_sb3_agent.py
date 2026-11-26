"""Tests for the SAC-via-stable-baselines3 integration: the registry entry,
the Agent-interface wrapper, a short real end-to-end training run through
scripts/train.py::train_sb3 with algo="sac_sb3" (CSV schema + loadable
checkpoint), and that the augmented-state ("estado aumentado",
enable_delay_wrapper) toggle works for it the same way it does for the
other two real agents -- it's an env-level concern, orthogonal to which
agent is used.
"""
import importlib.util
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# The "sac_sb3" agent (unlike "sac"/"random") needs stable-baselines3, an
# optional dependency (see rl_platform/agents/sac_sb3.py's lazy import --
# importing SB3SACAgent itself never requires the package, only actually
# building/loading one does). On a machine without it installed, these
# tests should skip cleanly instead of failing with ImportError.
pytest.importorskip("stable_baselines3")

from rl_platform.agents import make_agent, available_agents
from rl_platform.agents.sac_sb3 import SB3SACAgent, coerce_ent_coef
from rl_platform.envs import make_env, TRAFFIC_PRESETS


OBS_DIM, ACT_DIM = 10, 2


def _load_train_module():
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    path = os.path.join(scripts_dir, "train.py")
    spec = importlib.util.spec_from_file_location("train_script_sac_sb3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_lists_sac_sb3_alongside_sac_and_ppo():
    names = available_agents()
    assert "sac_sb3" in names
    assert "sac" in names  # from-scratch SAC must still be there too
    assert "ppo" in names


def test_coerce_ent_coef():
    assert coerce_ent_coef("auto") == "auto"
    assert coerce_ent_coef("auto_0.1") == "auto_0.1"
    assert coerce_ent_coef("0.05") == 0.05
    assert coerce_ent_coef(0.2) == 0.2


def test_sac_sb3_act_shape_and_bounds():
    agent = make_agent("sac_sb3", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.default_rng(0).normal(size=OBS_DIM).astype(np.float32)
    action = agent.act(obs, deterministic=True)
    assert action.shape == (ACT_DIM,)
    assert np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5)


def test_sac_sb3_observe_and_update_are_noops():
    agent = make_agent("sac_sb3", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    action = np.zeros(ACT_DIM, dtype=np.float32)
    agent.observe(obs, action, 1.0, obs, False)  # must not raise
    assert agent.update() == {}
    assert agent.ready_to_update() is False


def test_make_agent_sac_sb3_keeps_shared_kwargs_and_drops_ppo_only_ones():
    """buffer_size/batch_size/learning_starts are shared with our
    from-scratch SAC and must be kept; updates_per_step (from-scratch-SAC-
    only) and PPO-only kwargs must be dropped without erroring."""
    agent = make_agent(
        "sac_sb3",
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        buffer_size=5000,
        batch_size=64,
        learning_starts=100,
        updates_per_step=4,
        seed=0,
    )
    assert isinstance(agent, SB3SACAgent)
    assert agent._sac_kwargs.get("buffer_size") == 5000
    assert agent._sac_kwargs.get("batch_size") == 64
    assert agent._sac_kwargs.get("learning_starts") == 100
    assert "updates_per_step" not in agent._sac_kwargs


def test_sac_sb3_save_load_roundtrip_reproduces_actions(tmp_path):
    agent = make_agent("sac_sb3", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.default_rng(1).normal(size=OBS_DIM).astype(np.float32)
    action_before = agent.act(obs, deterministic=True)

    ckpt = str(tmp_path / "sac_sb3_test.zip")
    agent.save(ckpt)
    assert os.path.exists(ckpt)

    agent2 = make_agent("sac_sb3", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    agent2.load(ckpt)
    action_after = agent2.act(obs, deterministic=True)

    np.testing.assert_allclose(action_before, action_after, atol=1e-5)


def _base_args(tmp_path, agent="sac_sb3"):
    class Args:
        env = "racetrack"
        traffic = "light"
        total_steps = 100
        eval_every_steps = 50
        eval_episodes = 1
        checkpoint_every_steps = 100
        log_dir = str(tmp_path / "logs")
        checkpoint_dir = str(tmp_path / "checkpoints")
        seed = 0
        high_speed_reward = 0.4
        collision_reward = -5.0
        action_reward = -0.1
        speed_reward_shape = "linear"
        speed_reward_v0_frac = 0.1
        speed_reward_steepness_frac = 0.3
        buffer_size = 2000
        batch_size = 32
        learning_starts = 20
        sac_sb3_learning_rate = 3e-4
        sac_sb3_gamma = 0.99
        sac_sb3_tau = 0.005
        sac_sb3_train_freq = 1
        sac_sb3_gradient_steps = 1
        sac_sb3_ent_coef = "auto"
        sac_sb3_target_update_interval = 1
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
    args.agent = agent
    return args


def test_train_sb3_sac_end_to_end_writes_csv_and_loadable_checkpoint(tmp_path):
    train_module = _load_train_module()
    args = _base_args(tmp_path)

    def env_factory():
        return make_env(
            "racetrack",
            traffic_density=TRAFFIC_PRESETS["light"],
            render_mode="rgb_array",
            high_speed_reward=args.high_speed_reward,
            collision_reward=args.collision_reward,
            action_reward=args.action_reward,
        )

    run_name = "sac_sb3_pytest_smoke"
    final_ckpt = train_module.train_sb3(args, env_factory, run_name, algo="sac_sb3")

    assert final_ckpt.endswith(".zip")
    assert os.path.exists(final_ckpt)

    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    for col in ("step", "episode", "episode_return", "episode_len", "crashed", "mean_speed"):
        assert col in header

    agent = make_agent("sac_sb3", obs_dim=48, act_dim=2)
    agent.load(final_ckpt)
    obs, _info = env_factory().reset(seed=0)
    action = agent.act(obs, deterministic=True)
    assert action.shape == (2,)


def test_train_sb3_sac_with_augmented_state_is_just_a_toggle(tmp_path):
    """enable_delay_wrapper is an env-level concern: it must work for
    sac_sb3 the same way it already does for sac/ppo, with no agent-side
    change, by changing obs_dim (48 -> 80) that the agent picks up
    generically from the probed env."""
    from rl_platform.cli_common import env_kwargs_from_args

    train_module = _load_train_module()
    args = _base_args(tmp_path)
    args.enable_delay_wrapper = True
    args.action_delay_min_steps = 0
    args.action_delay_max_steps = 3
    args.delay_known_to_agent = True

    def env_factory():
        kwargs = dict(
            traffic_density=TRAFFIC_PRESETS["light"],
            render_mode="rgb_array",
            high_speed_reward=args.high_speed_reward,
            collision_reward=args.collision_reward,
            action_reward=args.action_reward,
        )
        kwargs.update(env_kwargs_from_args(args, delay_seed=args.seed))
        return make_env(args.env, **kwargs)

    probe_env = env_factory()
    obs_dim = probe_env.observation_space.shape[0]
    probe_env.close()
    assert obs_dim == 80  # 48 native + augmented-state fields (see project doc section 7)

    run_name = "sac_sb3_augmented_pytest_smoke"
    final_ckpt = train_module.train_sb3(args, env_factory, run_name, algo="sac_sb3")
    assert os.path.exists(final_ckpt)

    agent = make_agent("sac_sb3", obs_dim=obs_dim, act_dim=2)
    agent.load(final_ckpt)
    obs, _info = env_factory().reset(seed=0)
    assert obs.shape[0] == 80
    action = agent.act(obs, deterministic=True)
    assert action.shape == (2,)

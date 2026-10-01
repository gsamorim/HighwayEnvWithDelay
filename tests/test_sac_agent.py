"""Tests for the SAC agent: action shapes, replay buffer, gradient updates,
save/load round-trip, and that a handful of updates actually change the
network weights (i.e. learning is happening, not a no-op)."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from rl_platform.agents import make_agent, available_agents, SACAgent, SACConfig
from rl_platform.agents.replay_buffer import ReplayBuffer


OBS_DIM, ACT_DIM = 10, 2


def test_registry_lists_sac_and_random():
    names = available_agents()
    assert "sac" in names
    assert "random" in names


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(capacity=5, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    for i in range(8):  # more than capacity -> should wrap around
        buf.add(np.ones(OBS_DIM) * i, np.zeros(ACT_DIM), float(i), np.ones(OBS_DIM) * i, False)
    assert len(buf) == 5
    obs, act, rew, next_obs, done = buf.sample(4)
    assert obs.shape == (4, OBS_DIM)
    assert rew.shape == (4, 1)


def test_replay_buffer_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    rng = np.random.default_rng(0)
    for i in range(37):  # partially filled, not wrapped
        buf.add(rng.normal(size=OBS_DIM), rng.normal(size=ACT_DIM), float(i), rng.normal(size=OBS_DIM), i % 5 == 0)

    path = str(tmp_path / "buffer.npz")
    buf.save(path)

    buf2 = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=1)
    buf2.load(path)

    assert len(buf2) == len(buf) == 37
    np.testing.assert_allclose(buf2.obs[:37], buf.obs[:37])
    np.testing.assert_allclose(buf2.rewards[:37], buf.rewards[:37])
    np.testing.assert_allclose(buf2.dones[:37], buf.dones[:37])


def test_replay_buffer_save_load_after_wraparound(tmp_path):
    buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    for i in range(25):  # wraps around twice
        buf.add(np.ones(OBS_DIM) * i, np.zeros(ACT_DIM), float(i), np.ones(OBS_DIM) * i, False)

    path = str(tmp_path / "buffer.npz")
    buf.save(path)

    buf2 = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=1)
    buf2.load(path)
    assert len(buf2) == 10
    np.testing.assert_allclose(buf2.rewards.squeeze(), buf.rewards.squeeze())


def test_sac_act_shape_and_bounds():
    agent = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action = agent.act(obs, deterministic=False)
    assert action.shape == (ACT_DIM,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    action_det = agent.act(obs, deterministic=True)
    action_det2 = agent.act(obs, deterministic=True)
    np.testing.assert_allclose(action_det, action_det2, err_msg="deterministic actions should be repeatable")


def test_sac_not_ready_before_learning_starts():
    agent = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, learning_starts=50, batch_size=32, seed=0)
    assert not agent.ready_to_update()
    for _ in range(60):
        agent.observe(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 0.0, np.zeros(OBS_DIM), False)
    assert agent.ready_to_update()


def test_sac_update_changes_weights_and_returns_metrics():
    agent = make_agent(
        "sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, learning_starts=32, batch_size=32, buffer_size=1000, seed=0
    )
    rng = np.random.default_rng(0)
    for _ in range(64):
        obs = rng.normal(size=OBS_DIM).astype(np.float32)
        act = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
        next_obs = rng.normal(size=OBS_DIM).astype(np.float32)
        reward = float(rng.normal())
        agent.observe(obs, act, reward, next_obs, False)

    before = [p.clone() for p in agent.actor.parameters()]
    metrics = {}
    for _ in range(5):
        metrics = agent.update()
    after = list(agent.actor.parameters())

    assert metrics, "expected non-empty metrics once ready_to_update() is True"
    assert "critic_loss" in metrics and "actor_loss" in metrics and "alpha" in metrics
    assert any(not torch.allclose(b, a) for b, a in zip(before, after)), "actor weights did not change after updates"


def test_sac_save_load_roundtrip(tmp_path):
    agent = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    action_before = agent.act(obs, deterministic=True)

    path = str(tmp_path / "agent.pt")
    agent.save(path)

    agent2 = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=1)  # different seed/init
    action_before_load = agent2.act(obs, deterministic=True)
    agent2.load(path)
    action_after_load = agent2.act(obs, deterministic=True)

    np.testing.assert_allclose(action_before, action_after_load, atol=1e-5)
    assert not np.allclose(action_before_load, action_after_load), "load() should have changed agent2's behavior"


def test_sac_save_load_resumes_optimizer_state_and_step_count(tmp_path):
    """Checkpoints must carry optimizer momentum + the gradient-step counter,
    not just weights, so `scripts/live_trainer.py` can resume training
    byte-for-byte rather than restarting Adam from scratch every time the
    window is closed and reopened."""
    agent = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, learning_starts=16, batch_size=16,
                        buffer_size=500, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        agent.observe(
            rng.normal(size=OBS_DIM).astype(np.float32), rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32),
            float(rng.normal()), rng.normal(size=OBS_DIM).astype(np.float32), False,
        )
    for _ in range(10):
        agent.update()
    assert agent.update_count == 10
    # Adam's per-parameter running averages should be non-empty after real updates.
    assert len(agent.actor_opt.state_dict()["state"]) > 0

    path = str(tmp_path / "agent.pt")
    agent.save(path)

    agent2 = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, learning_starts=16, batch_size=16,
                         buffer_size=500, seed=1)
    assert agent2.update_count == 0
    agent2.load(path)
    assert agent2.update_count == 10
    assert len(agent2.actor_opt.state_dict()["state"]) == len(agent.actor_opt.state_dict()["state"])


def test_sac_load_tolerates_checkpoints_without_optimizer_state(tmp_path):
    """Simulates loading one of the flat eval-only checkpoints saved before
    optimizer state was added -- must not raise, and should fall back to a
    freshly-initialized optimizer."""
    agent = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=0)
    import torch
    old_style_ckpt = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "cfg": agent.cfg,
    }
    path = str(tmp_path / "old_style.pt")
    torch.save(old_style_ckpt, path)

    agent2 = make_agent("sac", obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=1)
    agent2.load(path)  # should not raise
    assert agent2.update_count == 0

import numpy as np
import pytest

from rl_platform.envs.delay_experiment import make_delay_experiment_env


@pytest.mark.parametrize("accel", [False, True])
def test_observation_size_is_the_same_with_and_without_delay(accel):
    sizes = {
        make_delay_experiment_env(accel=accel, delay_mode=mode, fixed_delay=2, traffic=0).observation_space.shape
        for mode in ("off", "random", "fixed")
    }
    act_dim = 2 if accel else 1
    assert sizes == {(2 * 12 * 12 + 1 + 6 * act_dim + 1,)}


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_fixed_delay_executes_action_from_k_steps_ago(k):
    env = make_delay_experiment_env(accel=False, delay_mode="fixed", fixed_delay=k, traffic=0)
    env.reset(seed=0)
    sent = []
    for t in range(12):
        a = np.array([0.05 * np.sin(t)], dtype=np.float32)
        sent.append(a)
        _, _, terminated, truncated, info = env.step(a)
        expected = sent[t - k] if t >= k else np.zeros(1)
        np.testing.assert_allclose(info["executed_action"], expected)
        assert info["delay_k"] == k
        if terminated or truncated:
            break


def test_random_delay_never_rises_more_than_one_step():
    env = make_delay_experiment_env(accel=False, delay_mode="random", max_delay=3, traffic=0)
    env.reset(seed=0)
    prev = 0
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        k = info["delay_k"]
        assert 0 <= k <= 3 and k <= prev + 1
        prev = k
        if terminated or truncated:
            env.reset()
            prev = 0


def test_observation_carries_last_actions_and_delay():
    env = make_delay_experiment_env(accel=True, delay_mode="fixed", fixed_delay=3, traffic=0)
    obs, _ = env.reset(seed=0)
    assert obs[-1] == pytest.approx(3 / 6)
    a = np.array([0.2, -0.1], dtype=np.float32)
    obs, *_ = env.step(a)
    np.testing.assert_allclose(obs[-1 - 12 : -1 - 10], a)  # ação mais recente primeiro


def test_standing_still_earns_two_thirds_of_driving_reward():
    env = make_delay_experiment_env(accel=True, delay_mode="off", traffic=0)
    env.reset(seed=0)
    env.unwrapped.vehicle.speed = 0.0
    _, reward, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert reward == pytest.approx(2 / 3, abs=0.02)


def test_reset_seed_reproduces_delay_sequence():
    def seq():
        env = make_delay_experiment_env(accel=False, delay_mode="random", traffic=0)
        env.reset(seed=7)
        return [env.step(np.zeros(1, dtype=np.float32))[4]["delay_k"] for _ in range(6)]

    assert seq() == seq()

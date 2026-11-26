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


def test_standing_still_earns_less_than_driving():
    env = make_delay_experiment_env(accel=True, delay_mode="off", traffic=0)
    env.reset(seed=0)
    env.unwrapped.vehicle.speed = 0.0
    _, reward, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert reward == pytest.approx(2.3 / 3.3, abs=0.02)  # centralizado, parado; dirigindo bem = 1


@pytest.mark.parametrize("speed", [0.0, 10.0])
def test_reward_stays_in_unit_interval_even_on_worst_crash(speed):
    """Pior caso: bate parado, esterçando no máximo, longe do centro da faixa."""
    env = make_delay_experiment_env(accel=True, delay_mode="off", traffic=0)
    env.reset(seed=0)
    base = env.unwrapped
    v = base.vehicle
    v.position = v.lane.position(v.lane.local_coordinates(v.position)[0], 2.0)  # ainda na pista
    v.speed = speed
    v.crashed = True
    _, reward, *_ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert 0.0 <= reward <= 1.0


def test_reset_seed_reproduces_delay_sequence():
    def seq():
        env = make_delay_experiment_env(accel=False, delay_mode="random", traffic=0)
        env.reset(seed=7)
        return [env.step(np.zeros(1, dtype=np.float32))[4]["delay_k"] for _ in range(6)]

    assert seq() == seq()


def _reward_wrapper(env):
    from rl_platform.envs.delay_experiment import ProjectRewardWrapper
    while not isinstance(env, ProjectRewardWrapper):
        env = env.env
    return env


def test_lap_crossing_counts_start_line_in_both_directions():
    env = make_delay_experiment_env(accel=False, traffic=0)
    env.reset(seed=0)
    w = _reward_wrapper(env)
    w._prev_xy = np.array([-0.5, 5.0])
    assert w._lap_crossing(np.array([0.5, 5.0])) == 1       # para a frente, na reta a→b
    w._prev_xy = np.array([0.5, 5.0])
    assert w._lap_crossing(np.array([-0.5, 5.0])) == -1     # para trás
    w._prev_xy = np.array([-0.5, -65.0])
    assert w._lap_crossing(np.array([0.5, -65.0])) == 0     # x = 0 na reta oposta não conta


def test_driving_three_laps_counts_three_crossings():
    """Controlador simples sem tráfego: ~1000 m em 100 s; com volta ≈ 300 m,
    largando no meio da reta, cruza a linha 3 vezes."""
    env = make_delay_experiment_env(accel=False, traffic=0)
    env.reset(seed=0)
    laps, progress, done = 0, 0.0, False
    while not done:
        v = env.unwrapped.vehicle
        s, _ = v.lane.local_coordinates(v.position)
        target = v.lane.position(s + 8, 0)
        angle = np.arctan2(target[1] - v.position[1], target[0] - v.position[0])
        steer = np.clip(((angle - v.heading + np.pi) % (2 * np.pi) - np.pi) / 0.5236, -1, 1)
        _, _, terminated, truncated, info = env.step(np.array([steer], dtype=np.float32))
        laps += info["lap_crossing"]
        progress += info["progress_m"]
        done = terminated or truncated
    assert truncated and not terminated
    assert laps == 3 and 950 < progress < 1050


def test_run_command_resumes_validates_and_refuses(tmp_path):
    from scripts.run_experiment import main

    common = ["run", "--algo", "ppo", "--condition", "C1", "--seed", "0", "--out-dir", str(tmp_path),
              "--eval-every", "0", "--eval-episodes", "1", "--duration-s", "4", "--traffic", "0"]
    run_dir = tmp_path / "ppo_C1_seed0"
    assert main(common + ["--timesteps", "64"]) == 0
    assert (run_dir / "model.zip").is_file()
    mtime = (run_dir / "model.zip").stat().st_mtime
    assert main(common + ["--timesteps", "64"]) == 0                      # já concluído: não retreina
    assert (run_dir / "model.zip").stat().st_mtime == mtime

    (run_dir / "eval_summary.csv").write_text("eval_delay,n_episodes\n")  # CSV só com cabeçalho
    assert main(common + ["--timesteps", "64"]) == 0                      # reavalia sem retreinar
    assert (run_dir / "model.zip").stat().st_mtime == mtime
    assert len((run_dir / "eval_summary.csv").read_text().splitlines()) == 8

    assert main(common + ["--timesteps", "128"]) == 3                     # outra configuração: recusa
    assert (run_dir / "model.zip").stat().st_mtime == mtime

"""End-to-end smoke tests: a short training run wires env+agent+trainer+logger
correctly, and the video export pipeline produces non-blank frames with a
readable HUD."""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl_platform.envs import make_env
from rl_platform.agents import make_agent
from rl_platform.training import Trainer, TrainerConfig
from rl_platform.training.plotting import read_csv_log, plot_training_curves
from rl_platform.visualization import overlay_hud
from rl_platform.visualization.video_export import record_episode, save_gif


def _env_factory(traffic="medium"):
    return lambda: make_env("racetrack", traffic_density=traffic, render_mode="rgb_array")


def test_short_training_run_produces_logs_and_checkpoint(tmp_path):
    env_factory = _env_factory("light")
    probe = env_factory()
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    agent = make_agent("sac", obs_dim=obs_dim, act_dim=act_dim, learning_starts=50, batch_size=32,
                        buffer_size=2000, seed=0)
    cfg = TrainerConfig(
        total_steps=300,
        start_steps=50,
        eval_every_steps=150,
        eval_episodes=1,
        checkpoint_every_steps=300,
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="pytest_run",
        seed=0,
    )
    trainer = Trainer(env_factory, agent, cfg)
    result = trainer.train()

    assert len(result["train_history"]) > 0
    assert len(result["eval_history"]) >= 1

    csv_path = os.path.join(cfg.log_dir, "pytest_run.csv")
    assert os.path.exists(csv_path)
    data = read_csv_log(csv_path)
    assert "episode_return" in data and len(data["episode_return"]) > 0

    ckpt_path = os.path.join(cfg.checkpoint_dir, "pytest_run_final.pt")
    assert os.path.exists(ckpt_path)

    plot_path = str(tmp_path / "curves.png")
    plot_training_curves(csv_path, plot_path)
    assert os.path.exists(plot_path) and os.path.getsize(plot_path) > 0


def test_video_export_frames_are_nonblank_with_hud(tmp_path):
    env = make_env("racetrack", traffic_density="medium", render_mode="rgb_array")
    agent = make_agent("random", obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])
    frames, summary = record_episode(env, lambda obs: agent.act(obs), seed=0, max_steps=10)
    env.close()

    assert len(frames) > 1
    assert frames[0].std() > 1.0, "recorded frame looks blank"

    hud_frame = overlay_hud(frames[0], ["hello", "world"])
    assert hud_frame.shape == frames[0].shape

    gif_path = str(tmp_path / "rollout.gif")
    save_gif(frames, gif_path, fps=10)
    assert os.path.exists(gif_path) and os.path.getsize(gif_path) > 0

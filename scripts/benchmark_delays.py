"""Train SAC policies on the oval and evaluate their final weights at fixed delays."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from importlib.metadata import version
from pathlib import Path
import sys

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_platform.agents import make_agent
from rl_platform.envs import MAX_DELAY_STEPS, make_env
from rl_platform.training import Trainer, TrainerConfig
from rl_platform.training.plotting import plot_training_curves


ENV_ID = "racetrack-oval-v1"
POLICY_HZ = 5
STRAIGHT_LENGTH_M = 80
EVAL_FIRST_SEED = 100_000
EVAL_COLUMNS = (
    "policy", "train_seed", "train_max_delay_steps", "eval_delay_steps",
    "n_episodes", "mean_return", "std_return", "crash_rate", "mean_speed", "mean_length",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_benchmark_env(duration: int, augmented: bool, min_delay: int, max_delay: int) -> gym.Env:
    env = make_env(
        "racetrack_oval",
        env_id=ENV_ID,
        duration=duration,
        policy_frequency=POLICY_HZ,
        oval_straight_length_m=STRAIGHT_LENGTH_M,
        traffic_density="medium",
        action_longitudinal=False,
        enable_delay_wrapper=augmented or max_delay > 0,
        action_delay_min_steps=min_delay,
        action_delay_max_steps=max_delay,
        delay_known_to_agent=augmented,
    )
    if max_delay and not augmented:
        # Keep real action delays, but give baseline SAC only the native
        # observation prefix that its actor was trained to accept.
        native_space = env.env.observation_space
        native_size = native_space.shape[0]
        env = gym.wrappers.TransformObservation(env, lambda obs: obs[:native_size], native_space)
    return gym.wrappers.TimeLimit(env, max_episode_steps=duration * POLICY_HZ)


def run_policy(args: argparse.Namespace, seed: int, train_max_delay: int, writer: csv.DictWriter) -> None:
    augmented = train_max_delay > 0
    label = "sac_atrasos" if augmented else "sac"
    name = f"{label}_seed{seed}"
    env_factory = partial(make_benchmark_env, args.duration, augmented, 0, train_max_delay)
    probe = env_factory()
    try:
        obs_dim = probe.observation_space.shape[0]
        act_dim = probe.action_space.shape[0]
    finally:
        probe.close()

    agent = make_agent("sac", obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    interval = min(args.total_steps, 50_000)
    trainer = Trainer(
        env_factory,
        agent,
        TrainerConfig(
            total_steps=args.total_steps,
            eval_every_steps=interval,
            eval_episodes=min(5, args.eval_episodes),
            checkpoint_every_steps=interval,
            log_dir=str(args.output_dir / "logs"),
            checkpoint_dir=str(args.output_dir / "checkpoints"),
            run_name=name,
            seed=seed,
        ),
    )
    trainer.logger.log_meta({
        "env_id": ENV_ID,
        "highway_env_version": version("highway-env"),
        "policy": label,
        "train_seed": seed,
        "train_max_delay_steps": train_max_delay,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "args": vars(args),
    })
    print(f"Training {name} for {args.total_steps} steps (delay 0..{train_max_delay})", flush=True)
    history = trainer.train()
    if history["train_history"]:
        plots_dir = args.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_training_curves(trainer.logger.csv_path, plots_dir / f"{name}.png", title=name)

    for delay in args.eval_delays:
        trainer.eval_env_factory = partial(make_benchmark_env, args.duration, augmented, delay, delay)
        stats = trainer.evaluate(args.eval_episodes, seed=EVAL_FIRST_SEED)
        writer.writerow({
            "policy": label,
            "train_seed": seed,
            "train_max_delay_steps": train_max_delay,
            "eval_delay_steps": delay,
            **stats,
        })
        print(
            f"{name}: delay={delay}, episodes={stats['n_episodes']}, "
            f"return={stats['mean_return']:.2f}, crash_rate={stats['crash_rate']:.2f}",
            flush=True,
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this benchmark")
    parser.add_argument("--duration", type=positive_int, default=100, help="simulated seconds per episode")
    parser.add_argument("--total-steps", type=positive_int, default=5_000, help="training steps per policy and seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds")
    parser.add_argument(
        "--train-max-delay", type=int, choices=range(MAX_DELAY_STEPS + 1), default=0,
        help="if positive, also train an augmented SAC with uniform delays from 0 to this value",
    )
    parser.add_argument("--eval-delays", type=int, choices=range(MAX_DELAY_STEPS + 1), nargs="+", default=[0])
    parser.add_argument("--eval-episodes", type=positive_int, default=10, help="episodes per trained policy and delay")
    args = parser.parse_args(argv)
    if min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be distinct nonnegative integers")
    if len(args.eval_delays) != len(set(args.eval_delays)):
        parser.error("--eval-delays must be distinct")

    try:
        args.output_dir.mkdir(parents=True)
    except FileExistsError:
        parser.error("--output-dir must be a new directory")

    training_delays = [0, args.train_max_delay] if args.train_max_delay else [0]
    summary_path = args.output_dir / "avaliacao.csv"
    with summary_path.open("w", encoding="utf-8", newline="", buffering=1) as summary:
        writer = csv.DictWriter(summary, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        for seed in args.seeds:
            for train_max_delay in training_delays:
                run_policy(args, seed, train_max_delay, writer)
    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()

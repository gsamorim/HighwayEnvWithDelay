#!/usr/bin/env python3
"""Train an agent on an environment from the registry.

Example:
    python scripts/train.py --env racetrack --agent sac --traffic medium \
        --total-steps 50000 --run-name sac_medium

Or drive it from the unified experiment config (any flag below still
overrides the file):
    python scripts/train.py --config configs/racetrack_experiment_config.jsonc
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_platform.envs import make_env, available_envs
from rl_platform.agents import make_agent, available_agents
from rl_platform.training import Trainer, TrainerConfig
from rl_platform.training.plotting import plot_training_curves
from rl_platform.config_loader import load_jsonc, experiment_config_to_arg_defaults
from rl_platform.cli_common import add_env_cli_args, env_kwargs_from_args, traffic_arg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="racetrack", choices=available_envs())
    p.add_argument("--agent", default="sac", choices=available_agents())
    p.add_argument("--traffic", default="medium", help="traffic preset (none/light/medium/heavy/jam) or an int")
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--start-steps", type=int, default=2_000)
    p.add_argument("--eval-every-steps", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-every-steps", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-dir", default="results/logs")
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--plot-dir", default="results/plots")
    # Reward-shaping overrides (racetrack-v0 has no native speed reward -- see
    # rl_platform/envs/wrappers.py::HighSpeedRewardWrapper -- so these matter a lot).
    p.add_argument("--high-speed-reward", type=float, default=0.4, help="coefficient for the speed-bonus reward wrapper")
    p.add_argument("--collision-reward", type=float, default=-5.0)
    p.add_argument("--action-reward", type=float, default=-0.1)
    p.add_argument("--speed-reward-shape", default="linear", choices=["linear", "tanh"],
                    help="'linear' (default, original): coef*clip(speed/speed_limit,0,1.2), never negative. "
                         "'tanh': coef*tanh((speed-v0)/steepness) -- small bounded negative bonus "
                         "standing still/reversing, steep rise, diminishing returns at high speed.")
    p.add_argument("--speed-reward-v0-frac", type=float, default=0.1,
                    help="(tanh shape only) zero-crossing speed as a fraction of speed_limit")
    p.add_argument("--speed-reward-steepness-frac", type=float, default=0.3,
                    help="(tanh shape only) rise steepness as a fraction of speed_limit")

    add_env_cli_args(p)
    return p


def main(argv=None):
    # A lightweight pre-parse just for --config, so its values can be
    # installed as argparse *defaults* before the real parse -- any flag
    # also given explicitly on the command line still wins (argparse
    # precedence: explicit CLI args override set_defaults()).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                      help="path to a .jsonc experiment config (configs/*.jsonc); "
                           "CLI flags always override its values")
    pre_args, remaining = pre.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        cfg = load_jsonc(pre_args.config)
        parser.set_defaults(**experiment_config_to_arg_defaults(cfg))
    args = parser.parse_args(remaining)

    run_name = args.run_name or f"{args.agent}_{args.env}_traffic-{args.traffic}_seed{args.seed}"

    def env_factory():
        kwargs = dict(
            traffic_density=traffic_arg(args.traffic),
            render_mode="rgb_array",
            high_speed_reward=args.high_speed_reward,
            collision_reward=args.collision_reward,
            action_reward=args.action_reward,
            speed_reward_shape=args.speed_reward_shape,
            speed_reward_v0_frac=args.speed_reward_v0_frac,
            speed_reward_steepness_frac=args.speed_reward_steepness_frac,
        )
        kwargs.update(env_kwargs_from_args(args, delay_seed=args.seed))
        return make_env(args.env, **kwargs)

    probe_env = env_factory()
    obs_dim = probe_env.observation_space.shape[0]
    act_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = make_agent(
        args.agent,
        obs_dim=obs_dim,
        act_dim=act_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        learning_starts=args.learning_starts,
        seed=args.seed,
    )

    trainer_cfg = TrainerConfig(
        total_steps=args.total_steps,
        start_steps=args.start_steps,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        checkpoint_every_steps=args.checkpoint_every_steps,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        run_name=run_name,
        seed=args.seed,
    )
    trainer = Trainer(env_factory, agent, trainer_cfg)
    trainer.logger.log_meta(
        {
            "env": args.env,
            "agent": args.agent,
            "traffic": args.traffic,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "args": vars(args),
        }
    )

    print(f"Training {args.agent} on {args.env} (traffic={args.traffic}) for {args.total_steps} steps ...")
    trainer.train()

    os.makedirs(args.plot_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")
    plot_path = os.path.join(args.plot_dir, f"{run_name}.png")
    try:
        plot_training_curves(csv_path, plot_path, title=f"{args.agent} on {args.env} (traffic={args.traffic})")
        print(f"Saved training curves to {plot_path}")
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"Plotting failed: {e}")

    print(f"Done. Logs: {csv_path}")
    print(f"Final checkpoint: {os.path.join(args.checkpoint_dir, run_name + '_final.pt')}")


if __name__ == "__main__":
    main()

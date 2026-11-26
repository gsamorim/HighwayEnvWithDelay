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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_platform.envs import make_env, available_envs
from rl_platform.agents import make_agent, available_agents
from rl_platform.training import Trainer, TrainerConfig, RunLogger
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

    # PPO-only knobs (--agent ppo): forwarded to stable-baselines3's own PPO
    # constructor (see rl_platform/agents/ppo_sb3.py::SB3_PPO_KWARGS).
    # Ignored, with no error, when --agent sac/random -- so a single
    # .jsonc/CLI invocation can switch --agent without editing these.
    p.add_argument("--ppo-n-steps", type=int, default=2048,
                    help="rollout length collected before each PPO update (SB3 default: 2048)")
    p.add_argument("--ppo-n-epochs", type=int, default=10,
                    help="minibatch epochs per rollout (SB3 default: 10)")
    p.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    p.add_argument("--ppo-gamma", type=float, default=0.99)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-clip-range", type=float, default=0.2)
    p.add_argument("--ppo-ent-coef", type=float, default=0.0)
    p.add_argument("--ppo-vf-coef", type=float, default=0.5)

    # SAC-via-SB3-only knobs (--agent sac_sb3): forwarded to
    # stable-baselines3's own SAC constructor (see
    # rl_platform/agents/sac_sb3.py::SB3_SAC_KWARGS). Ignored, with no
    # error, when --agent sac/random/ppo. --buffer-size/--batch-size/
    # --learning-starts above are SHARED with our from-scratch SAC (same
    # meaning for both), so they are not repeated here.
    p.add_argument("--sac-sb3-learning-rate", type=float, default=3e-4)
    p.add_argument("--sac-sb3-gamma", type=float, default=0.99)
    p.add_argument("--sac-sb3-tau", type=float, default=0.005,
                    help="target-network Polyak averaging rate (SB3 default: 0.005)")
    p.add_argument("--sac-sb3-train-freq", type=int, default=1,
                    help="env steps between gradient updates (SB3 default: 1)")
    p.add_argument("--sac-sb3-gradient-steps", type=int, default=1,
                    help="gradient updates per train_freq interval (SB3 default: 1)")
    p.add_argument("--sac-sb3-ent-coef", default="auto",
                    help='"auto" (default, SB3 autotunes it like our from-scratch SAC does), '
                         '"auto_<value>" (custom init), or a fixed float')
    p.add_argument("--sac-sb3-target-update-interval", type=int, default=1)

    add_env_cli_args(p)
    return p


def train_sb3(args, env_factory, run_name: str, algo: str) -> str:
    """Real training for either stable-baselines3-backed agent (`algo` is
    "ppo" or "sac_sb3"), via SB3's own `.learn()` loop -- NOT through
    `rl_platform.training.Trainer`. For "ppo" that's because the Trainer's
    loop is off-policy-only by design (see `rl_platform/agents/ppo_sb3.py`'s
    module docstring). For "sac_sb3" -- despite SAC itself being
    off-policy -- it's because SB3's `SAC` doesn't expose its replay
    buffer/gradient-step cadence as something meant to be driven externally
    per env step (see `rl_platform/agents/sac_sb3.py`'s module docstring);
    `.learn()` is SB3's intended entry point for both algorithms, which is
    exactly why one function here can drive either.

    Writes a CSV log in the same schema `Trainer` writes (so
    `plot_training_curves` works unmodified on either SB3-backed run) and a
    `.zip` checkpoint (SB3's own save format, not our `.pt` state_dict
    format -- `scripts/record.py` tells them apart by extension).
    Returns the final checkpoint path.
    """
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        raise ImportError(
            f"--agent {algo} needs stable-baselines3, which isn't installed "
            "in this Python environment. Install it with:\n"
            "    pip install stable-baselines3\n"
            "(it's listed in requirements.txt). --agent sac/random don't "
            "need it and are unaffected."
        ) from e

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = RunLogger(args.log_dir, run_name)
    logger.log_meta(
        {
            "env": args.env,
            "agent": algo,
            "traffic": args.traffic,
            "args": vars(args),
        }
    )

    class _LogAndCheckpointCallback(BaseCallback):
        """Logs one CSV row per finished episode (matching `Trainer`'s
        column schema: step/episode/episode_return/episode_len/crashed/
        mean_speed) and periodically evaluates + checkpoints, mirroring
        `Trainer.train()`'s cadence/console-output so a PPO run's console
        and CSV look the same shape as a SAC run's."""

        def __init__(self):
            super().__init__(verbose=0)
            self.episode_idx = 0
            self._next_eval_at = args.eval_every_steps
            self._next_ckpt_at = args.checkpoint_every_steps
            self._t_start = time.time()

        def _on_step(self) -> bool:
            # VecEnv with a single sub-env: index 0 throughout.
            infos = self.locals.get("infos", [{}])
            dones = self.locals.get("dones", [False])
            if dones[0]:
                info = infos[0]
                ep_info = info.get("episode", {})
                row = {
                    "step": self.num_timesteps,
                    "episode": self.episode_idx,
                    "episode_return": round(float(ep_info.get("r", float("nan"))), 4),
                    "episode_len": int(ep_info.get("l", 0)),
                    "crashed": int(bool(info.get("crashed", False))),
                    "mean_speed": float(info.get("speed", float("nan"))),
                }
                logger.log_row(row)
                self.episode_idx += 1

            if self.num_timesteps >= self._next_eval_at:
                self._next_eval_at += args.eval_every_steps
                eval_stats = self._evaluate(args.eval_episodes, seed=10_000 + self.num_timesteps)
                eval_stats["step"] = self.num_timesteps
                eval_stats["wall_time_s"] = round(time.time() - self._t_start, 1)
                print(
                    f"[eval @ step {self.num_timesteps:>7d}] "
                    f"return={eval_stats['mean_return']:.2f}+/-{eval_stats['std_return']:.2f} "
                    f"crash_rate={eval_stats['crash_rate']:.2f} "
                    f"mean_speed={eval_stats['mean_speed']:.2f} "
                    f"({eval_stats['wall_time_s']:.0f}s elapsed)"
                )

            if self.num_timesteps >= self._next_ckpt_at:
                self._next_ckpt_at += args.checkpoint_every_steps
                ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}_step{self.num_timesteps}.zip")
                self.model.save(ckpt_path)

            return True

        def _evaluate(self, n_episodes: int, seed: int):
            import numpy as np

            eval_env = env_factory()
            returns, crashes, speeds = [], [], []
            for ep in range(n_episodes):
                obs, info = eval_env.reset(seed=seed + ep)
                done = False
                ep_return, ep_speeds, crashed = 0.0, [], False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_return += float(reward)
                    crashed = crashed or bool(info.get("crashed", False))
                    if "speed" in info:
                        ep_speeds.append(float(info["speed"]))
                returns.append(ep_return)
                crashes.append(float(crashed))
                speeds.append(float(np.mean(ep_speeds)) if ep_speeds else float("nan"))
            eval_env.close()
            return {
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "crash_rate": float(np.mean(crashes)),
                "mean_speed": float(np.nanmean(speeds)),
            }

    # SB3's Monitor wrapper is what populates info["episode"] = {"r", "l", "t"}
    # on the terminal step -- that's what the callback reads to log a CSV row
    # per finished episode; our env's own "crashed"/"speed" info keys survive
    # alongside it since Monitor only adds a key, it doesn't replace info.
    vec_env = DummyVecEnv([lambda: Monitor(env_factory())])

    if algo == "ppo":
        from stable_baselines3 import PPO as SB3Algo
        model = SB3Algo(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            verbose=0,
            n_steps=args.ppo_n_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_n_epochs,
            learning_rate=args.ppo_learning_rate,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            clip_range=args.ppo_clip_range,
            ent_coef=args.ppo_ent_coef,
            vf_coef=args.ppo_vf_coef,
        )
    elif algo == "sac_sb3":
        from stable_baselines3 import SAC as SB3Algo
        from rl_platform.agents.sac_sb3 import coerce_ent_coef
        model = SB3Algo(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            verbose=0,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            learning_rate=args.sac_sb3_learning_rate,
            gamma=args.sac_sb3_gamma,
            tau=args.sac_sb3_tau,
            train_freq=args.sac_sb3_train_freq,
            gradient_steps=args.sac_sb3_gradient_steps,
            ent_coef=coerce_ent_coef(args.sac_sb3_ent_coef),
            target_update_interval=args.sac_sb3_target_update_interval,
        )
    else:
        raise ValueError(f"train_sb3() does not know algo '{algo}' (expected 'ppo' or 'sac_sb3')")

    print(f"Training {algo} (stable-baselines3) on {args.env} (traffic={args.traffic}) for {args.total_steps} steps ...")
    model.learn(total_timesteps=args.total_steps, callback=_LogAndCheckpointCallback())

    final_ckpt = os.path.join(args.checkpoint_dir, f"{run_name}_final.zip")
    model.save(final_ckpt)
    logger.close()
    vec_env.close()
    return final_ckpt


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

    if args.agent in ("ppo", "sac_sb3"):
        # Real training for either SB3-backed agent goes through SB3's own
        # `.learn()` loop, not the generic off-policy `Trainer` below -- see
        # rl_platform/agents/ppo_sb3.py / sac_sb3.py and train_sb3()'s own
        # docstring for why. `final_ckpt` is a `.zip` (SB3's save format),
        # not `.pt`.
        final_ckpt = train_sb3(args, env_factory, run_name, algo=args.agent)
    else:
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
        final_ckpt = os.path.join(args.checkpoint_dir, run_name + "_final.pt")

    os.makedirs(args.plot_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"{run_name}.csv")
    plot_path = os.path.join(args.plot_dir, f"{run_name}.png")
    try:
        plot_training_curves(csv_path, plot_path, title=f"{args.agent} on {args.env} (traffic={args.traffic})")
        print(f"Saved training curves to {plot_path}")
    except Exception as e:  # pragma: no cover - plotting is best-effort
        print(f"Plotting failed: {e}")

    print(f"Done. Logs: {csv_path}")
    print(f"Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Headless rollout recorder -- works in a display-less sandbox.

Renders episodes with `render_mode="rgb_array"` (SDL's dummy video driver
avoids needing a real display: `SDL_VIDEODRIVER=dummy`) and saves them as
MP4/GIF with an on-screen HUD, so a trained policy can be inspected without a
live window. Example:

    SDL_VIDEODRIVER=dummy python scripts/record.py \
        --checkpoint results/checkpoints/sac_racetrack_traffic-medium_seed0_final.pt \
        --traffic heavy --episodes 2 --out results/videos/sac_heavy.mp4

For a PPO checkpoint (trained via --agent ppo, see scripts/train.py), pass
--agent ppo and the matching `.zip` checkpoint (stable-baselines3's own save
format, not our `.pt` state_dict -- the two agents' `agent.load()` each know
how to read their own):

    SDL_VIDEODRIVER=dummy python scripts/record.py \
        --agent ppo \
        --checkpoint results/checkpoints/ppo_racetrack_traffic-medium_seed0_final.zip \
        --traffic heavy --episodes 2 --out results/videos/ppo_heavy.mp4

Same for --agent sac_sb3 (SAC via stable-baselines3, a second implementation
of SAC alongside our from-scratch one) -- also a `.zip` checkpoint:

    SDL_VIDEODRIVER=dummy python scripts/record.py \
        --agent sac_sb3 \
        --checkpoint results/checkpoints/sac_sb3_racetrack_traffic-medium_seed0_final.zip \
        --traffic heavy --episodes 2 --out results/videos/sac_sb3_heavy.mp4

IMPORTANT: if the checkpoint was trained with non-default env knobs that
change the observation size -- right now that's only --enable-delay-wrapper
-- this script must be given the SAME ones, or the checkpoint won't load (its
network input size won't match). The simplest way to guarantee that is to
record with the exact same --config file used for training:

    SDL_VIDEODRIVER=dummy python scripts/record.py \
        --config configs/racetrack_experiment_config.jsonc \
        --checkpoint results/checkpoints/<run_name>_final.pt \
        --out results/videos/best_run.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_platform.envs import make_env, available_envs
from rl_platform.agents import make_agent
from rl_platform.visualization import overlay_hud, save_video, save_gif
from rl_platform.config_loader import load_jsonc, experiment_config_to_arg_defaults
from rl_platform.cli_common import add_env_cli_args, env_kwargs_from_args, traffic_arg


def record_episode_with_hud(env, act_fn, seed, agent_label, traffic_label, ep_idx, n_episodes):
    obs, info = env.reset(seed=seed)
    frames = []
    ep_return = 0.0
    crashed = False
    speeds = []
    done, step = False, 0
    frame = env.render()
    frames.append(overlay_hud(frame, [
        f"agent: {agent_label}", f"traffic: {traffic_label}",
        f"episode {ep_idx + 1}/{n_episodes}  step 0", "return: 0.00",
    ]))
    while not done:
        action = act_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += float(reward)
        crashed = crashed or bool(info.get("crashed", False))
        if "speed" in info:
            speeds.append(float(info["speed"]))
        step += 1
        frame = env.render()
        frames.append(overlay_hud(frame, [
            f"agent: {agent_label}", f"traffic: {traffic_label}",
            f"episode {ep_idx + 1}/{n_episodes}  step {step}",
            f"return: {ep_return:.2f}",
            f"speed: {info.get('speed', float('nan')):.2f}",
            f"crashed: {crashed}",
        ]))
    summary = {
        "return": ep_return, "crashed": crashed,
        "mean_speed": float(sum(speeds) / len(speeds)) if speeds else float("nan"),
        "steps": step,
    }
    return frames, summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="racetrack", choices=available_envs())
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agent", default="sac")
    p.add_argument("--traffic", default="medium")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default="results/videos/rollout.mp4")
    p.add_argument("--gif", action="store_true", help="also save a .gif next to --out")
    add_env_cli_args(p)
    return p


def main(argv=None):
    # Same --config pre-parse pattern as scripts/train.py, so recording can
    # reuse the exact config a run was trained with (see module docstring).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, remaining = pre.parse_known_args(argv)

    parser = build_arg_parser()
    if pre_args.config:
        cfg = load_jsonc(pre_args.config)
        parser.set_defaults(**experiment_config_to_arg_defaults(cfg))
    args = parser.parse_args(remaining)

    env_kwargs = dict(
        traffic_density=traffic_arg(args.traffic),
        render_mode="rgb_array",
    )
    env_kwargs.update(env_kwargs_from_args(args, delay_seed=args.seed))

    env = make_env(args.env, **env_kwargs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = make_agent(args.agent, obs_dim=obs_dim, act_dim=act_dim)
    if args.checkpoint:
        agent.load(args.checkpoint)
        agent_label = os.path.basename(args.checkpoint)
    else:
        agent_label = f"untrained {args.agent}"

    all_frames = []
    for ep in range(args.episodes):
        frames, summary = record_episode_with_hud(
            env,
            lambda obs: agent.act(obs, deterministic=True),
            seed=args.seed + ep,
            agent_label=agent_label,
            traffic_label=args.traffic,
            ep_idx=ep,
            n_episodes=args.episodes,
        )
        all_frames.extend(frames)
        print(f"episode {ep}: return={summary['return']:.2f} crashed={summary['crashed']} "
              f"mean_speed={summary['mean_speed']:.2f} steps={summary['steps']}")

    save_video(all_frames, args.out, fps=args.fps)
    print(f"Saved video: {args.out}")
    if args.gif:
        gif_path = os.path.splitext(args.out)[0] + ".gif"
        save_gif(all_frames, gif_path, fps=args.fps)
        print(f"Saved gif: {gif_path}")
    env.close()


if __name__ == "__main__":
    main()

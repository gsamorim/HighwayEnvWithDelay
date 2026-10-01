#!/usr/bin/env python3
"""Traffic-density sweep: evaluate one or more trained agents across a range
of `other_vehicles` settings to see whether/how their driving behavior
adapts to traffic.

For each (agent, traffic level) pair we run several deterministic evaluation
episodes and record: mean return, crash rate, off-road-without-collision
rate, mean speed, mean episode length, and mean |acceleration| / |steering|
(a simple proxy for "cautious/smooth" vs. "aggressive/jerky" driving).

Example comparing three specialist agents across a traffic sweep:

    SDL_VIDEODRIVER=dummy python scripts/traffic_experiment.py \
        --checkpoints medium=results/checkpoints/sac_racetrack_medium_final.pt \
                       none=results/checkpoints/sac_racetrack_none_final.pt \
                       heavy=results/checkpoints/sac_racetrack_heavy_final.pt \
        --traffic-values 0,2,4,6,8,10,13,16 --episodes 8 \
        --out-csv results/experiments/traffic_sweep.csv \
        --out-plot results/plots/traffic_sweep.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl_platform.envs import make_env
from rl_platform.agents import make_agent
from rl_platform.training.plotting import plot_traffic_sweep


def evaluate_agent_at_traffic(agent, traffic_value: int, n_episodes: int, seed: int) -> dict:
    env = make_env("racetrack", traffic_density=traffic_value, render_mode="rgb_array")
    returns, crashes, offroads, speeds, lengths = [], [], [], [], []
    abs_accels, abs_steers = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return, ep_len = 0.0, 0
        ep_speeds, ep_accels, ep_steers = [], [], []
        crashed = False
        last_info = info
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_len += 1
            crashed = crashed or bool(info.get("crashed", False))
            if "speed" in info:
                ep_speeds.append(float(info["speed"]))
            abs_accels.append(abs(float(action[0])))
            ep_accels.append(abs(float(action[0])))
            abs_steers.append(abs(float(action[1])))
            ep_steers.append(abs(float(action[1])))
            last_info = info

        on_road_at_end = bool(last_info.get("rewards", {}).get("on_road_reward", True))
        offroad_without_crash = (not crashed) and (not on_road_at_end)

        returns.append(ep_return)
        crashes.append(float(crashed))
        offroads.append(float(offroad_without_crash))
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else float("nan"))
        lengths.append(ep_len)

    env.close()
    return {
        "n_vehicles": traffic_value,
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "crash_rate": float(np.mean(crashes)),
        "offroad_rate": float(np.mean(offroads)),
        "mean_speed": float(np.nanmean(speeds)),
        "mean_length": float(np.mean(lengths)),
        "mean_abs_accel": float(np.mean(abs_accels)) if abs_accels else float("nan"),
        "mean_abs_steer": float(np.mean(abs_steers)) if abs_steers else float("nan"),
    }


def parse_checkpoints(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not path:
            raise ValueError(f"--checkpoints entries must look like name=path, got '{pair}'")
        out[name] = path
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs, e.g. medium=ckpt.pt")
    p.add_argument("--agent", default="sac")
    p.add_argument("--traffic-values", default="0,2,4,6,8,10,13,16")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out-csv", default="results/experiments/traffic_sweep.csv")
    p.add_argument("--out-json", default="results/experiments/traffic_sweep.json")
    p.add_argument("--out-plot-prefix", default="results/plots/traffic_sweep")
    args = p.parse_args(argv)

    traffic_values = [int(v) for v in args.traffic_values.split(",")]
    checkpoints = parse_checkpoints(args.checkpoints)

    probe = make_env("racetrack", traffic_density=0)
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    all_rows = []
    for agent_name, ckpt_path in checkpoints.items():
        agent = make_agent(args.agent, obs_dim=obs_dim, act_dim=act_dim)
        agent.load(ckpt_path)
        print(f"\n=== Agent '{agent_name}' ({ckpt_path}) ===")
        for tv in traffic_values:
            stats = evaluate_agent_at_traffic(agent, tv, args.episodes, args.seed)
            stats["agent"] = agent_name
            stats["traffic"] = str(tv)
            all_rows.append(stats)
            print(
                f"  other_vehicles={tv:>2d}  return={stats['mean_return']:7.2f}±{stats['std_return']:5.2f}  "
                f"crash={stats['crash_rate']:.2f}  offroad={stats['offroad_rate']:.2f}  "
                f"speed={stats['mean_speed']:5.2f}  |accel|={stats['mean_abs_accel']:.2f}  |steer|={stats['mean_abs_steer']:.2f}"
            )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with open(args.out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved raw results to {args.out_csv} and {args.out_json}")

    for agent_name in checkpoints:
        rows = [r for r in all_rows if r["agent"] == agent_name]
        plot_path = f"{args.out_plot_prefix}_{agent_name}.png"
        plot_traffic_sweep(rows, plot_path, title=f"Effect of traffic density on '{agent_name}' policy")
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

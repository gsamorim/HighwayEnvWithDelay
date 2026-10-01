#!/usr/bin/env python3
"""Live pygame viewer for a trained agent.

Requires an actual display -- run this on your own machine (not in a
headless server/sandbox). Example:

    python scripts/watch.py --checkpoint results/checkpoints/sac_racetrack_traffic-medium_seed0_final.pt \
        --traffic heavy --episodes 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_platform.envs import make_env
from rl_platform.agents import make_agent
from rl_platform.visualization.pygame_viewer import watch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="path to a saved agent checkpoint; omit to watch a random agent")
    p.add_argument("--agent", default="sac")
    p.add_argument("--traffic", default="medium")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        traffic = int(args.traffic)
    except ValueError:
        traffic = args.traffic

    env = make_env("racetrack", traffic_density=traffic, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = make_agent(args.agent, obs_dim=obs_dim, act_dim=act_dim)
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        print("No checkpoint given -- watching an untrained/random agent.")

    def act_fn(obs):
        return agent.act(obs, deterministic=True)

    watch(
        env,
        act_fn,
        n_episodes=args.episodes,
        seed=args.seed,
        hud_extra={"agent": args.agent, "traffic": args.traffic},
    )
    env.close()


if __name__ == "__main__":
    main()

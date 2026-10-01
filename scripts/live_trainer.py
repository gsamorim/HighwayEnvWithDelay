#!/usr/bin/env python3
"""Continuous, closeable, resumable multi-agent live training dashboard.

Opens one pygame window showing every agent's racetrack panel side by side,
training all of them continuously while you watch. Closing the window (or
Ctrl+C) saves every agent's full state -- network weights, optimizer state,
replay buffer, and cumulative step/episode counters -- to
`results/live_checkpoints/<agent-name>/`. Running the exact same command
again picks each agent back up from where it left off; it does not restart
training from scratch.

Run on a machine with a real display for the interactive experience:

    python scripts/live_trainer.py --agents medium=sac:medium heavy=sac:heavy none=sac:none

Press **F11** anytime to toggle windowed fullscreen (a borderless window
sized to your screen resolution, with the dashboard scaled up and
letterboxed to fit -- not exclusive/"real" fullscreen, so switching away and
back doesn't fight your window manager). Pass `--fullscreen` to start in
that mode. Each agent panel always shows its current acceleration and
steering as live gauge bars, in addition to the numeric stats.

In a headless sandbox (no display), pygame still runs against a dummy
surface, so the same command trains just as well -- you just can't see the
window. Use `--record path.mp4` to save a timelapse of the dashboard instead,
and `--max-runtime-seconds 60` / `--max-ticks N` to give a run a defined end
point (useful for demos/CI; omit both to run forever until you close the
window or Ctrl+C).

Examples:

    # Fresh 3-agent run, one specialist per traffic level, checkpointed as it goes
    SDL_VIDEODRIVER=dummy python scripts/live_trainer.py \\
        --agents medium=sac:medium heavy=sac:heavy none=sac:none \\
        --record results/videos/live_dashboard.mp4 --max-runtime-seconds 60

    # Re-run the same command later: agents resume, they don't restart
    SDL_VIDEODRIVER=dummy python scripts/live_trainer.py \\
        --agents medium=sac:medium heavy=sac:heavy none=sac:none

    # Warm-start a live agent from one of the earlier batch-trained checkpoints
    python scripts/live_trainer.py \\
        --agents medium=sac:medium \\
        --warm-start medium=results/checkpoints/sac_racetrack_medium_v2_final.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_platform.training.live_app import AgentSpec, LiveTrainerApp


def _traffic_arg(v: str):
    try:
        return int(v)
    except ValueError:
        return v


def parse_agent_spec(raw: str) -> AgentSpec:
    """Parses `name=[agent_type:]traffic`, e.g.:
        medium=medium            -> sac agent, medium traffic, name "medium"
        heavy=sac:heavy          -> sac agent, heavy traffic, name "heavy"
        baseline=random:medium   -> random-baseline agent, medium traffic
    """
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--agents entries must look like name=traffic, got '{raw}'")
    name, _, rhs = raw.partition("=")
    if ":" in rhs:
        agent_type, _, traffic = rhs.partition(":")
    else:
        agent_type, traffic = "sac", rhs
    return AgentSpec(name=name, traffic=_traffic_arg(traffic), agent_type=agent_type)


def parse_warm_start(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"--warm-start entries must look like name=path, got '{pair}'")
        out[name] = path
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--agents", nargs="+", default=["medium=sac:medium", "heavy=sac:heavy", "none=sac:none"],
        help="name=[agent_type:]traffic entries, e.g. medium=sac:medium heavy=sac:heavy",
    )
    p.add_argument("--warm-start", nargs="*", default=[], help="name=path entries to seed weights from an existing flat checkpoint")
    p.add_argument("--checkpoint-root", default="results/live_checkpoints")
    p.add_argument("--steps-per-frame", type=int, default=1, help="env+train steps per rendered frame, per agent. Higher = faster wall-clock "
                   "training throughput but a lower actual frame rate (each step includes a real gradient update, "
                   "which is the dominant per-frame cost with multiple agents on CPU) -- raise this if you care more "
                   "about training speed than a smooth-looking window")
    p.add_argument("--fps", type=int, default=60, help="target/cap frame rate. The ACTUAL rate is printed to the console every few seconds "
                   "([perf] ... fps actual) and, with multiple CPU-trained agents, will typically be well under this -- "
                   "that's real compute cost (training + rendering per frame), not a display bug; lower --steps-per-frame, "
                   "run fewer --agents, or reduce --panel-size to raise it")
    p.add_argument("--panel-size", type=int, default=480, help="pixel width of each agent's panel in the grid (video + HUD strip below it)")
    p.add_argument("--fullscreen", action="store_true", help="start in windowed-fullscreen mode (borderless, matches your screen resolution); toggle anytime with F11")
    p.add_argument("--autosave-every-steps", type=int, default=2000, help="periodic safety autosave per agent; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--start-steps", type=int, default=1_000, help="pure-random exploration steps before an agent starts using its own policy")
    p.add_argument("--high-speed-reward", type=float, default=1.2)
    p.add_argument("--collision-reward", type=float, default=-3.0)
    p.add_argument("--action-reward", type=float, default=-0.05)
    p.add_argument("--max-ticks", type=int, default=None, help="stop after this many rendered frames (omit to run until closed)")
    p.add_argument("--max-runtime-seconds", type=float, default=None, help="stop after this much wall-clock time (omit to run until closed)")
    p.add_argument("--record", default=None, help="also save a timelapse video of the dashboard to this path")
    p.add_argument("--record-every-n-frames", type=int, default=1)
    args = p.parse_args(argv)

    specs = [parse_agent_spec(a) for a in args.agents]
    warm_start = parse_warm_start(args.warm_start)
    for spec in specs:
        spec.warm_start_path = warm_start.get(spec.name)
        spec.start_steps = args.start_steps
        spec.agent_kwargs = dict(
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
        )

    app = LiveTrainerApp(
        specs,
        checkpoint_root=args.checkpoint_root,
        steps_per_frame=args.steps_per_frame,
        fps=args.fps,
        autosave_every_steps=args.autosave_every_steps,
        panel_size=args.panel_size,
        seed=args.seed,
        env_kwargs=dict(
            high_speed_reward=args.high_speed_reward,
            collision_reward=args.collision_reward,
            action_reward=args.action_reward,
        ),
    )
    app.run(
        max_ticks=args.max_ticks,
        max_runtime_seconds=args.max_runtime_seconds,
        record_path=args.record,
        record_every_n_frames=args.record_every_n_frames,
        start_fullscreen=args.fullscreen,
    )


if __name__ == "__main__":
    main()

"""Shared CLI plumbing for scripts that build a racetrack env from flags.

Keeps every script that builds an env (train.py, record.py, and any future
one) wired to IDENTICAL flag names/semantics for the knobs added on top of
the native racetrack factory, so a --config jsonc or a set of CLI flags
behaves exactly the same whether you're training or recording. This matters
in particular for `enable_delay_wrapper`: it changes the observation's
dimensionality, so a checkpoint trained with it enabled can only be loaded
into an env built with the same setting.
"""
from __future__ import annotations

import argparse


def add_env_cli_args(parser: argparse.ArgumentParser) -> None:
    """Adds every env-construction flag beyond --env/--traffic (those two
    are added by each script itself, since their defaults/choices differ
    slightly between train.py and record.py)."""
    parser.add_argument(
        "--env-id", default=None,
        help="override the native HighwayEnv gym id, e.g. racetrack-v0, racetrack-v1, "
             "racetrack-large-v0/v1, racetrack-oval-v0/v1 (default depends on --env)",
    )
    parser.add_argument(
        "--oval-straight-length-m", type=int, default=None,
        help="native `length` (oval env_id only); default ~100 if unset",
    )
    parser.add_argument(
        "--oval-obstacle-preset", default=None,
        help="native oval obstacle preset (oval env_id only): block_lane | force_decision",
    )
    parser.add_argument("--traffic-speed-min", type=float, default=None)
    parser.add_argument("--traffic-speed-max", type=float, default=None)
    parser.add_argument(
        "--enable-delay-wrapper", action="store_true",
        help="wrap the env with DelayedActionMDPWrapper (adds action-history/"
             "pending-queue/remaining-time/last-action dims to the observation -- "
             "must match between the run that trained a checkpoint and any run "
             "that loads it, e.g. for recording)",
    )
    parser.add_argument("--action-delay-min-steps", type=int, default=0)
    parser.add_argument("--action-delay-max-steps", type=int, default=0)
    parser.add_argument("--delay-known-to-agent", action="store_true")
    parser.add_argument("--observation-delay-steps", type=int, default=0)


def traffic_arg(v: str):
    try:
        return int(v)
    except ValueError:
        return v


def env_kwargs_from_args(args: argparse.Namespace, *, delay_seed=None) -> dict:
    """Builds the subset of RacetrackEnvConfig overrides that came from the
    flags added by `add_env_cli_args`, omitting anything left at its
    null/False default so make_env(...)'s own defaults still apply."""
    kwargs: dict = {}
    if getattr(args, "env_id", None) is not None:
        kwargs["env_id"] = args.env_id
    if getattr(args, "oval_straight_length_m", None) is not None:
        kwargs["oval_straight_length_m"] = args.oval_straight_length_m
    if getattr(args, "oval_obstacle_preset", None) is not None:
        kwargs["oval_obstacle_preset"] = args.oval_obstacle_preset
    if getattr(args, "traffic_speed_min", None) is not None or getattr(args, "traffic_speed_max", None) is not None:
        kwargs["traffic_speed_min"] = args.traffic_speed_min
        kwargs["traffic_speed_max"] = args.traffic_speed_max
    if getattr(args, "enable_delay_wrapper", False):
        kwargs.update(
            enable_delay_wrapper=True,
            action_delay_min_steps=args.action_delay_min_steps,
            action_delay_max_steps=args.action_delay_max_steps,
            delay_known_to_agent=args.delay_known_to_agent,
            observation_delay_steps=args.observation_delay_steps,
            delay_seed=delay_seed,
        )
    return kwargs

# HighwayEnv Racetrack + SAC

A from-scratch Soft Actor-Critic (SAC) implementation trained on
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)'s `racetrack-v0`
environment, built around a small plugin architecture so new environments and
new agent architectures can be dropped in later without touching the
training loop.

```
rl_platform/
  envs/
    registry.py       # name -> factory function; register_env("foo", make_foo_env)
    racetrack.py       # racetrack-v0 factory + config (traffic presets, obs type, reward shaping)
    wrappers.py         # FlattenObservation, RewardScaler, ForceRenderEnabled
  agents/
    base.py             # Agent ABC: act / observe / update / save / load
    sac.py               # SACAgent (the algorithm)
    networks.py          # GaussianPolicy (tanh-squashed actor), TwinQNetwork
    replay_buffer.py      # FIFO replay buffer
    random_agent.py        # trivial baseline / reference Agent implementation
    registry.py             # name -> factory function; register_agent("ppo", make_ppo_agent)
  training/
    trainer.py           # off-policy training loop (batch mode), agent-agnostic
    live_app.py            # continuous multi-agent training loop + pygame dashboard (interactive mode)
    checkpointing.py         # save/load a full training-state bundle (weights+optimizer+replay buffer+meta)
    logger.py                 # CSV + JSON run logging
    plotting.py                 # learning-curve and traffic-sweep plots
  visualization/
    overlay.py            # HUD text overlay on rendered frames
    grid.py                 # compose N agent panels into one dashboard image
    video_export.py           # headless rollout -> MP4/GIF (for sandboxes without a display)
    pygame_viewer.py             # live pygame window (run locally, needs a real display)
scripts/
  train.py               # batch-train one agent on an env from the registry, then stop
  live_trainer.py          # *run* multiple agents at once in a pygame dashboard, continuously,
                           # resuming from disk if you stop and restart it (see below)
  record.py               # export a rollout video/gif headlessly
  watch.py                 # live pygame viewer of one already-trained agent (local machine only)
  traffic_experiment.py     # evaluate agent(s) across a traffic-density sweep
tests/                    # pytest suite: env, agent, training loop, live app, video export
results/                  # logs / checkpoints / plots / videos / experiments (generated)
```

## Why this architecture

- **Environments are a registry, not a hardcoded import.** `rl_platform.envs.registry.make_env("racetrack", ...)`
  looks up a factory function. Adding HighwayEnv's `intersection-v0`, `merge-v0`,
  or an entirely different simulator later is one `register_env(...)` call away, and
  nothing in `training/` or `scripts/` needs to change.
- **Agents share one interface** (`Agent.act/observe/update/save/load`) and are
  looked up the same way via `rl_platform.agents.registry.make_agent("sac", ...)`.
  The `Trainer` only calls those five methods, so a PPO, TD3, or DreamerV3 agent
  registered as `"ppo"`/`"td3"`/... becomes selectable with `--agent ppo` in
  `scripts/train.py`, and `traffic_experiment.py` can then pit them against SAC
  on the identical traffic sweep.
- **Observations are flattened generically** (`FlattenObservation` wrapper)
  so the MLP-based SAC agent doesn't need to know about HighwayEnv's
  `(vehicles, features)` observation shape; a future CNN-based agent can instead
  request `flatten_obs=False` and consume the raw grid.

## The environment: `racetrack-v0`

Continuous control, closed-loop track, ego vehicle + IDM-driven traffic.

- **Action space** (2D continuous, `[-1, 1]^2`): **acceleration** and
  **steering**. HighwayEnv's default racetrack config actually only exposes
  steering (throttle is driven by a discrete target-speed selector); we
  explicitly turn on `longitudinal=True` in `RacetrackEnvConfig` so the agent
  controls both, per the brief.
- **Observation space**: `Kinematics` — for each of the `vehicles_count`
  nearest vehicles (ego + neighbors), `[presence, x, y, vx, vy, heading,
  long_off, lat_off]` in the ego's frame, normalized, flattened to a vector.
  (`OccupancyGrid`, HighwayEnv's default for this env, is also supported via
  `observation_type="OccupancyGrid"` for a future CNN agent.)
- **Reward**: HighwayEnv's built-in lane-centering + collision + action-cost
  terms (see `RacetrackEnvConfig.to_highway_config` for the exact weights used).
- **Traffic density** — the main environment knob this project investigates —
  maps to HighwayEnv's `other_vehicles` config key (an *upper bound*: the
  actual number of IDM vehicles spawned each episode is randint(0,
  other_vehicles), plus one guaranteed lead car when > 0). Presets:

  | preset  | other_vehicles |
  |---------|----------------|
  | none    | 0              |
  | light   | 2              |
  | medium  | 5              |
  | heavy   | 10             |
  | jam     | 16             |

  Pass any of these names or a raw integer as `--traffic`.

## The agent: SAC

Standard off-policy, maximum-entropy actor-critic
([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)) with automatic
temperature tuning ([Haarnoja et al. 2019](https://arxiv.org/abs/1812.05905)):
tanh-squashed Gaussian policy, clipped double-Q critics with Polyak-averaged
targets, entropy coefficient learned to hit a target entropy of `-act_dim`.
Implemented from scratch in ~200 lines of PyTorch (`rl_platform/agents/sac.py`)
rather than pulled from a larger RL library, so the whole algorithm is
auditable in one file.

## Usage

```bash
pip install -r requirements.txt

# Train (headless-safe; SDL_VIDEODRIVER=dummy avoids needing a display)
SDL_VIDEODRIVER=dummy python scripts/train.py --traffic medium --total-steps 50000

# Record a rollout as MP4/GIF (works in a sandbox with no display)
SDL_VIDEODRIVER=dummy python scripts/record.py \
    --checkpoint results/checkpoints/sac_racetrack_traffic-medium_seed0_final.pt \
    --traffic heavy --episodes 2 --out results/videos/demo.mp4 --gif

# Watch live in a real pygame window (run on your own machine, not headless)
python scripts/watch.py --checkpoint results/checkpoints/sac_racetrack_traffic-medium_seed0_final.pt --traffic heavy

# Sweep traffic density for one or more trained agents
SDL_VIDEODRIVER=dummy python scripts/traffic_experiment.py \
    --checkpoints medium=results/checkpoints/sac_racetrack_medium_final.pt \
    --traffic-values 0,2,4,6,8,10,13,16 --episodes 8

# Tests
SDL_VIDEODRIVER=dummy pytest tests/ -v
```

## Continuous multi-agent live training (`scripts/live_trainer.py`)

`scripts/train.py` is *batch* mode: it runs for a fixed number of steps and
stops. `scripts/live_trainer.py` is a different mode entirely: a pygame
window that trains **several agents at once**, continuously, for as long as
you leave it open, and picks every agent back up from exactly where it left
off if you close the window and run the same command again later.

```bash
# Fresh run: one specialist per traffic level, all three training at once,
# rendered side by side in one window.
python scripts/live_trainer.py --agents medium=sac:medium heavy=sac:heavy none=sac:none

# ... let it run, watch the dashboard, close the window (or Ctrl+C) whenever ...

# Run the exact same command again later: each agent resumes -- its network
# weights, optimizer momentum, replay buffer, and cumulative step/episode
# counters were all persisted, so this continues training rather than
# starting over.
python scripts/live_trainer.py --agents medium=sac:medium heavy=sac:heavy none=sac:none
```

Each agent gets its own environment instance and trains independently every
frame (`--steps-per-frame` env+gradient steps between each render, so
training speed isn't capped by your monitor's refresh rate); the window
shows all of them in a grid (with a gutter between panels, not jammed
together, and biased toward a landscape layout -- e.g. the default 3 agents
render as one row, not a taller-than-wide 2x2), each with a fixed-size
dashboard strip below its video feed -- cumulative env steps, episode
count, last return, speed, crash count, entropy temperature, a color-coded
traffic badge, and two always-on gauge bars showing the agent's *current*
acceleration and steering action every frame (fill direction/color show
throttle-vs-brake and left-vs-right at a glance, the exact value is printed
next to each gauge). The HUD strip is a fixed size and solid (not a
translucent overlay on the video), which is what keeps it from jittering
frame to frame. Panels default to 480px wide (`--panel-size` to change).

**The window always fits your screen.** On launch it checks your actual
screen resolution and, if the dashboard's native size wouldn't fit, renders
everything (panels, fonts, gauges) at a smaller size that does -- rather
than opening oversized or resampling an oversized image every frame.
Fullscreen (below) fills whatever display you're on.

**Display mode**: the window is created with `pygame.SCALED`, which puts
rendering on SDL2's actual presentation path with real vsync, instead of
the legacy raw-surface path (which on Windows typically has no real vsync
and can show tearing). If a window still opens bigger than your screen, or
still visibly tears/glitches (not just feels slow -- see below), that's a
bug -- please say exactly what it looks like (constant vs. occasional, the
whole window vs. part of it, a screenshot if possible).

**Actual frame rate**: `--fps` (default 60) is a cap, not a guarantee. Every
few seconds the console prints the real measured rate, e.g. `[perf] 11.4 fps
actual (target 60)`. With multiple agents training simultaneously on CPU,
that number will usually be well under 60 -- each rendered frame includes a
real SAC gradient update per agent (measured on the dev machine at ~8-9ms
each) plus HighwayEnv's own drawing and this dashboard's HUD (another
~15-18ms per agent), so 3 agents can easily add up to 100ms+ of real work
per frame. That's genuine compute cost, not a display-layer bug, and no
`pygame.display` flag can make it faster. What actually helps, in order of
impact: lower `--steps-per-frame` (default 1; the training side of the
per-frame cost scales directly with it), run fewer `--agents` at once,
reduce `--panel-size`. If the console number itself looks wrong (e.g. it
says 55 fps but the window still visibly stutters/tears), that points back
at the display layer and is worth reporting with the exact `[perf]` line.

While it's running: **S** saves immediately without quitting, **SPACE**
pauses/resumes every agent, **1**-**9** pauses/resumes one agent by its
position in `--agents`, **F11** toggles fullscreen (pass `--fullscreen` to
start in that mode), and closing the window
(or **ESC**, or Ctrl+C, or a periodic autosave every `--autosave-every-steps`)
saves everyone.

Each agent's full state lives in `results/live_checkpoints/<name>/`:
`model.pt` (network + optimizer weights + gradient-step counter),
`buffer.npz` (replay buffer contents), `meta.json` (cumulative env
steps/episodes/crashes + when it was last saved). You can also warm-start a
live agent from one of the flat, eval-only checkpoints produced by
`scripts/train.py`:

```bash
python scripts/live_trainer.py --agents medium=sac:medium \
    --warm-start medium=results/checkpoints/sac_racetrack_medium_v2_final.pt
```

In a display-less sandbox (like the one this was developed in),
`SDL_VIDEODRIVER=dummy python scripts/live_trainer.py ...` still runs and
trains correctly against a dummy surface -- you just can't see the window.
Add `--record path.mp4` to save a timelapse of the dashboard instead, and
`--max-runtime-seconds 60` (or `--max-ticks N`) to give a headless/CI run a
defined stopping point (omit both to run forever until closed).

## A HighwayEnv rendering quirk we worked around

HighwayEnv's default pygame `EnvViewer` silently disables *all* drawing
(returning solid-black `render()` frames, no error) whenever
`SDL_VIDEODRIVER=dummy` is set — see `EnvViewer.__init__`. That env var is
exactly what you need to set to render headlessly in a container with no
display. `rl_platform/envs/wrappers.py::ForceRenderEnabled` detects this and
re-enables drawing after the viewer is lazily constructed, which is what
makes `scripts/record.py` (and the video-export tests) actually produce
non-blank footage in a sandbox. This is applied automatically by
`make_racetrack_env` whenever a `render_mode` is requested.

## Results & notes

See `RESULTS.md` for training curves, the traffic-density sweep findings,
and behavioral observations, and `results/` for the raw logs, checkpoints,
plots, and demo videos.

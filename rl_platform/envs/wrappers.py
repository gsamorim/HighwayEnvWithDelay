"""Generic observation wrappers usable by any environment in the registry."""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenObservation(gym.ObservationWrapper):
    """Flattens an arbitrary-shaped Box observation into a 1D vector.

    HighwayEnv's Kinematics/OccupancyGrid observations are returned as
    (N, F) or (C, H, W) arrays. Most simple MLP-based agents (like the SAC
    implementation here) expect a flat vector, so this wrapper reshapes the
    observation space accordingly. Kept separate from the agent so that a
    future CNN-based agent can instead consume the unflattened observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        flat_dim = int(np.prod(env.observation_space.shape))
        low = env.observation_space.low.reshape(-1)
        high = env.observation_space.high.reshape(-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)


class ForceRenderEnabled(gym.Wrapper):
    """Work around a HighwayEnv quirk: its `EnvViewer` silently disables all
    drawing (leaving `render()` return solid-black frames) whenever
    `SDL_VIDEODRIVER=dummy` is set -- see
    `highway_env.envs.common.graphics.EnvViewer.__init__`. That env var is
    exactly what a display-less cloud sandbox needs to set so pygame doesn't
    fail to open a display, so without this wrapper `render_mode="rgb_array"`
    silently produces empty video/GIF frames when running headless.

    We let the viewer lazily construct itself as usual, then flip its
    `enabled` flag back on so it actually draws the road/vehicles onto its
    offscreen surface. Harmless (and a no-op) when a real display driver is
    in use, e.g. when `scripts/watch.py` is run locally.
    """

    def render(self):
        viewer = getattr(self.unwrapped, "viewer", None)
        if viewer is None:
            # First call: let HighwayEnv construct (and possibly disable) it.
            self.env.render()
            viewer = getattr(self.unwrapped, "viewer", None)
        if viewer is not None and not viewer.enabled:
            viewer.enabled = True
        return self.env.render()


class RewardScaler(gym.RewardWrapper):
    """Optionally rescales the environment reward, useful for stabilizing SAC's
    entropy-regularized objective across environments with very different
    reward magnitudes."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class HighSpeedRewardWrapper(gym.Wrapper):
    """Adds an explicit speed incentive to `racetrack-v0`.

    This fixes a real gap in HighwayEnv's `RacetrackEnv._rewards`: unlike
    `highway-v0`/`intersection-v0`, it has no speed term at all -- only
    `lane_centering_reward`, `action_reward` (which *penalizes* the norm of
    the action, i.e. penalizes accelerating/steering at all), `collision_reward`,
    and the `on_road_reward` gate. A `high_speed_reward` key in the env config
    is silently ignored because `RacetrackEnv` never looks it up. Without this
    wrapper, an SAC agent trained here has no incentive to move quickly at
    all and reliably converges to crawling around the track at low speed --
    which we observed empirically before adding this fix (see RESULTS.md).

    Two interchangeable shapes for the bonus, both gated to 0 off-road
    (consistent with the base env's own on-road gating) so the agent can't
    farm the speed bonus by cutting corners off-track:

    - ``shape="linear"`` (default, original/unchanged behavior): adds
      ``coef * clip(speed / speed_limit, 0, 1.2)``. Monotonic ramp with two
      hard elbows (floor at 0 -- so standing still and reversing are both
      exactly neutral, never penalized -- and a soft ceiling at 1.2x), no
      diminishing returns below the ceiling.
    - ``shape="tanh"``: adds ``coef * tanh((speed - v0) / steepness)``, an
      S-curve matching the discouraging-crawling / rewarding-driving /
      diminishing-returns-at-high-speed shape requested explicitly in
      conversation: a small, bounded NEGATIVE bonus for standing still or
      reversing (both saturate near -coef rather than growing unbounded),
      a steep rise through zero just past ``v0``, and a flattening
      (saturating near +coef, not linearly unbounded) approach to the top
      speed range -- instead of a flat 0 below the limit and a hard ceiling
      above it. ``v0`` and ``steepness`` default to fractions of
      ``speed_limit`` (``v0_frac``/``steepness_frac``) so they scale
      automatically with whatever speed_limit a config uses.
    """

    def __init__(
        self,
        env: gym.Env,
        coef: float,
        speed_limit: float,
        shape: str = "linear",
        v0_frac: float = 0.1,
        steepness_frac: float = 0.3,
    ):
        super().__init__(env)
        if shape not in ("linear", "tanh"):
            raise ValueError(f"Unknown HighSpeedRewardWrapper shape: {shape!r}")
        self.coef = coef
        self.speed_limit = speed_limit
        self.shape = shape
        self.v0 = v0_frac * speed_limit
        self.steepness = steepness_frac * speed_limit

    def _speed_bonus(self, speed: float) -> float:
        if self.shape == "linear":
            speed_frac = float(np.clip(speed / self.speed_limit, 0.0, 1.2))
            return self.coef * speed_frac
        # shape == "tanh"
        return self.coef * float(np.tanh((speed - self.v0) / self.steepness))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        on_road = info.get("rewards", {}).get("on_road_reward", True)
        speed = info.get("speed", 0.0)
        bonus = self._speed_bonus(speed) if on_road else 0.0
        info = dict(info)
        info["speed_bonus"] = bonus
        return obs, reward + bonus, terminated, truncated, info


# --- Delay-robust SAC additions (RDMDP-style augmented state) ---------------
#
# Both wrappers below are OURS: they sit entirely outside the native
# HighwayEnv classes (RacetrackEnv / RacetrackEnvOval), composing via
# gym.Wrapper exactly like the ones above. No native class is subclassed,
# patched, or otherwise modified -- the native env config dict (built in
# racetrack.py::to_highway_config) is still what drives the simulation.

MAX_DELAY_STEPS = 6
# Fixed buffer capacity. Kept constant across every experiment variation
# (including delay=0 baselines that opt into the wrapper) so the agent's
# input dimension never changes between training and evaluation, even when
# training samples a narrower delay range than evaluation uses.


class TrafficSpeedWrapper(gym.Wrapper):
    """Randomizes IDM traffic vehicles' speed each episode.

    `other_vehicles`' *initial* speed is not an exposed native config key
    (HighwayEnv spawns them at the track's own default), so this wrapper
    overrides it right after `reset()`, turning traffic speed into a
    first-class experiment axis independent of traffic *intensity*
    (`other_vehicles` / `traffic_density`, handled natively).
    """

    def __init__(self, env: gym.Env, speed_min: float, speed_max: float):
        super().__init__(env)
        if speed_min > speed_max:
            raise ValueError(f"speed_min ({speed_min}) must be <= speed_max ({speed_max})")
        self.speed_min = speed_min
        self.speed_max = speed_max

    def _randomize_traffic_speed(self):
        road = getattr(self.unwrapped, "road", None)
        if road is None:
            return
        controlled_ids = {id(v) for v in getattr(self.unwrapped, "controlled_vehicles", [])}
        rng = getattr(self.unwrapped, "np_random", np.random)
        for vehicle in road.vehicles:
            if id(vehicle) in controlled_ids:
                continue
            speed = float(rng.uniform(self.speed_min, self.speed_max))
            vehicle.speed = speed
            if hasattr(vehicle, "target_speed"):
                vehicle.target_speed = speed

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._randomize_traffic_speed()
        return obs, info


class DelayedActionMDPWrapper(gym.Wrapper):
    """Augments the observation so a flat-vector MLP agent can learn a
    delay-robust policy (RDMDP-style; Bouteiller et al. 2020, "Reinforcement
    Learning with Random Delays"), using a *fixed-capacity* command queue so
    train and eval always share the same input size.

    Must be applied AFTER the observation is already a flat 1D Box (i.e.
    after `FlattenObservation`): it concatenates its own extra features onto
    that vector rather than returning a `Dict`, to match this repo's
    MLP-based SAC (`obs_dim = env.observation_space.shape[0]`).

    Augmented vector layout (in this order):
        [ base_obs | action_history (capacity x act_dim)
                   | pending_queue   (capacity x act_dim)
                   | pending_remaining_time (capacity,)
                   | last_executed_action (act_dim,) ]
    A human-readable (non-flattened) breakdown of the same data is also put
    in `info["delay_debug"]` every step/reset, for logging/analysis.

    Regras da fila de comandos (R1-R7)
    -----------------------------------
    R1. Ao enviar uma acao no passo `send_step`, sorteia-se um atraso em
        `[min_delay_steps, max_delay_steps]` e fixa-se `apply_step =
        send_step + atraso` no momento do envio (nao e re-sorteado depois).
    R2. Quando `delay_known_to_agent=True`, `pending_queue` e ordenado por
        `apply_step` crescente (ordem de aplicacao -- informacao legitima,
        ja que o tempo restante tambem e revelado). Quando
        `delay_known_to_agent=False`, `pending_queue` e ordenado por
        `send_step` (ordem de despacho, que o agente ja conhece por si so)
        -- NUNCA por `apply_step` nesse modo, pois isso vazaria a ordem
        relativa dos atrasos mesmo com `pending_remaining_time` zerado.
    R3. A cada `step()`, resolvem-se todos os comandos "devidos"
        (`apply_step <= current_step`).
    R4. Em caso de mais de um devido no mesmo passo (reordenacao), vence o
        de maior `send_step` (o mais fresco); os demais devidos sao
        descartados. Se mesmo o mais fresco tiver `send_step` menor que
        `last_applied_send_step`, ele tambem e descartado por inteiro
        (comando obsoleto chegando fora de ordem).
    R5. Se nenhum comando sobrevive no passo, repete-se a ultima acao
        executada (hold).
    R6. Invariante de capacidade: `len(_in_flight)` nunca pode exceder
        `capacity`; isso indicaria um atraso configurado alem do que o
        buffer suporta, e dispara `RuntimeError` imediatamente (falha alto,
        nunca silenciosamente trunca a fila).
    R7. `action_history` (acoes enviadas, mais recentes) e independente de
        `pending_queue` (comandos ainda nao aplicados) -- a primeira e
        sempre preenchida a cada passo, mesmo quando o atraso sorteado e 0.

    Reset e reprodutibilidade do RNG de atraso
    -------------------------------------------
    `env.reset(seed=s)` reinicializa o fluxo de sorteios de atraso a partir
    de `s` (via `SeedSequence(s).spawn(1)`) sempre que o wrapper NAO recebeu
    seed propria no construtor -- assim uma avaliacao com `--seed 41`
    reproduz tanto o spawn/trafego quanto a sequencia de atrasos sorteados.
    Um `reset()` sem seed explicita NAO mexe no RNG -- ele da continuidade
    ao fluxo, para que resets repetidos durante o treino nao repliquem a
    mesma sequencia de atrasos todo episodio. Se o wrapper foi construido
    com uma seed propria (`seed=...`), ela manda sempre e `reset(seed=...)`
    nao a sobrescreve.
    """

    def __init__(
        self,
        env: gym.Env,
        min_delay_steps: int = 0,
        max_delay_steps: int = 0,
        delay_known_to_agent: bool = False,
        capacity: int = MAX_DELAY_STEPS,
        obs_delay_steps: int = 0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if not (0 <= min_delay_steps <= max_delay_steps <= capacity):
            raise ValueError(
                f"Require 0 <= min_delay_steps ({min_delay_steps}) <= "
                f"max_delay_steps ({max_delay_steps}) <= capacity ({capacity})"
            )
        if obs_delay_steps < 0:
            raise ValueError(f"obs_delay_steps must be >= 0 (got {obs_delay_steps})")
        if not (isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1):
            raise TypeError(
                "DelayedActionMDPWrapper expects an already-flat 1D Box observation "
                "-- apply it after FlattenObservation."
            )
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("DelayedActionMDPWrapper expects a continuous Box action space.")

        self.min_delay_steps = min_delay_steps
        self.max_delay_steps = max_delay_steps
        self.delay_known_to_agent = delay_known_to_agent
        self.capacity = capacity
        # NOTE: observation delay is a SIMPLE fixed-length FIFO, with no
        # reordering support (unlike action delay). Observations arrive from
        # a single source in env order, so there is no "out of order" case
        # to resolve the way there is for actions sent over a channel.
        # Whether this should eventually also support reordering is an open
        # pendencia -- see the project doc.
        self.obs_delay_steps = obs_delay_steps

        # Remember whether an explicit seed was given to the WRAPPER itself
        # (as opposed to the env's reset(seed=...)) -- see reset() below for
        # why this distinction matters for reproducibility.
        self._explicit_seed = seed
        self._rng = np.random.default_rng(seed)
        self._act_dim = int(env.action_space.shape[0])
        action_low = env.action_space.low
        action_high = env.action_space.high

        extra_low = np.concatenate([
            np.tile(action_low, self.capacity),          # action_history
            np.tile(action_low, self.capacity),          # pending_queue
            np.zeros(self.capacity, dtype=np.float32),   # pending_remaining_time
            action_low,                                  # last_executed_action
        ])
        extra_high = np.concatenate([
            np.tile(action_high, self.capacity),
            np.tile(action_high, self.capacity),
            np.full(self.capacity, float(self.capacity), dtype=np.float32),
            action_high,
        ])
        low = np.concatenate([env.observation_space.low, extra_low]).astype(np.float32)
        high = np.concatenate([env.observation_space.high, extra_high]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._reset_internal_state()

    def _reset_internal_state(self):
        self._current_step = 0
        self._in_flight: list[dict[str, Any]] = []
        self._action_history = np.zeros((self.capacity, self._act_dim), dtype=np.float32)
        self._last_executed_action = np.zeros(self._act_dim, dtype=np.float32)
        self._last_applied_send_step = -1
        self._obs_queue: "deque[np.ndarray]" = deque(maxlen=self.obs_delay_steps + 1)

    def _sample_delay(self) -> int:
        if self.min_delay_steps == self.max_delay_steps:
            return self.min_delay_steps
        return int(self._rng.integers(self.min_delay_steps, self.max_delay_steps + 1))

    def _push_action_history(self, action: np.ndarray):
        self._action_history = np.roll(self._action_history, shift=-1, axis=0)
        self._action_history[-1] = action

    def _resolve_action_to_apply(self, current_step: int):
        """Implements R3-R5. Mutates `self._in_flight` (removes resolved
        commands) and `self._last_executed_action`/`_last_applied_send_step`
        when a command is actually applied."""
        due = [cmd for cmd in self._in_flight if cmd["apply_step"] <= current_step]
        discarded = 0
        action_to_apply = self._last_executed_action
        applied_send_step = None
        if due:
            winner = max(due, key=lambda c: c["send_step"])  # R4: freshest wins
            discarded += len(due) - 1
            if winner["send_step"] < self._last_applied_send_step:
                # Even the freshest due command is stale: discard it too, hold.
                discarded += 1
            else:
                action_to_apply = winner["action"]
                applied_send_step = winner["send_step"]
            self._in_flight = [cmd for cmd in self._in_flight if cmd["apply_step"] > current_step]
        if applied_send_step is not None:
            self._last_executed_action = action_to_apply
            self._last_applied_send_step = max(self._last_applied_send_step, applied_send_step)
        return action_to_apply, discarded

    def _build_pending_arrays(self, current_step: int):
        # R6: checked HERE (post-resolution), not right after scheduling --
        # a command whose apply_step == current_step is resolved (and
        # removed from _in_flight) in this same step, so the steady-state
        # invariant is "at most `capacity` commands survive resolution",
        # which is exactly satisfiable when max_delay_steps <= capacity.
        if len(self._in_flight) > self.capacity:
            raise RuntimeError(
                f"DelayedActionMDPWrapper capacity invariant violated: "
                f"{len(self._in_flight)} commands still in flight after resolution > "
                f"capacity {self.capacity}. max_delay_steps ({self.max_delay_steps}) "
                f"exceeds what the buffer supports."
            )
        # R2: internal/known-mode ordering is by (apply_step, send_step) --
        # i.e. application order, which is meaningful information once the
        # agent is told the real remaining time. But when the delay is
        # supposed to be HIDDEN (delay_known_to_agent=False), exposing the
        # queue in apply-order still leaks the relative delay ranking
        # through slot position alone (e.g. a shorter-delay command sent
        # later would appear in an earlier slot than a longer-delay command
        # sent earlier) even though pending_remaining_time is zeroed out.
        # Fixed real bug: previously this was unconditionally sorted by
        # apply_step regardless of delay_known_to_agent. In hidden mode the
        # visible order must instead be send order (dispatch order), which
        # the agent already knows on its own (it sent the commands) and so
        # carries no extra information about the delay distribution.
        if self.delay_known_to_agent:
            pending_sorted = sorted(self._in_flight, key=lambda c: (c["apply_step"], c["send_step"]))
        else:
            pending_sorted = sorted(self._in_flight, key=lambda c: c["send_step"])
        pending_queue = np.zeros((self.capacity, self._act_dim), dtype=np.float32)
        pending_remaining_time = np.zeros(self.capacity, dtype=np.float32)
        for i, cmd in enumerate(pending_sorted):
            pending_queue[i] = cmd["action"]
            pending_remaining_time[i] = cmd["apply_step"] - current_step
        return pending_queue, pending_remaining_time

    def _augmented_obs(self, raw_obs, current_step: int):
        pending_queue, pending_remaining_time = self._build_pending_arrays(current_step)
        remaining_time_for_agent = (
            pending_remaining_time if self.delay_known_to_agent
            else np.zeros_like(pending_remaining_time)
        )
        base = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        flat = np.concatenate([
            base,
            self._action_history.reshape(-1),
            pending_queue.reshape(-1),
            remaining_time_for_agent,
            self._last_executed_action,
        ]).astype(np.float32)
        debug = {
            "action_history": self._action_history.copy(),
            "pending_queue": pending_queue,
            "pending_remaining_time": pending_remaining_time,  # true values, even when masked from the agent
            "last_executed_action": self._last_executed_action.copy(),
            "delay_known_to_agent": self.delay_known_to_agent,
            "in_flight_count": len(self._in_flight),
        }
        return flat, debug

    def _delayed_raw_obs(self, raw_obs):
        """Simple FIFO observation delay (no reordering -- see docstring)."""
        self._obs_queue.append(np.asarray(raw_obs, dtype=np.float32).copy())
        return self._obs_queue[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fixed real bug: the delay RNG used to be fixed once at __init__
        # and never touched again, so env.reset(seed=s) reproduced the
        # vehicle spawn/traffic but NOT the sampled delay sequence -- calling
        # reset(seed=41) twice in the same process gave two DIFFERENT delay
        # sequences (confirmed empirically). Fix: when reset() is given an
        # explicit seed AND the wrapper itself was constructed without its
        # own pinned seed (self._explicit_seed is None, i.e. the caller
        # didn't ask for an independent delay RNG), re-derive the delay RNG
        # from that reset seed via SeedSequence.spawn so it's reproducible
        # per-seed but decorrelated from the env's own RNG stream. A bare
        # reset() (no seed kwarg) intentionally does NOT touch the RNG --
        # it continues the existing stream, so repeated resets during
        # training don't replay identical delay sequences every episode.
        reset_seed = kwargs.get("seed")
        if reset_seed is not None and self._explicit_seed is None:
            self._rng = np.random.default_rng(np.random.SeedSequence(reset_seed).spawn(1)[0])
        self._reset_internal_state()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_queue.append(np.asarray(obs, dtype=np.float32).copy())
        delayed_obs = self._obs_queue[0]
        flat_obs, debug = self._augmented_obs(delayed_obs, self._current_step)
        info = dict(info)
        info["delay_debug"] = debug
        return flat_obs, info

    def step(self, action):
        # Fixed real bug: np.asarray(action, dtype=np.float32) does NOT copy
        # when `action` is already a float32 ndarray -- it returns the SAME
        # object. If the caller reuses one buffer across steps (a common
        # pattern to avoid per-step allocations) and mutates it in place
        # after this call returns, that mutation silently corrupts the
        # command we stashed in `_in_flight` for up to `capacity` future
        # steps (confirmed empirically: pending_queue showed the caller's
        # LATER mutated values instead of the action actually sent). Always
        # take our own immutable snapshot at send time.
        action = np.array(action, dtype=np.float32, copy=True)
        send_step = self._current_step
        delay = self._sample_delay()
        apply_step = send_step + delay

        self._in_flight.append({"send_step": send_step, "apply_step": apply_step, "action": action})
        self._push_action_history(action)  # R7: independent of pending_queue resolution below

        action_to_apply, discarded_this_step = self._resolve_action_to_apply(send_step)

        obs, reward, terminated, truncated, info = self.env.step(action_to_apply)
        self._current_step += 1

        delayed_obs = self._delayed_raw_obs(obs)
        flat_obs, debug = self._augmented_obs(delayed_obs, self._current_step)
        debug.update({
            "send_step": send_step,
            "apply_step": apply_step,
            "delay_sampled": delay,
            "action_applied": action_to_apply,
            "discarded_this_step": discarded_this_step,
        })
        info = dict(info)
        info["delay_debug"] = debug
        return flat_obs, reward, terminated, truncated, info


class SafeSpawnWrapper(gym.Wrapper):
    """Works around a real HighwayEnv bug in `RacetrackEnvOval._make_vehicles`:
    the controlled vehicle's initial longitudinal position on the "a"-"b"
    straight is sampled from a HARDCODED `rng.uniform(20.0, 50.0)`, regardless
    of the track's configured `length`. With the native default (`length`
    ~100, straight lane ~101m long) that's always safely inside the segment.
    But with a short straight -- exactly the "3 oval variations" this project
    uses (80/30/5 m) -- that hardcoded range can and does overrun the actual
    lane: at `length=5` (lane length 6m) it ALWAYS does (confirmed 20/20
    resets); at `length=30` (lane length 31m) it does about a quarter of the
    time; `length=80` (lane length 81m) is safe. The car ends up extrapolated
    past the straight's end, off any real lane -- which is exactly the "spawn
    fora da pista" symptom.

    This is a bug in the native env, not in our config/wrappers, and it's not
    reachable through `config=` (the spawn range is hardcoded in
    `_make_vehicles`, not a config key). Patching/subclassing
    `RacetrackEnvOval` would violate the project's rule of never touching
    native HighwayEnv classes, so instead this wrapper runs AFTER the native
    `reset()`, detects a controlled vehicle that landed off its own assigned
    lane, and repositions it to a safe longitudinal offset that actually
    fits within that lane's real length -- then recomputes the observation
    (and info) so they reflect the corrected position rather than the
    native, broken one. It's a no-op whenever the native spawn was already
    valid (e.g. `length=80`, or non-oval tracks), so it's always safe to
    apply.

    Only the controlled vehicle(s) are affected by this bug -- IDM/other
    traffic vehicles already sample their longitudinal position from their
    own lane's real `.length`, confirmed unaffected even at `length=5`.

    Also guards against a second, related native issue: the oval's optional
    obstacle presets (`block_lane`/`force_decision`, native `RacetrackEnvOval`
    config keys) place `Obstacle` objects at `length - <hardcoded offset>`
    world coordinates -- also never scaled/validated against the configured
    `length`. Confirmed empirically across the 3 straight-length variations
    used in this project (5/30/80 m):

    - `length=5`: every obstacle lands off any lane entirely (pure no-op --
      not dangerous, just silently doesn't add the intended challenge).
    - `length=30`: `block_lane` obstacles land on-lane but on the *curve*
      leading into the straight (not "mid-straight", as designed);
      `force_decision` is a no-op (off-lane).
    - `length=80`: `block_lane` obstacles land squarely inside the native
      hardcoded controlled-vehicle spawn range (20-50 m) -- i.e. the car can
      spawn essentially on top of, or a few meters from, an obstacle, with
      no chance to react. Measured as close as ~4.7 m at spawn in one seed.
      This is a genuine, confirmed unfairness: an episode-ending collision
      completely unrelated to policy quality.

    And a THIRD, even worse variant of the same root bug: the dedicated
    "front vehicle" that `_make_vehicles` spawns whenever `other_vehicles >
    0` (`IDMVehicle.make_on_lane(road, ("b","c", <same lane as controlled>),
    longitudinal=uniform(0, lane("b","c").length), ...)`) has NO proximity
    check against the controlled vehicle at all -- unlike every *other*
    traffic vehicle spawned after it, which the native code does check
    (`for v in road.vehicles: if dist < 20.0: break`, native comment
    "Prevent early collisions"). On a long straight this "front vehicle" is
    always far enough ahead (node "b" is ~100m+ away) for that gap to not
    matter in practice. On the short straights this project uses, it does:
    confirmed empirically (40-60 seeds per combination) with the full wrapper
    stack already applied (including the two fixes above):

    - `length=5`: min distance as low as **0.04 m** (literal overlap) across
      multiple seeds, independent of `other_vehicles` count (only need
      `other_vehicles > 0` to trigger the dedicated front-vehicle spawn).
    - `length=30`: min distance as low as **0.18 m**, in up to ~20% of seeds
      under 5 m.
    - `length=80`: safe (min ~20-38 m) -- the controlled-vehicle spawn range
      (20-50 m on "a"-"b") stays far from node "b" (at 81 m), where the
      front vehicle spawns.

    This wrapper closes all three gaps the same way: after `reset()`, if any
    controlled vehicle starts within `min_obstacle_clearance_m` of any
    `road.objects` obstacle OR any other vehicle in `road.vehicles`
    (traffic, or another controlled vehicle), it resamples a longitudinal
    position on the SAME lane that clears every such entity by that margin
    (rejection sampling, falling back to a position chosen from the largest
    projected gap if no candidate clears in 50 tries). The default
    clearance (15 m) is a reaction-distance heuristic (stopping distance
    from `speed_limit=10 m/s` at `|acceleration| <= 5 m/s^2` is ~10 m, plus a
    couple of policy steps' margin at `policy_frequency=5`), not a hard
    physical limit -- pass `min_obstacle_clearance_m=` to tune it. Traffic
    vehicles keep moving after reset (unlike static obstacles). The margin
    is a best-effort target at reset and may be impossible on short or
    crowded lanes.

    None of this makes the obstacle presets scale meaningfully to short
    straights (e.g. `length=5` still can't fit `block_lane`'s 4 obstacles in
    any way that resembles the native design). This wrapper only adjusts
    the starting state.

    If `initial_controlled_speed` is set, initialize controlled vehicles at
    that speed. Steering-only kinematic actions apply zero acceleration,
    so the vehicle maintains it during normal driving and the simulator
    retains its native collision dynamics.
    """

    def __init__(
        self,
        env: gym.Env,
        min_obstacle_clearance_m: float = 15.0,
        initial_controlled_speed: float | None = None,
    ):
        super().__init__(env)
        self.min_obstacle_clearance_m = min_obstacle_clearance_m
        self.initial_controlled_speed = initial_controlled_speed

    def _fix_offlane_controlled_vehicles(self) -> bool:
        unwrapped = self.unwrapped
        road = getattr(unwrapped, "road", None)
        if road is None:
            return False
        fixed_any = False
        for vehicle in getattr(unwrapped, "controlled_vehicles", []):
            lane = road.network.get_lane(vehicle.lane_index)
            if lane.on_lane(vehicle.position):
                continue
            margin = 0.5
            safe_high = max(lane.length - margin, margin)
            longitudinal = float(road.np_random.uniform(margin, safe_high))
            vehicle.position = lane.position(longitudinal, 0)
            vehicle.heading = lane.heading_at(longitudinal)
            fixed_any = True
        return fixed_any

    def _fix_unsafe_entity_proximity(self) -> bool:
        """Covers obstacles (seção 12 do project doc) AND other vehicles,
        in particular the native "front vehicle" that -- unlike every other
        traffic vehicle -- is spawned with no proximity check at all
        against the controlled vehicle."""
        unwrapped = self.unwrapped
        road = getattr(unwrapped, "road", None)
        if road is None:
            return False
        controlled = list(getattr(unwrapped, "controlled_vehicles", []))
        fixed_any = False
        for vehicle in controlled:
            objects = getattr(road, "objects", None) or []
            other_vehicles = [v for v in getattr(road, "vehicles", []) if v is not vehicle]
            nearby_positions = [np.asarray(o.position, dtype=float) for o in objects] + [
                np.asarray(v.position, dtype=float) for v in other_vehicles
            ]
            if not nearby_positions:
                continue
            vehicle_pos = np.asarray(vehicle.position, dtype=float)
            nearest = min(float(np.linalg.norm(vehicle_pos - p)) for p in nearby_positions)
            if nearest >= self.min_obstacle_clearance_m:
                continue
            lane = road.network.get_lane(vehicle.lane_index)
            margin = 0.5
            safe_high = max(lane.length - margin, margin)
            placed = False
            for _ in range(50):
                longitudinal = float(road.np_random.uniform(margin, safe_high))
                candidate_pos = np.asarray(lane.position(longitudinal, 0), dtype=float)
                if all(
                    float(np.linalg.norm(candidate_pos - p)) >= self.min_obstacle_clearance_m
                    for p in nearby_positions
                ):
                    vehicle.position = lane.position(longitudinal, 0)
                    vehicle.heading = lane.heading_at(longitudinal)
                    placed = True
                    break
            if not placed:
                # 50 random tries all landed too close to *something* --
                # typically means the lane itself is short relative to
                # `min_obstacle_clearance_m` and/or crowded (e.g. a traffic
                # vehicle uniformly spawned anywhere on a ~30m turn lane, so
                # unlike the two obstacle presets it isn't reliably biased
                # toward the segment's end -- "use the start" would just
                # trade one unsafe spot for another). Fall back to the
                # longitudinal position that maximizes the minimum distance
                # to every nearby entity: project each onto this lane and
                # place the vehicle in the middle of the largest resulting
                # gap. Best-effort -- if the lane is simply too short/
                # crowded to clear everything, this is the closest we can
                # get, not a hard guarantee.
                longitudinal = self._largest_gap_longitudinal(lane, nearby_positions, margin, safe_high)
                vehicle.position = lane.position(longitudinal, 0)
                vehicle.heading = lane.heading_at(longitudinal)
            fixed_any = True
        return fixed_any

    @staticmethod
    def _largest_gap_longitudinal(lane, nearby_positions, margin, safe_high) -> float:
        projected = [margin, safe_high]
        for p in nearby_positions:
            try:
                s, _lateral = lane.local_coordinates(p)
            except Exception:
                continue
            if margin <= s <= safe_high:
                projected.append(float(s))
        projected.sort()
        best_mid, best_gap = margin, -1.0
        for a, b in zip(projected[:-1], projected[1:]):
            gap = b - a
            if gap > best_gap:
                best_gap = gap
                best_mid = (a + b) / 2.0
        return best_mid

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        changed_spawn = self._fix_offlane_controlled_vehicles()
        changed_proximity = self._fix_unsafe_entity_proximity()
        unwrapped = self.unwrapped
        changed_speed = False
        if self.initial_controlled_speed is not None:
            for vehicle in unwrapped.controlled_vehicles:
                if vehicle.speed != self.initial_controlled_speed:
                    vehicle.speed = self.initial_controlled_speed
                    changed_speed = True
        if changed_spawn or changed_proximity or changed_speed:
            obs = unwrapped.observation_type.observe()
            info = unwrapped._info(obs, action=unwrapped.action_space.sample())
        return obs, info

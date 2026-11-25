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

    Adds `coef * clip(speed / speed_limit, 0, 1.2)` to the reward each step,
    gated to 0 off-road (consistent with the base env's own on-road gating)
    so the agent can't farm the speed bonus by cutting corners off-track.
    """

    def __init__(self, env: gym.Env, coef: float, speed_limit: float):
        super().__init__(env)
        self.coef = coef
        self.speed_limit = speed_limit

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        on_road = info.get("rewards", {}).get("on_road_reward", True)
        speed = info.get("speed", 0.0)
        speed_frac = float(np.clip(speed / self.speed_limit, 0.0, 1.2)) if on_road else 0.0
        bonus = self.coef * speed_frac
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
    R2. Comandos pendentes (`_in_flight`) sao conceitualmente ordenados por
        `apply_step` crescente (ordenacao aplicada ao montar `pending_queue`).
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
        pending_sorted = sorted(self._in_flight, key=lambda c: c["apply_step"])  # R2
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
        self._reset_internal_state()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_queue.append(np.asarray(obs, dtype=np.float32).copy())
        delayed_obs = self._obs_queue[0]
        flat_obs, debug = self._augmented_obs(delayed_obs, self._current_step)
        info = dict(info)
        info["delay_debug"] = debug
        return flat_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
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

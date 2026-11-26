"""Ambiente do experimento final: Racetrack Oval com atraso aleatório de ação.

- Recompensa (``ProjectRewardWrapper``)::

      r = lmap(1/(1+4d²) − colisão − 0,3|a_direção| + clip(v/10, 0, 1), [−1, 2] → [0, 1]) × na_pista

  A penalidade de ação usa só a direção da ação *executada* (a que chega ao
  veículo depois do atraso), como na recompensa nativa.
- Observação: grade de ocupação nativa (2×12×12, ``presence`` + ``on_road``)
  achatada + velocidade do ego (``EgoSpeedObservation``) + últimas 6 ações
  emitidas + atraso efetivo mais recente (``RandomActionDelayWrapper``). O
  tamanho é o mesmo com e sem atraso; só muda com ``accel`` (1 ou 2 dims de ação).
- Atraso de ação: ``k_t = min(k_sorteado, k_{t-1} + 1)``, executa ``a_{t-k_t}``.
  Na avaliação, ``k`` é fixo durante o episódio.

O ``info`` de cada passo carrega ``delay_k`` (atraso efetivo), ``progress_m``
(avanço ao longo da faixa, em metros), ``heading_delta`` (para contar voltas)
e ``off_road``.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from highway_env import utils

from .racetrack import RacetrackEnvConfig, make_racetrack_env

N_ACTIONS_IN_OBS = 6        # = maior atraso avaliado
MAX_EVAL_DELAY = 6
DEFAULT_TRAIN_MAX_DELAY = 3
ENV_ID = "racetrack-oval-v1"
STEERING_PENALTY = 0.3
REWARD_RANGE = (-1.0, 2.0)


def _wrap_angle(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


class ProjectRewardWrapper(gym.Wrapper):
    """Substitui a recompensa nativa pela do projeto (ver docstring do módulo)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_heading = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_heading = float(self.unwrapped.vehicle.heading)
        return obs, info

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        base = self.unwrapped
        vehicle = base.vehicle
        lane = vehicle.lane
        s, lateral = lane.local_coordinates(vehicle.position)
        speed = float(vehicle.speed)
        steering = float(np.asarray(action, dtype=np.float64).reshape(-1)[-1])  # direção é a última componente
        on_road = bool(vehicle.on_road)
        crashed = bool(vehicle.crashed)

        raw = (
            1.0 / (1.0 + base.config["lane_centering_cost"] * lateral ** 2)
            - float(crashed)
            - STEERING_PENALTY * abs(steering)
            + float(np.clip(speed / base.config["speed_limit"], 0.0, 1.0))
        )
        reward = float(utils.lmap(raw, list(REWARD_RANGE), [0.0, 1.0])) * float(on_road)

        dt = 1.0 / base.config["policy_frequency"]
        heading = float(vehicle.heading)
        info = dict(info)
        info["progress_m"] = speed * dt * float(np.cos(heading - lane.heading_at(s)))
        info["heading_delta"] = _wrap_angle(heading - self._prev_heading)
        info["off_road"] = not on_road
        info["reward_raw"] = raw
        self._prev_heading = heading
        return obs, reward, terminated, truncated, info


class EgoSpeedObservation(gym.ObservationWrapper):
    """Acrescenta ``v / speed_limit`` a uma observação já achatada (a grade
    de ocupação não informa a velocidade do ego)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n + 1,), dtype=np.float32)

    def observation(self, observation):
        base = self.unwrapped
        v = base.vehicle.speed / base.config["speed_limit"]
        return np.concatenate([np.asarray(observation, dtype=np.float32).reshape(-1), [v]]).astype(np.float32)


class RandomActionDelayWrapper(gym.Wrapper):
    """Atraso de execução de ação com fila e observação aumentada.

    Modos:
      - ``mode="off"``: ``k_t = 0`` sempre (baseline; a observação continua
        aumentada, para ter o mesmo tamanho).
      - ``mode="random"``: ``k_sorteado ~ U{0..max_delay}`` a cada passo,
        ``k_t = min(k_sorteado, k_{t-1} + 1)``.
      - ``mode="fixed"``: ``k_t = fixed_delay`` durante todo o episódio.

    Executa ``a_{t-k_t}``. A fila começa com ações neutras (zeros). Quando o
    atraso cai, as ações intermediárias são puladas; quando sobe, repete-se a
    anterior (consequência direta da regra acima).

    Observação: ``[obs_base | últimas N ações emitidas (mais recente primeiro)
    | k_{t} / MAX_EVAL_DELAY]``, em que ``k_t`` é o atraso efetivo do último
    passo executado.

    RNG: ``reset(seed=s)`` recria o gerador a partir de ``s`` (reprodutível);
    ``reset()`` sem seed continua o fluxo.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "off",
        max_delay: int = DEFAULT_TRAIN_MAX_DELAY,
        fixed_delay: int = 0,
        n_actions_in_obs: int = N_ACTIONS_IN_OBS,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if mode not in ("off", "random", "fixed"):
            raise ValueError(f"mode inválido: {mode!r}")
        if not (0 <= max_delay <= n_actions_in_obs and 0 <= fixed_delay <= n_actions_in_obs):
            raise ValueError("atrasos devem estar em [0, n_actions_in_obs]")
        if not (isinstance(env.observation_space, spaces.Box) and len(env.observation_space.shape) == 1):
            raise TypeError("RandomActionDelayWrapper espera observação 1D (aplique depois de achatar).")
        self.mode = mode
        self.max_delay = max_delay
        self.fixed_delay = fixed_delay
        self.n = n_actions_in_obs
        self._act_dim = int(env.action_space.shape[0])
        self._rng = np.random.default_rng(seed)
        n_obs = env.observation_space.shape[0] + self.n * self._act_dim + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.k_counts = np.zeros(self.n + 1, dtype=np.int64)  # histograma acumulado de k_t
        self._reset_queue()

    def _reset_queue(self):
        # _emitted[0] = ação emitida mais recente; guarda n+1 para cobrir k = n.
        self._emitted = deque([np.zeros(self._act_dim, np.float32) for _ in range(self.n + 1)], maxlen=self.n + 1)
        self._k = self.fixed_delay if self.mode == "fixed" else 0

    def _next_k(self) -> int:
        if self.mode == "off":
            return 0
        if self.mode == "fixed":
            return self.fixed_delay
        sampled = int(self._rng.integers(0, self.max_delay + 1))
        return min(sampled, self._k + 1)

    def _augment(self, obs):
        history = np.concatenate(list(self._emitted)[: self.n])
        return np.concatenate([
            np.asarray(obs, dtype=np.float32).reshape(-1), history, [self._k / MAX_EVAL_DELAY]
        ]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if kwargs.get("seed") is not None:
            self._rng = np.random.default_rng(np.random.SeedSequence(kwargs["seed"]).spawn(1)[0])
        self._reset_queue()
        return self._augment(obs), info

    def step(self, action):
        action = np.array(action, dtype=np.float32, copy=True).reshape(self._act_dim)
        self._emitted.appendleft(action)
        self._k = self._next_k()
        executed = self._emitted[self._k]
        self.k_counts[self._k] += 1
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info["delay_k"] = self._k
        info["executed_action"] = executed.copy()
        return self._augment(obs), reward, terminated, truncated, info


def make_delay_experiment_env(
    accel: bool,
    delay_mode: str = "off",
    max_delay: int = DEFAULT_TRAIN_MAX_DELAY,
    fixed_delay: int = 0,
    traffic: str | int = "medium",
    straight_length_m: int = 30,
    duration_s: int = 100,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Monta o ambiente de uma condição do experimento (ver docstring do módulo)."""
    config = RacetrackEnvConfig(
        env_id=ENV_ID,
        traffic_density=traffic,
        duration=duration_s,
        observation_type="OccupancyGrid",
        action_longitudinal=accel,
        action_lateral=True,
        oval_straight_length_m=straight_length_m,
        high_speed_reward=0.0,      # a recompensa do projeto já inclui a velocidade
        lane_centering_cost=4.0,
    )
    env = make_racetrack_env(config=config, render_mode=render_mode, flatten_obs=False)
    env = ProjectRewardWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = EgoSpeedObservation(env)
    env = RandomActionDelayWrapper(env, mode=delay_mode, max_delay=max_delay, fixed_delay=fixed_delay)
    return env

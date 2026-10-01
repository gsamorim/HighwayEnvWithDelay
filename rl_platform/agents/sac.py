"""Soft Actor-Critic (Haarnoja et al., 2018) with automatic entropy tuning
(Haarnoja et al., 2019), implemented over flat-vector observations and
continuous action spaces such as HighwayEnv's [acceleration, steering].

This is a from-scratch, dependency-light implementation (just PyTorch) so the
whole learning algorithm is auditable in one file, rather than pulling in a
larger RL framework.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .base import Agent
from .networks import GaussianPolicy, TwinQNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005                 # target network Polyak averaging rate
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    autotune_alpha: bool = True
    target_entropy: float | None = None  # default: -act_dim
    updates_per_step: int = 1
    learning_starts: int = 1_000
    device: str = "cpu"
    seed: int | None = None


class SACAgent(Agent):
    """Off-policy, maximum-entropy actor-critic agent for continuous control."""

    name = "sac"

    def __init__(self, config: SACConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.actor = GaussianPolicy(config.obs_dim, config.act_dim, config.hidden_sizes).to(self.device)
        self.critic = TwinQNetwork(config.obs_dim, config.act_dim, config.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=config.critic_lr)

        self.autotune_alpha = config.autotune_alpha
        target_entropy = config.target_entropy
        if target_entropy is None:
            target_entropy = -float(config.act_dim)
        self.target_entropy = target_entropy
        if self.autotune_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.init_alpha), device=self.device, requires_grad=True
            )
            self.alpha_opt = Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=self.device)

        self.buffer = ReplayBuffer(config.buffer_size, config.obs_dim, config.act_dim, seed=config.seed)
        self._update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def update_count(self) -> int:
        """Total number of gradient updates this agent has ever taken,
        preserved across save/load so a resumed run's logs/schedules can
        pick up where they left off instead of resetting to 0."""
        return self._update_count

    # --- Agent interface -----------------------------------------------
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    def observe(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)

    def ready_to_update(self) -> bool:
        return len(self.buffer) >= max(self.cfg.batch_size, self.cfg.learning_starts)

    def update(self) -> dict[str, float]:
        if not self.ready_to_update():
            return {}

        metrics: dict[str, float] = {}
        for _ in range(self.cfg.updates_per_step):
            metrics = self._update_once()
            self._update_count += 1
        return metrics

    def _update_once(self) -> dict[str, float]:
        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        rew = torch.as_tensor(rew, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            target_q = rew.squeeze(-1) + (1 - done.squeeze(-1)) * self.cfg.gamma * q_next

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Actor update ----
        pi_action, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, pi_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Temperature (alpha) update ----
        alpha_loss_val = 0.0
        if self.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        # ---- Target network soft update ----
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": alpha_loss_val,
            "mean_q": float(q_pi.mean().item()),
            "entropy": float(-logp.mean().item()),
        }

    def save(self, path: str) -> None:
        """Save everything needed to either (a) just run the policy
        (actor/critic weights) or (b) fully resume training later, byte-for-
        byte equivalent to never having stopped: optimizer momentum/variance
        state and the gradient-step counter are included too. (The replay
        buffer is *not* part of this file -- it can be large -- see
        `ReplayBuffer.save`/`.load` and `rl_platform.training.checkpointing`,
        which bundles both together for `scripts/live_trainer.py`.)"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict() if self.autotune_alpha else None,
                "update_count": self._update_count,
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint written by `save()`. Tolerant of older
        checkpoints saved before optimizer state / step count were added
        (e.g. the flat `results/checkpoints/*_final.pt` files from earlier
        eval-only runs): those keys simply won't be present, and we fall
        back to freshly-initialized optimizers rather than raising, so an
        old checkpoint can still be used to warm-start live/continued
        training (just without bit-exact optimizer momentum continuity)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        if ckpt.get("actor_opt") is not None:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if ckpt.get("critic_opt") is not None:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if self.autotune_alpha and ckpt.get("alpha_opt") is not None:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self._update_count = ckpt.get("update_count", 0)

"""Bundles an agent's full training state -- network + optimizer weights,
and (for off-policy agents) its replay buffer -- into one directory, plus a
JSON sidecar of run metadata (traffic config, cumulative env steps, wall
clock trained, etc).

This is what makes `scripts/live_trainer.py` able to stop and resume: each
agent gets a `results/live_checkpoints/<name>/` directory containing
`model.pt`, `buffer.npz` (optional), and `meta.json`. Deliberately agent-type
agnostic (it just calls `agent.save`/`agent.load` and, if present,
`agent.buffer.save`/`.load`), so a future non-SAC agent registered in
`rl_platform.agents.registry` gets the same resumability for free as long as
it implements `Agent.save`/`Agent.load` -- only agents that also expose a
`.buffer` attribute (replay-buffer-based, i.e. off-policy) get the buffer
persisted too.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from ..agents.base import Agent


def save_agent_bundle(
    dir_path: str,
    agent: Agent,
    meta: dict[str, Any],
    include_buffer: bool = True,
) -> None:
    os.makedirs(dir_path, exist_ok=True)
    agent.save(os.path.join(dir_path, "model.pt"))

    buffer = getattr(agent, "buffer", None)
    if include_buffer and buffer is not None:
        buffer.save(os.path.join(dir_path, "buffer.npz"))

    meta = dict(meta)
    meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["saved_at_epoch"] = time.time()
    with open(os.path.join(dir_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_agent_bundle(
    dir_path: str,
    agent: Agent,
    load_buffer: bool = True,
) -> Optional[dict[str, Any]]:
    """Returns the saved meta dict if a checkpoint existed and was loaded,
    else None (caller should treat this as "start fresh")."""
    model_path = os.path.join(dir_path, "model.pt")
    if not os.path.exists(model_path):
        return None

    agent.load(model_path)

    buffer = getattr(agent, "buffer", None)
    buffer_path = os.path.join(dir_path, "buffer.npz")
    if load_buffer and buffer is not None and os.path.exists(buffer_path):
        buffer.load(buffer_path)

    meta_path = os.path.join(dir_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def bundle_exists(dir_path: str) -> bool:
    return os.path.exists(os.path.join(dir_path, "model.pt"))

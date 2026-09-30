"""Config loading, seeding and hypernetwork checkpoint I/O."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # a single top-level wrapper key is unwrapped
    if isinstance(cfg, dict) and len(cfg) == 1:
        (only,) = cfg.values()
        if isinstance(only, dict):
            return only
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_hyper(manager, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"manager": manager.state_dict(), "layer_names": list(manager.layer_names)}, path)


def pad_task_slots(state: dict[str, torch.Tensor], n_slots: int) -> dict[str, torch.Tensor]:
    """Widen a checkpoint written for fewer concepts (e.g. fifty) to `n_slots` task slots, so a
    sequence can continue on a longer config. The new slots get exactly the values a fresh network
    starts with: zero rows of ortho_basis and canon_pooled (the key of task k is drawn when task
    k starts, its basis row is written when it ends). Every other tensor is
    copied unchanged."""
    if "ortho_basis" not in state:
        return state
    old = state["ortho_basis"].shape[0]
    if old == n_slots:
        return state
    if old > n_slots:
        raise ValueError(f"checkpoint has {old} task slots, the config only {n_slots}")
    new = dict(state)
    for name in ("ortho_basis", "canon_pooled"):
        t = state[name]
        new[name] = torch.cat([t, torch.zeros(n_slots - old, t.shape[1], dtype=t.dtype, device=t.device)])
    return new


def load_hyper(manager, path: str, map_location: str = "cpu") -> dict[str, Any]:
    blob = torch.load(path, map_location=map_location)
    state = blob["manager"]
    if hasattr(manager, "ortho_basis"):
        state = pad_task_slots(state, manager.ortho_basis.shape[0])
    manager.load_state_dict(state)
    return blob

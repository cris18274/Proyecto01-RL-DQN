"""Utilities shared by independent reinforcement-learning implementations."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is best-effort, not a promise across different CPU/OS builds.


def make_env(env_id: str, render_mode=None):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "Falta gymnasium. Ejecuta: python -m pip install -r requirements.txt"
        ) from exc
    return gym.make(env_id, render_mode=render_mode)


def seed_env(env, seed: int):
    env.action_space.seed(seed)
    return env.reset(seed=seed)[0]


def ensure_discrete_env(env) -> tuple[int, int]:
    # Avoid duck-typing an unsupported continuous-action Box space as discrete.
    from gymnasium import spaces
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError(f"DQN requires discrete actions, got {env.action_space}")
    if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
        raise ValueError("Expected a vector-valued continuous observation")
    return int(env.observation_space.shape[0]), int(env.action_space.n)


def make_run_name(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{seed:04d}"


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def metadata() -> dict:
    import importlib.metadata as meta
    packages = {}
    for name in ('gymnasium', 'torch', 'numpy', 'matplotlib'):
        try:
            packages[name] = meta.version(name)
        except meta.PackageNotFoundError:
            packages[name] = 'not installed'
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'pytorch_threads': torch.get_num_threads(),
        'packages': packages,
    }


def linear_epsilon(step: int, total_steps: int, start: float, end: float,
                   decay_fraction: float) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    frac = min(1.0, max(0.0, step / decay_steps))
    return float(start + frac * (end - start))


def parse_seeds(arg: str) -> list[int]:
    seeds = [int(s.strip()) for s in arg.split(',') if s.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use one or more unique integer seeds, e.g. 11,22,33')
    return seeds

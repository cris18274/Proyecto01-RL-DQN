"""Q-learning tabular from scratch. No pre-built RL agents.

Equation: Q[s,a] <- Q[s,a] + alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a]).
At MDP terminal states the continuation term is zero; time-limit truncations
may bootstrap because they are not an environment-defined terminal transition.
"""
from __future__ import annotations

import numpy as np


CARTPOLE_CLIP_RANGES = np.array([
    [-2.4, 2.4],          # cart position, terminal before |x| > 2.4
    [-3.0, 3.0],          # velocity is officially unbounded: engineering clipping
    [-0.20943951, 0.20943951],  # pole angle, about +/- 12 degrees
    [-3.5, 3.5],          # angular velocity unbounded: engineering clipping
], dtype=np.float64)


class Discretizer:
    def __init__(self, bins: int, ranges: np.ndarray = CARTPOLE_CLIP_RANGES):
        if bins < 2:
            raise ValueError('bins must be >= 2')
        ranges = np.asarray(ranges, dtype=np.float64)
        if ranges.ndim != 2 or ranges.shape[1] != 2 or np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError('ranges must be (n_features, 2) with low < high')
        self.bins = int(bins)
        self.ranges = ranges.copy()
        self.edges = [np.linspace(lo, hi, bins + 1)[1:-1] for lo, hi in ranges]
        self.n_states = int(bins ** len(self.edges))
        self.n_features = len(self.edges)

    def encode(self, obs) -> int:
        x = np.asarray(obs, dtype=np.float64)
        if x.shape != (self.n_features,):
            raise ValueError(f'expected observation shape ({self.n_features},), got {x.shape}')
        coords = tuple(int(np.searchsorted(edges, value, side='right'))
                       for edges, value in zip(self.edges, x))
        return int(np.ravel_multi_index(coords, [self.bins] * self.n_features))


class QLearningAgent:
    def __init__(self, discretizer: Discretizer, n_actions: int,
                 alpha: float = 0.15, gamma: float = 0.99, seed: int = 0):
        self.discretizer = discretizer
        self.n_actions = int(n_actions)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.q = np.zeros((discretizer.n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def act(self, obs, epsilon: float = 0.0) -> int:
        state = self.discretizer.encode(obs)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        # Random tie-break improves initial exploration when all entries are zero.
        row = self.q[state]
        if epsilon <= 0.0:
            return int(np.argmax(row))  # strictly greedy and reproducible evaluation
        winners = np.flatnonzero(np.isclose(row, row.max()))
        return int(self.rng.choice(winners))

    def update(self, obs, action: int, reward: float, next_obs,
               terminated: bool) -> float:
        s, ns = self.discretizer.encode(obs), self.discretizer.encode(next_obs)
        continuation = 0.0 if terminated else self.gamma * float(self.q[ns].max())
        td_error = float(reward) + continuation - float(self.q[s, action])
        self.q[s, action] += self.alpha * td_error
        return td_error

    def save(self, path) -> None:
        np.savez_compressed(path, q=self.q, bins=np.int64(self.discretizer.bins),
                            ranges=self.discretizer.ranges,
                            alpha=np.float64(self.alpha), gamma=np.float64(self.gamma))

    @classmethod
    def load(cls, path) -> 'QLearningAgent':
        with np.load(path, allow_pickle=False) as data:
            disc = Discretizer(int(data['bins']), data['ranges'])
            n_actions = data['q'].shape[1]
            obj = cls(disc, n_actions, float(data['alpha']), float(data['gamma']))
            obj.q[:] = data['q']
            return obj

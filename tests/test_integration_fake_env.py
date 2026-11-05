"""Offline integration test: uses a tiny deterministic dummy environment, NOT Gymnasium.
This verifies the whole logging/checkpoint pipeline when the real library is absent.
Scientific training still REQUIRES the actual Gymnasium environments.
"""
import json
import numpy as np

import rl01.training as training
from rl01.analysis import analyze
from rl01.dqn import DQNConfig


class FakeActionSpace:
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.counter = 0

    def reset(self, seed=None):
        self.counter = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.counter += 1
        obs = np.full(4, 0.01 * self.counter, dtype=np.float32)
        return obs, 1., self.counter >= 9, False, {}

    def close(self):
        pass


def test_training_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'make_env', lambda name: FakeEnv())
    monkeypatch.setattr(training, 'ensure_discrete_env', lambda env: (4, 2))
    root = tmp_path / 'tier1'
    for seed in (11, 22):
        training.train_one(env_id='CartPole-v1', algo='tabular',
                           config_name='tab_bins6',
                           run_dir=root / f'tab_bins6_seed{seed:04d}',
                           steps=36, seed=seed, eval_every=12,
                           eval_episodes=2, final_eval_episodes=5,
                           epsilon_start=1.0, epsilon_end=.1,
                           epsilon_decay_fraction=.7, bins=6)
    analyze(root)
    assert (root / 'learning_curves.png').exists()
    assert (root / 'summary_by_variant.csv').exists()
    assert len(list(root.glob('*/policy.npz'))) == 2
    with (root/'tab_bins6_seed0011'/'summary.json').open() as f:
        row = json.load(f)
    assert row['train_steps'] == 36 and row['final_holdout_n_episodes'] == 5


def test_dqn_end_to_end_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'make_env', lambda name: FakeEnv())
    monkeypatch.setattr(training, 'ensure_discrete_env', lambda env: (4, 2))
    out = tmp_path / 'dqn_base_seed0007'
    training.train_one(env_id='CartPole-v1', algo='dqn',
                       config_name='dqn_base', run_dir=out,
                       steps=48, seed=7, eval_every=24,
                       eval_episodes=2, final_eval_episodes=5,
                       epsilon_start=1.0, epsilon_end=.1,
                       epsilon_decay_fraction=.7,
                       dqn_cfg=DQNConfig(hidden_width=16,
                                         buffer_capacity=128, batch_size=8,
                                         learning_starts=8, train_frequency=2,
                                         target_update_frequency=10))
    assert (out / 'policy.pt').exists()
    assert 'final_holdout' in (out / 'evaluations.csv').read_text()

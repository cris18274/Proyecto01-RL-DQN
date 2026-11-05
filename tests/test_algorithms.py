import numpy as np
import torch

from rl01.common import linear_epsilon
from rl01.dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from rl01.tabular import CARTPOLE_CLIP_RANGES, Discretizer, QLearningAgent


def test_discretization_extremes_and_shape():
    d = Discretizer(6, CARTPOLE_CLIP_RANGES)
    assert d.n_states == 6**4
    assert d.encode([0, 0, 0, 0]) < d.n_states
    assert d.encode([-1e6] * 4) == 0
    assert d.encode([1e6] * 4) == d.n_states - 1


def test_tabular_q_update_correct_terminal_and_nonterminal():
    d = Discretizer(6)
    agent = QLearningAgent(d, 2, alpha=0.5, gamma=0.9)
    s = np.array([0, 0, 0, 0], dtype=float)
    ns = np.array([1.0, 0, 0, 0], dtype=float)
    agent.q[d.encode(ns), :] = [2.0, 3.0]
    agent.update(s, 0, 1.0, ns, terminated=False)
    assert np.isclose(agent.q[d.encode(s), 0], 0.5 * (1 + .9 * 3))
    agent.q[d.encode(s), 1] = 0
    agent.update(s, 1, 1.0, ns, terminated=True)
    assert np.isclose(agent.q[d.encode(s), 1], 0.5)


def test_linear_epsilon_limits():
    assert np.isclose(linear_epsilon(0, 100, 1., .05, .7), 1.)
    assert np.isclose(linear_epsilon(70, 100, 1., .05, .7), .05)
    assert np.isclose(linear_epsilon(100, 100, 1., .05, .7), .05)


def test_replay_overwrite_and_sample():
    b = ReplayBuffer(capacity=3, obs_dim=4, seed=4)
    for i in range(7):
        obs = np.array([i] * 4, dtype=np.float32)
        b.add(obs, i % 2, float(i), obs+1, terminated=(i == 6))
    assert b.size == 3
    sample = b.sample(2, torch.device('cpu'))
    assert len(sample) == 5 and sample[0].shape == (2, 4)
    assert set(b.rewards.tolist()) == {4., 5., 6.}


def test_dqn_updates_online_but_not_target_then_sync(tmp_path):
    torch.manual_seed(7)
    cfg = DQNConfig(learning_rate=1e-2, hidden_width=16,
                    batch_size=4, buffer_capacity=64, learning_starts=1)
    a = DQNAgent(4, 2, cfg, seed=7)
    for i in range(20):
        s = np.array([i*.01, 0., 0., 0.], dtype=np.float32)
        a.observe(s, i % 2, 1.0, s+0.1, terminated=False)
    old_online = [p.clone() for p in a.online.parameters()]
    old_target = [p.clone() for p in a.target.parameters()]
    assert a.update() is not None
    assert any(not torch.equal(p, q) for p, q in zip(a.online.parameters(), old_online))
    assert all(torch.equal(p, q) for p, q in zip(a.target.parameters(), old_target))
    a.sync_target()
    assert all(torch.equal(p, q) for p, q in zip(a.target.parameters(), a.online.parameters()))
    path = tmp_path / 'model.pt'
    a.save(path)
    b = DQNAgent.load(path)
    obs = torch.zeros(1, 4)
    assert torch.allclose(a.online(obs), b.online(obs))


def test_ddqn_one_update_and_no_target_ablation():
    for double, disable_target in [(True, False), (False, True)]:
        cfg = DQNConfig(hidden_width=16, batch_size=4, buffer_capacity=32,
                        double_dqn=double, disable_target=disable_target)
        a = DQNAgent(6, 3, cfg, seed=2)
        for i in range(12):
            a.observe(np.ones(6, dtype=np.float32)*i, i%3,
                      -1., np.ones(6, dtype=np.float32)*(i+1), False)
        assert np.isfinite(a.update())


def test_tabular_load_save(tmp_path):
    a = QLearningAgent(Discretizer(10), 2)
    a.q[0,0] = 5.0
    p = tmp_path/'model.npz'
    a.save(p)
    b = QLearningAgent.load(p)
    assert b.q[0,0] == 5.0 and b.discretizer.bins == 10

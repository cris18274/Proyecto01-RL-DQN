"""Original PyTorch DQN and optional Double DQN, without agent libraries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DQNConfig:
    learning_rate: float = 5e-4
    gamma: float = 0.99
    hidden_width: int = 128
    batch_size: int = 64
    buffer_capacity: int = 50000
    learning_starts: int = 1000
    train_frequency: int = 4
    target_update_frequency: int = 1000  # measured in environment steps
    grad_clip_norm: float = 10.0
    disable_target: bool = False          # Tier 2 ablation: change one component
    double_dqn: bool = False             # Tier 3 option B: Double DQN


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Fixed-size ring buffer for decorrelated mini-batch sampling."""
    def __init__(self, capacity: int, obs_dim: int, seed: int):
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise ValueError(f'Cannot sample {batch_size} transitions from {self.size}')
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device, dtype=torch.long),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.terminated[idx], device=device),
        )


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig,
                 seed: int = 0, device: str = 'cpu'):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.device = torch.device(device)
        self.online = QNetwork(obs_dim, n_actions, cfg.hidden_width).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, cfg.hidden_width).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.buffer_capacity, obs_dim, seed + 12345)
        self.action_rng = np.random.default_rng(seed + 10)
        self.updates = 0

    def act(self, obs, epsilon: float = 0.0) -> int:
        if self.action_rng.random() < epsilon:
            return int(self.action_rng.integers(self.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self.device).unsqueeze(0)
            return int(self.online(obs_t).argmax(dim=1).item())

    def observe(self, obs, action, reward, next_obs, terminated):
        self.replay.add(obs, action, reward, next_obs, terminated)

    def update(self) -> Optional[float]:
        cfg = self.cfg
        if self.replay.size < cfg.batch_size:
            return None
        obs, actions, rewards, next_obs, terminal = self.replay.sample(
            cfg.batch_size, self.device)
        q_sa = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Ablation: when disabled, online weights determine both sides.
            bootstrap_net = self.online if cfg.disable_target else self.target
            if cfg.double_dqn:
                # Selection by online network; evaluation by target network.
                next_action = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = bootstrap_net(next_obs).gather(1, next_action).squeeze(1)
            else:
                next_q = bootstrap_net(next_obs).max(dim=1).values
            target_q = rewards + cfg.gamma * (1.0 - terminal) * next_q
        loss = F.smooth_l1_loss(q_sa, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def sync_target(self):
        if not self.cfg.disable_target:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path) -> None:
        # Saving state_dict rather than pickled class is safer and portable.
        torch.save({
            'model_state_dict': self.online.state_dict(),
            'obs_dim': self.obs_dim, 'n_actions': self.n_actions,
            'config': asdict(self.cfg),
        }, path)

    @classmethod
    def load(cls, path, device='cpu') -> 'DQNAgent':
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=device)
        cfg = DQNConfig(**payload['config'])
        obj = cls(int(payload['obs_dim']), int(payload['n_actions']), cfg, device=device)
        obj.online.load_state_dict(payload['model_state_dict'])
        obj.sync_target()
        obj.online.eval()
        return obj

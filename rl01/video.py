"""Record a greedy trained policy as an actual GIF (not stock footage)."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio

from .common import make_env
from .dqn import DQNAgent
from .tabular import QLearningAgent


def render_gif(run_dir: Path, output: Path, seed: int = 777, fps: int = 15) -> dict:
    run_dir = Path(run_dir)
    with (run_dir / 'config.json').open(encoding='utf-8') as f:
        cfg = json.load(f)
    env = make_env(cfg['environment'], render_mode='rgb_array')
    try:
        agent = (QLearningAgent.load(run_dir / 'policy.npz')
                 if cfg['agent']['algo'] == 'tabular'
                 else DQNAgent.load(run_dir / 'policy.pt', device='cpu'))
        obs, _ = env.reset(seed=seed)
        frames = [env.render()]
        episode_return = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, 0.0))
            episode_return += float(reward)
            frames.append(env.render())
            done = bool(terminated or truncated)
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output), frames, duration=1.0 / fps, loop=0)
        return {'gif': str(output), 'episode_return': episode_return,
                'frames': len(frames), 'seed': seed}
    finally:
        env.close()

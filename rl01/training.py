"""Training/evaluation protocol shared by tabular Q-learning and DQN."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .common import (append_csv, ensure_discrete_env, linear_epsilon, make_env,
                     metadata, read_csv, seed_env, seed_everything, write_json)
from .dqn import DQNAgent, DQNConfig
from .tabular import CARTPOLE_CLIP_RANGES, Discretizer, QLearningAgent


THRESHOLDS = {'CartPole-v1': 475.0, 'Acrobot-v1': -100.0,
              'LunarLander-v3': 200.0}  # Explicit assignment thresholds.


def evaluate(agent, env, seed: int, episodes: int) -> tuple[np.ndarray, int]:
    """Deterministic greedy evaluation; no training updates or epsilon exploration."""
    returns = []
    env_steps = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done, ep_return = False, 0.0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
            env_steps += 1
        returns.append(ep_return)
    return np.asarray(returns, dtype=np.float64), env_steps


def train_one(*, env_id: str, algo: str, config_name: str, run_dir: Path,
              steps: int, seed: int, eval_every: int, eval_episodes: int,
              final_eval_episodes: int, epsilon_start: float,
              epsilon_end: float, epsilon_decay_fraction: float,
              bins: Optional[int] = None, dqn_cfg: Optional[DQNConfig] = None,
              alpha: float = 0.15, gamma: float = 0.99,
              clip_ranges=None, device: str = 'cpu') -> dict:
    """One seed is an independent experiment; all variants get equal training steps.

    Training steps exclude the separate evaluation environment. Final success
    is assessed on 100 fresh test episodes with disjoint seeds.
    """
    if steps <= 0 or eval_every <= 0 or eval_episodes <= 0 or final_eval_episodes <= 0:
        raise ValueError('steps, eval_every, and evaluation episodes must be positive')
    if algo not in ('tabular', 'dqn'):
        raise ValueError('algo must be tabular or dqn')
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    # Torch small MLPs generally run more efficiently with 1 CPU thread.
    if device == 'cpu':
        torch.set_num_threads(1)

    training_env = make_env(env_id)
    eval_env = make_env(env_id)
    try:
        obs_dim, n_actions = ensure_discrete_env(training_env)
        if algo == 'tabular':
            if env_id != 'CartPole-v1':
                raise ValueError('CartPole-v1 is the included tabular discretization')
            ranges = (CARTPOLE_CLIP_RANGES if clip_ranges is None
                      else np.asarray(clip_ranges, dtype=np.float64))
            agent = QLearningAgent(Discretizer(int(bins), ranges),
                                   n_actions, alpha=alpha, gamma=gamma, seed=seed)
            config = {'algo': algo, 'bins': bins, 'alpha': alpha, 'gamma': gamma,
                      'clip_ranges': ranges.tolist()}
        else:
            if dqn_cfg is None:
                raise ValueError('dqn_cfg required for DQN')
            agent = DQNAgent(obs_dim, n_actions, dqn_cfg, seed=seed, device=device)
            config = {'algo': algo, 'dqn': asdict(dqn_cfg)}

        setup = {
            'environment': env_id, 'config_name': config_name, 'seed': seed,
            'steps': steps, 'eval_every': eval_every,
            'checkpoint_eval_episodes': eval_episodes,
            'final_holdout_eval_episodes': final_eval_episodes,
            'epsilon': {'start': epsilon_start, 'end': epsilon_end,
                        'decay_fraction': epsilon_decay_fraction},
            'agent': config, 'machine': metadata(),
        }
        write_json(run_dir / 'config.json', setup)
        train_start = time.perf_counter()
        obs = seed_env(training_env, seed)
        episode_return = 0.0
        episode_length = 0
        episode_idx = 0
        last_loss = float('nan')
        eval_step_count = 0
        eval_seed = seed + 1_000_000  # same evaluation initial conditions per checkpoint

        def record_evaluation(step: int, phase: str, count: int, seed_base: int):
            nonlocal eval_step_count
            scores, count_steps = evaluate(agent, eval_env, seed_base, count)
            eval_step_count += count_steps
            row = {
                'environment': env_id, 'algorithm': algo,
                'variant': config_name, 'seed': seed,
                'env_steps': step, 'phase': phase,
                'n_episodes': count, 'mean_return': float(scores.mean()),
                'std_return': float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                'evaluation_env_steps': count_steps,
                'elapsed_wall_seconds': round(time.perf_counter() - train_start, 3),
            }
            append_csv(run_dir / 'evaluations.csv', row)
            # Keep episode-level results for uncertainty and auditing.
            for i, score in enumerate(scores):
                append_csv(run_dir / 'eval_episodes.csv', {
                    'phase': phase, 'env_steps': step, 'seed': seed_base + i,
                    'return': float(score),
                })
            return row

        initial_row = record_evaluation(0, 'checkpoint', eval_episodes, eval_seed)

        for step in range(1, steps + 1):
            epsilon = linear_epsilon(step - 1, steps, epsilon_start,
                                     epsilon_end, epsilon_decay_fraction)
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = training_env.step(action)
            if algo == 'tabular':
                agent.update(obs, action, reward, next_obs, bool(terminated))
            else:
                agent.observe(obs, action, reward, next_obs, bool(terminated))
                if step >= dqn_cfg.learning_starts and step % dqn_cfg.train_frequency == 0:
                    loss = agent.update()
                    if loss is not None:
                        last_loss = loss
                if (not dqn_cfg.disable_target and
                        step % dqn_cfg.target_update_frequency == 0):
                    agent.sync_target()

            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                episode_idx += 1
                append_csv(run_dir / 'train_episodes.csv', {
                    'environment': env_id, 'algorithm': algo,
                    'variant': config_name, 'seed': seed,
                    'episode': episode_idx, 'env_steps': step,
                    'episode_length': episode_length,
                    'train_return': episode_return, 'epsilon': epsilon,
                    'last_loss': last_loss,
                })
                obs, _ = training_env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                obs = next_obs

            if step % eval_every == 0 or step == steps:
                row = record_evaluation(step, 'checkpoint', eval_episodes, eval_seed)
                print(f'[{env_id} | {config_name} | seed={seed}] '
                      f'{step}/{steps} steps | greedy={row["mean_return"]:.1f} '
                      f'| eps={epsilon:.3f}', flush=True)

        # A separate holdout is used for the final >=100-episode criterion.
        final = record_evaluation(steps, 'final_holdout', final_eval_episodes,
                                  seed + 2_000_000)
        if algo == 'tabular':
            checkpoint = run_dir / 'policy.npz'
        else:
            checkpoint = run_dir / 'policy.pt'
        agent.save(checkpoint)
        elapsed = time.perf_counter() - train_start
        threshold = THRESHOLDS.get(env_id)
        checkpoint_rows = [r for r in read_csv(run_dir / 'evaluations.csv')
                           if r['phase'] == 'checkpoint']
        crossing_steps = next(
            (int(r['env_steps']) for r in checkpoint_rows
             if threshold is not None and float(r['mean_return']) >= threshold),
            None,
        )
        reached = bool(final['mean_return'] >= threshold) if threshold is not None else None
        result = {
            'environment': env_id, 'algorithm': algo, 'variant': config_name,
            'seed': seed, 'train_steps': steps,
            'initial_checkpoint_mean_return': initial_row['mean_return'],
            'final_holdout_mean_return': final['mean_return'],
            'final_holdout_std_return': final['std_return'],
            'final_holdout_n_episodes': final_eval_episodes,
            'threshold_in_assignment': threshold,
            'threshold_met_on_holdout': reached,
            'first_checkpoint_crossing_steps_exploratory': crossing_steps,
            'total_evaluation_steps_excluded_from_train_budget': eval_step_count,
            'wall_clock_seconds_including_evaluation': round(elapsed, 3),
            'model_file': checkpoint.name,
        }
        write_json(run_dir / 'summary.json', result)
        return result
    finally:
        training_env.close()
        eval_env.close()

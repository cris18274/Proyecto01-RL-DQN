"""Single entry point for required experiments, analysis, and agent videos.

    python -m rl01.cli smoke
    python -m rl01.cli tier1 --seeds 11,22,33 --steps 80000
    python -m rl01.cli tier2 --seeds 11,22,33 --steps 100000
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from .analysis import analyze
from .common import parse_seeds, write_json
from .dqn import DQNConfig
from .training import train_one
from .tabular import CARTPOLE_CLIP_RANGES
from .video import render_gif


def common_args(parser):
    parser.add_argument('--seeds', default='11,22,33',
                        help='comma-separated independent random seeds')
    parser.add_argument('--steps', type=int, default=None,
                        help='environment steps per run, excluding evaluation')
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--final-eval-episodes', type=int, default=100)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true',
                        help='delete existing output for an identical run and rerun')


def _execute(label, algo, env_id, seed, args, *, bins=None, cfg=None,
             eps_end=0.05, eps_decay=0.70, clip_ranges=None):
    out = args.out
    run_dir = out / f'{label}_seed{seed:04d}'
    if run_dir.exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f'{run_dir} already exists. '
                                  'Use --overwrite to replace it (training starts over).')
    return train_one(
        env_id=env_id, algo=algo, config_name=label, run_dir=run_dir,
        steps=args.steps, seed=seed, eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        epsilon_start=1.0, epsilon_end=eps_end,
        epsilon_decay_fraction=eps_decay,
        bins=bins, dqn_cfg=cfg,
        alpha=0.15, gamma=0.99, clip_ranges=clip_ranges, device='cpu',
    )


def tier1(args):
    args.out = args.out or Path('results/tier1')
    args.steps = args.steps if args.steps is not None else 80000
    seeds = parse_seeds(args.seeds)
    for bins in (6, 10, 20):
        for seed in seeds:
            _execute(f'tab_bins{bins}', 'tabular', 'CartPole-v1', seed,
                     args, bins=bins)
    if args.include_range_study:
        tighter = CARTPOLE_CLIP_RANGES.copy()
        tighter[1] = [-1.5, 1.5]
        tighter[3] = [-2.0, 2.0]
        for seed in seeds:
            _execute('tab_bins10_tight', 'tabular', 'CartPole-v1',
                     seed, args, bins=10, clip_ranges=tighter)
    cfg = DQNConfig(learning_rate=1e-3, learning_starts=1000,
                    target_update_frequency=1000)
    for seed in seeds:
        _execute('dqn_base', 'dqn', 'CartPole-v1', seed, args, cfg=cfg)
    analyze(args.out)


def tier2(args):
    args.out = args.out or Path('results/tier2')
    args.steps = args.steps if args.steps is not None else 100000
    seeds = parse_seeds(args.seeds)
    env_id = args.environment
    # OFAT: default + two alternative values for EACH of TWO parameters.
    base = DQNConfig(learning_rate=5e-4, target_update_frequency=1000,
                     learning_starts=1000)
    variants = [
        ('dqn_base', base),
        ('lr_0p00025', replace(base, learning_rate=2.5e-4)),
        ('lr_0p001', replace(base, learning_rate=1e-3)),
        ('target_250', replace(base, target_update_frequency=250)),
        ('target_4000', replace(base, target_update_frequency=4000)),
        ('ablation_no_target', replace(base, disable_target=True)),
    ]
    if args.include_double:
        variants.append(('tier3_double_dqn', replace(base, double_dqn=True)))
    for label, cfg in variants:
        for seed in seeds:
            _execute(label, 'dqn', env_id, seed, args, cfg=cfg,
                     eps_end=0.10, eps_decay=0.80)
    analyze(args.out)


def smoke(args):
    """Small verification run only: absolutely NOT scientific results."""
    args.out = args.out or Path('results/smoke')
    args.steps = args.steps if args.steps is not None else 320
    args.eval_every = min(args.eval_every, 160)
    args.eval_episodes = 2
    args.final_eval_episodes = 5
    cfg = DQNConfig(learning_rate=1e-3, batch_size=16,
                    learning_starts=32, buffer_capacity=1024,
                    train_frequency=4, target_update_frequency=64,
                    hidden_width=32)
    for label, algo, env, bins, spec in [
        ('tab_bins6', 'tabular', 'CartPole-v1', 6, None),
        ('dqn_base', 'dqn', 'CartPole-v1', None, cfg),
        ('ablation_no_target', 'dqn', 'Acrobot-v1', None,
         replace(cfg, disable_target=True)),
    ]:
        _execute(label, algo, env, 7, args, bins=bins, cfg=spec)
    print('Smoke test completed. DO NOT use its scores in the research paper.')


def build_parser():
    parser = argparse.ArgumentParser(description='Independent RL agents: Q-learning and DQN')
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ('smoke', 'tier1', 'tier2'):
        p = sub.add_parser(cmd)
        common_args(p)
        if cmd == 'tier1':
            p.add_argument('--include-range-study', action='store_true',
                           help='additional 10-bin clipping-range sensitivity study')
        if cmd == 'tier2':
            p.add_argument('--environment', choices=['Acrobot-v1', 'LunarLander-v3'],
                           default='Acrobot-v1')
            p.add_argument('--include-double', action='store_true',
                           help='add Tier 3: Double DQN (for two-person teams)')
    p = sub.add_parser('analyze')
    p.add_argument('--tier-dir', type=Path, required=True)
    p = sub.add_parser('video')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--fps', type=int, default=15)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == 'tier1':
        tier1(args)
    elif args.command == 'tier2':
        tier2(args)
    elif args.command == 'smoke':
        smoke(args)
    elif args.command == 'analyze':
        analyze(args.tier_dir)
    elif args.command == 'video':
        result = render_gif(args.run_dir, args.out, seed=args.seed, fps=args.fps)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

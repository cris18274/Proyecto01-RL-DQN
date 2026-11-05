"""Regenerate every paper figure from the original experiment CSV files.

Usage (from repository root):
    python report/make_paper_figures.py
    python report/make_paper_figures.py --results results --out report/figures_reproduced

This script NEVER trains agents and NEVER fabricates returns. Checkpoint curves
use evaluations.csv phase=checkpoint; bar charts use summary_by_variant.csv.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing source data: {path}')
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def curves_for_tier(tier: Path, variants: list[str]) -> dict:
    """Return one curve per seed and variant, keyed by environment steps."""
    traces = defaultdict(list)
    for variant in variants:
        for run_dir in sorted(tier.glob(f'{variant}_seed*')):
            rows = csv_rows(run_dir / 'evaluations.csv')
            data = {int(r['env_steps']): float(r['mean_return'])
                    for r in rows if r['phase'] == 'checkpoint'}
            if data:
                traces[variant].append(data)
        if not traces[variant]:
            raise RuntimeError(f'No checkpoint traces for {variant} in {tier}')
    return traces


def plot_curves(tier: Path, output: Path, *, variants: list[str],
                env_name: str, threshold: float) -> None:
    traces = curves_for_tier(tier, variants)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for name in variants:
        runs = traces[name]
        common_x = sorted(set.intersection(*(set(r) for r in runs)))
        series = np.array([[r[x] for x in common_x] for r in runs], dtype=float)
        mean = series.mean(axis=0)
        sd = series.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        (line,) = ax.plot(common_x, mean, linewidth=2, label=f'{name} (n={len(runs)})')
        ax.fill_between(common_x, mean - sd, mean + sd, color=line.get_color(),
                        alpha=0.15, linewidth=0)
    ax.axhline(threshold, ls='--', color='gray', alpha=0.85,
               label=f'Course criterion: {threshold:g}')
    ax.set(title=f'{env_name}: greedy checkpoint return, mean ± SD',
           xlabel='Environment training steps (evaluation excluded)',
           ylabel='Mean return over 10 greedy episodes')
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, fontsize=8, loc='best')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summary_for_tier(tier: Path) -> dict:
    rows = csv_rows(tier / 'summary_by_variant.csv')
    return {r['variant']: (float(r['final_mean_across_seeds']),
                           float(r['std_across_seeds'])) for r in rows}


def plot_cartpole_budget(results: Path, output: Path):
    a = summary_for_tier(results / 'tier1')
    b = summary_for_tier(results / 'tier1_200k')
    keys = ['tab_bins6', 'tab_bins10', 'tab_bins20', 'dqn_base']
    labels = ['6 bins', '10 bins', '20 bins', 'DQN']
    for k in keys:
        if k not in a or k not in b:
            raise RuntimeError(f'Missing {k} in CartPole summary CSVs')
    x = np.arange(len(keys)); w = 0.36
    m80, s80 = np.array([a[k] for k in keys]).T
    m200, s200 = np.array([b[k] for k in keys]).T
    fig, ax = plt.subplots(figsize=(8.5, 4.75))
    ax.bar(x-w/2, m80, width=w, yerr=s80, capsize=3, label='80k steps', alpha=0.90)
    ax.bar(x+w/2, m200, width=w, yerr=s200, capsize=3, label='200k steps', alpha=0.90)
    ax.axhline(475, color='gray', ls='--', alpha=0.85,
               label='Course criterion: 475')
    ax.set(xticks=x, xticklabels=labels,
           ylabel='100-episode final return: mean ± SD across 3 seeds',
           title='CartPole: independently trained 80k vs 200k budgets')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_acrobot_bars(results: Path, output: Path):
    scores = summary_for_tier(results / 'tier2')
    keys = ['dqn_base', 'lr_0p00025', 'lr_0p001',
            'target_250', 'target_4000', 'ablation_no_target']
    labels = ['DQN base', 'LR .00025', 'LR .001',
              'Target 250', 'Target 4000', 'No target']
    for k in keys:
        if k not in scores:
            raise RuntimeError(f'Missing {k} in Acrobot summary CSV')
    vals, errs = np.array([scores[k] for k in keys]).T
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ax.bar(np.arange(len(keys)), vals, yerr=errs, capsize=3, alpha=0.9)
    ax.axhline(-100, color='gray', ls='--', label='Course criterion: -100')
    ax.set(xticks=np.arange(len(keys)), xticklabels=labels,
           ylabel='100-episode final return: mean ± SD across 3 seeds',
           title='Acrobot: parameter sensitivity and no-target ablation')
    ax.tick_params(axis='x', labelrotation=18)
    ax.set_ylim(-560, -20)
    ax.grid(alpha=0.2, axis='y')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--out', type=Path, default=ROOT/'report'/'figures_reproduced')
    args = parser.parse_args()
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    plot_curves(args.results/'tier1', output/'tier1_curves_light.png',
                variants=['tab_bins6','tab_bins10','tab_bins20','dqn_base'],
                env_name='CartPole-v1 (80k)', threshold=475)
    plot_curves(args.results/'tier1_200k', output/'tier1_200k_curves_light.png',
                variants=['tab_bins6','tab_bins10','tab_bins20','dqn_base'],
                env_name='CartPole-v1 (200k)', threshold=475)
    plot_cartpole_budget(args.results, output/'paper_cartpole_budget.png')
    plot_curves(args.results/'tier2', output/'tier2_curves_light.png',
                variants=['dqn_base','target_250','target_4000','ablation_no_target'],
                env_name='Acrobot-v1 (100k)', threshold=-100)
    plot_acrobot_bars(args.results, output/'tier2_bars_light.png')
    print('Regenerated five manuscript figures from saved experimental CSVs:')
    for p in sorted(output.glob('*.png')):
        print(' ',p)


if __name__ == '__main__':
    main()

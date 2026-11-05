"""Post-process actual runs. No simulated metrics or fabricated curves."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import read_csv, write_json


def _summary_records(tier_dir: Path):
    paths = sorted(tier_dir.glob('*/summary.json'))
    if not paths:
        raise FileNotFoundError(f'No completed run summaries found in {tier_dir}')
    rows = []
    for path in paths:
        with path.open(encoding='utf-8') as f:
            rows.append(json.load(f))
    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(tier_dir: Path):
    tier_dir = Path(tier_dir)
    summaries = _summary_records(tier_dir)
    _write_csv(tier_dir / 'final_by_seed.csv', summaries)
    groups = defaultdict(list)
    for s in summaries:
        groups[s['variant']].append(s)
    output = []
    for variant, rows in sorted(groups.items()):
        values = np.array([r['final_holdout_mean_return'] for r in rows], dtype=float)
        times = np.array([r['wall_clock_seconds_including_evaluation'] for r in rows])
        std = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
        # t critical 95%, approximate for n > 10; report as exploratory due small n.
        t95 = {1: None, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
               6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        critical = t95.get(len(values), 1.96)
        margin = (critical * std / math.sqrt(len(values))
                  if critical is not None else float('nan'))
        output.append({
            'variant': variant, 'environment': rows[0]['environment'],
            'algorithm': rows[0]['algorithm'], 'n_seeds': len(values),
            'final_mean_across_seeds': float(values.mean()),
            'std_across_seeds': std,
            'ci95_low_exploratory': float(values.mean() - margin),
            'ci95_high_exploratory': float(values.mean() + margin),
            'total_wall_hours_all_seeds': float(times.sum() / 3600.0),
            'threshold_assignment': rows[0]['threshold_in_assignment'],
            'seeds_that_met_threshold': sum(r['threshold_met_on_holdout'] is True for r in rows),
        })
    _write_csv(tier_dir / 'summary_by_variant.csv', output)
    write_json(tier_dir / 'summary_by_variant.json', output)
    _paired_baseline(tier_dir, summaries)
    _plot_curves(tier_dir, summaries)
    print(f'Written: {tier_dir / "summary_by_variant.csv"}')
    for row in output:
        print(f'  {row["variant"]:25s} '
              f'mean={row["final_mean_across_seeds"]:9.2f} '
              f'SD={row["std_across_seeds"]:8.2f} '
              f'seeds={row["n_seeds"]}')


def _plot_curves(tier_dir: Path, summaries: list[dict]):
    traces = defaultdict(list)
    for summary in summaries:
        run_dir = tier_dir / f'{summary["variant"]}_seed{summary["seed"]:04d}'
        points = [r for r in read_csv(run_dir / 'evaluations.csv')
                  if r['phase'] == 'checkpoint']
        if points:
            traces[summary['variant']].append({int(r['env_steps']): float(r['mean_return'])
                                               for r in points})
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for variant, per_seed in sorted(traces.items()):
        # Intersection ensures every plotted mean/SD uses the same seeds at each x.
        common_x = sorted(set.intersection(*(set(trace) for trace in per_seed)))
        if not common_x:
            continue
        curves = np.array([[trace[s] for s in common_x] for trace in per_seed])
        mu = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mu)
        line, = ax.plot(common_x, mu, linewidth=1.8, label=f'{variant} (n={len(curves)})')
        if len(curves) > 1:
            ax.fill_between(common_x, mu - sd, mu + sd,
                            color=line.get_color(), alpha=0.16, linewidth=0)
    threshold = summaries[0].get('threshold_in_assignment')
    if threshold is not None:
        ax.axhline(float(threshold), color='gray', linestyle='--', alpha=0.7,
                   label=f'Assignment threshold: {threshold:g}')
    ax.set_xlabel('Training environment steps (evaluation steps excluded)')
    ax.set_ylabel('Mean greedy evaluation return (10 episodes per seed)')
    ax.set_title(f'{summaries[0]["environment"]} | learning curves, ±1 SD across seeds')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(tier_dir / 'learning_curves.png', dpi=180)
    fig.savefig(tier_dir / 'learning_curves.pdf')
    plt.close(fig)

    # A second chart with all raw seeds makes instability transparent.
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for variant, runs in sorted(traces.items()):
        for idx, points in enumerate(runs):
            x = sorted(points)
            y = [points[s] for s in x]
            ax.plot(x, y, alpha=0.55, linewidth=1, label=variant if idx == 0 else None)
    ax.set_xlabel('Training environment steps')
    ax.set_ylabel('Greedy evaluation return')
    ax.set_title(f'{summaries[0]["environment"]} | raw trajectories for each seed')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(tier_dir / 'individual_seeds.png', dpi=180)
    plt.close(fig)


def _paired_baseline(tier_dir: Path, summaries: list[dict]):
    """Within-seed contrasts (not independent two-sample comparisons).

    Means are exploratory and NOT a claim of statistical significance.
    """
    by_variant = defaultdict(dict)
    for row in summaries:
        by_variant[row['variant']][row['seed']] = row
    baseline = by_variant.get('dqn_base', {})
    if not baseline:
        return
    rows = []
    for variant, seeds in sorted(by_variant.items()):
        if variant == 'dqn_base':
            continue
        common_seeds = sorted(set(seeds) & set(baseline))
        if not common_seeds:
            continue
        diff = np.asarray([
            float(seeds[seed]['final_holdout_mean_return']) -
            float(baseline[seed]['final_holdout_mean_return'])
            for seed in common_seeds
        ], dtype=float)
        n = len(diff)
        sample_sd = float(diff.std(ddof=1)) if n > 1 else float('nan')
        t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
               6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        critical = t95.get(n, 1.96)
        margin = critical * sample_sd / math.sqrt(n) if n > 1 else float('nan')
        rows.append({
            'variant': variant, 'baseline': 'dqn_base',
            'paired_seeds': ','.join(str(seed) for seed in common_seeds),
            'n_pairs': n,
            'mean_return_difference_variant_minus_baseline': float(diff.mean()),
            'paired_sd': sample_sd,
            'paired_ci95_low_exploratory': float(diff.mean() - margin),
            'paired_ci95_high_exploratory': float(diff.mean() + margin),
        })
    _write_csv(tier_dir / 'paired_vs_dqn_base.csv', rows)

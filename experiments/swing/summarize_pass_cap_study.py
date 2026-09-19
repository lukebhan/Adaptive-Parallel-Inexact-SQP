#!/usr/bin/env python3
"""Audit every attempted Swing step and plot the paired cap experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("AOTD-rebuild", "AOTD-sketch")
LABELS = {"AOTD-rebuild": "GMRES", "AOTD-sketch": "Sketch"}
MS = (4, 10, 20, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    data, summary, failures = {}, [], []
    for cap in (6, 50):
        rows = [json.loads(p.read_text()) for p in sorted(
            (root / f"cap{cap}/swing/records").glob("*.json"))]
        rows = [r for r in rows if r["method"] in METHODS]
        assert len(rows) == 40
        assert len({(r['method'], r['M'], r['seed']) for r in rows}) == 40
        assert all(r["max_inner_passes"] == cap for r in rows)
        data[cap] = rows
        for method in METHODS:
            for m in MS:
                group = [r for r in rows if r["method"] == method and r["M"] == m]
                assert len(group) == 5
                steps = [t for r in group for t in r["trajectory"]]
                passes = [p for t in steps for p in t["passes"]]
                for r in group:
                    for t in r["trajectory"]:
                        p = t["passes"][-1]
                        if p["outcome"] != "accept":
                            failures.append(dict(cap=cap, method=method, M=m,
                                seed=r['seed'], tau=t['tau'], passes=len(t['passes']),
                                outcome=p['outcome'], residual_ratio=p['r_norm']/p['acc_rhs'],
                                alpha=t['alpha'], line_search_accepted=t['accepted']))
                        else:
                            assert p['r_norm'] <= p['acc_rhs']
                            assert p['gdir'] <= p['descent_rhs']
                summary.append(dict(
                    cap=cap, method=method, M=m,
                    converged=sum(r['converged'] and r['kkt'] < 1e-6 for r in group),
                    steps=len(steps),
                    ungated_steps=sum(t['passes'][-1]['outcome'] != 'accept' for t in steps),
                    failed_linesearches=sum(not t['accepted'] for t in steps),
                    mean_flops=float(np.mean([r['flops'] for r in group])),
                    std_flops=float(np.std([r['flops'] for r in group])),
                    mean_steps=float(np.mean([len(r['trajectory']) for r in group])),
                    mean_passes=float(np.mean([sum(len(t['passes']) for t in r['trajectory']) for r in group])),
                    max_passes=max(len(t['passes']) for t in steps),
                    max_overlap=max(max(p['b_used']) for p in passes),
                    min_local_tolerance=min(min(p['eps_used']) for p in passes),
                    max_kkt=max(r['kkt'] for r in group),
                    min_kkt=min(r['kkt'] for r in group),
                    min_alpha=min(t['alpha'] for t in steps),
                    desc_failures=sum(p['outcome'] == 'desc_fail' for p in passes),
                    max_last_accuracy_ratio=max(t['passes'][-1]['r_norm']/t['passes'][-1]['acc_rhs'] for t in steps),
                ))
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'ungated_steps.json').write_text(json.dumps(failures, indent=2) + '\n')
    with (root / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    index = {(r['cap'], r['method'], r['M']): r for r in summary}

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = {6: '#d55e00', 50: '#0072b2'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), layout='constrained')
    for ax, method in zip(axes[0], METHODS):
        x = np.arange(len(MS))
        for cap, dx in ((6, -.18), (50, .18)):
            cells = [index[cap, method, m] for m in MS]
            ax.bar(x+dx, [r['mean_flops']/1e7 for r in cells], width=.34,
                   yerr=[r['std_flops']/1e7 for r in cells], capsize=3,
                   color=colors[cap], label=f'Cap {cap}')
        ax.set(xticks=x, xticklabels=MS, xlabel='Subdomains M',
               ylabel='Modeled work (10⁷ FLOPs)', title=LABELS[method] + ': mean ± SD, five seeds')
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    for method, marker in zip(METHODS, ('o', 's')):
        for cap, style in ((6, '--'), (50, '-')):
            cells = [index[cap, method, m] for m in MS]
            ax.plot(range(len(MS)), [100*r['ungated_steps']/r['steps'] for r in cells],
                    marker=marker, linestyle=style, color=colors[cap],
                    label=f'{LABELS[method]}, cap {cap}')
    ax.set(xticks=range(len(MS)), xticklabels=MS, xlabel='Subdomains M',
           ylabel='Steps without passing both gates (%)', title='All attempted outer steps')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[1, 1]
    for cap, marker in ((50, 'o'), (6, 'x')):
        row = next(r for r in data[cap] if r['method'] == METHODS[0] and r['M'] == 10 and r['seed'] == 1)
        passes = row['trajectory'][0]['passes']
        ax.semilogy(range(1, len(passes)+1), [p['r_norm']/p['acc_rhs'] for p in passes],
                    marker=marker, color=colors[cap], label=f'Cap {cap}')
    ax.axhline(1, color='black', linestyle=':', label='Accuracy threshold')
    ax.set(xlabel='Pass in first outer step (1-based)', ylabel='Residual / accuracy threshold',
           title='GMRES, M=10, seed 1')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(root / 'swing_cap_comparison.pdf')
    fig.savefig(root / 'swing_cap_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharey=True, layout='constrained')
    for i, method in enumerate(METHODS):
        for j, m in enumerate(MS):
            ax = axes[i, j]
            for cap in (6, 50):
                group = [r for r in data[cap] if r['method'] == method and r['M'] == m]
                curves = [[t['grad_L_norm'] for t in r['trajectory']] + [r['kkt']] for r in group]
                length = max(map(len, curves))
                arr = np.log10(np.maximum([c + [c[-1]]*(length-len(c)) for c in curves], 1e-30))
                mean, std = arr.mean(0), arr.std(0)
                ax.semilogy(range(length), 10**mean, color=colors[cap], label=f'Cap {cap}')
                ax.fill_between(range(length), 10**(mean-std), 10**(mean+std), color=colors[cap], alpha=.15)
            ax.axhline(1e-6, color='black', linestyle=':', linewidth=.8)
            ax.set(title=f'{LABELS[method]}, M={m}', xlabel='Completed outer steps')
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel('KKT residual')
            if i == 0 and j == 0:
                ax.legend(frameon=False)
    fig.savefig(root / 'swing_cap_convergence.pdf')
    fig.savefig(root / 'swing_cap_convergence.png', dpi=180)
    plt.close(fig)

    lines = ['# Swing accuracy-pass cap experiment', '',
             'Paired fresh runs: N=1000; M=4,10,20,50; five initial-condition seeds; '
             'GMRES and sketch; caps 6 and 50. All other solver settings match between '
             'paired runs, including 100 local Krylov iterations and overlap cap 110. '
             'Each case starts a fresh Python process (sketch RNG seed 0).', '',
             'The original fallback after cap exhaustion is retained in this experiment. '
             'A gate failure counts any applied outer step whose final pass did not '
             'satisfy both the implemented accuracy and descent checks. Counts below '
             'use actual trajectory steps; the existing outer_iters field also counts '
             'the final convergence check.', '',
             '| Solver | M | Converged (6 → 50) | Ungated steps (6 → 50) | Mean work, 10⁷ FLOPs (6 → 50) | Work ratio | Mean applied steps (6 → 50) | Max passes at 50 | Max overlap at 50 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for m in MS:
            old, new = index[6, method, m], index[50, method, m]
            lines.append(f"| {LABELS[method]} | {m} | {old['converged']}/5 → {new['converged']}/5 | "
                f"{old['ungated_steps']}/{old['steps']} → {new['ungated_steps']}/{new['steps']} | "
                f"{old['mean_flops']/1e7:.2f} → {new['mean_flops']/1e7:.2f} | "
                f"{new['mean_flops']/old['mean_flops']:.3f} | {old['mean_steps']:.1f} → {new['mean_steps']:.1f} | "
                f"{new['max_passes']} | {new['max_overlap']} |")
    for cap in (6, 50):
        cells = [r for r in summary if r['cap'] == cap]
        lines.extend(['', f"Cap {cap}: {sum(r['converged'] for r in cells)}/40 converged; "
            f"{sum(r['ungated_steps'] for r in cells)}/{sum(r['steps'] for r in cells)} ungated steps; "
            f"{sum(r['failed_linesearches'] for r in cells)} failed line searches; "
            f"maximum {max(r['max_passes'] for r in cells)} passes in one step; "
            f"final KKT range {min(r['min_kkt'] for r in cells):.3e}–{max(r['max_kkt'] for r in cells):.3e}."])
    old_by_key = {(r['method'], r['M'], r['seed']): r for r in data[6]}
    # Check that the paired experiments have identical first-step prefixes.
    # Divergence after the first cap-6 exhaustion is the intended treatment.
    for row in data[50]:
        old = old_by_key[row['method'], row['M'], row['seed']]
        prefix = min(len(old['trajectory'][0]['passes']), len(row['trajectory'][0]['passes']))
        assert old['trajectory'][0]['passes'][:prefix] == row['trajectory'][0]['passes'][:prefix]
    cost_change = max(abs(r['cost']-old_by_key[r['method'], r['M'], r['seed']]['cost']) for r in data[50])
    lines.extend(['', f'Maximum absolute final objective change between paired runs: {cost_change:.3e}.',
        'All 40 paired cases have identical logged first-step prefixes before the cap can change behavior.', ''])
    remaining = [f for f in failures if f['cap'] == 50]
    if remaining:
        lines.extend(['## Remaining ungated steps at cap 50', '',
            '| Solver | M | Seed | Outer step (0-based) | Residual / threshold | Applied alpha |',
            '|---|---:|---:|---:|---:|---:|'])
        lines.extend(f"| {LABELS[f['method']]} | {f['M']} | {f['seed']} | {f['tau']} | "
                     f"{f['residual_ratio']:.6g} | {f['alpha']:.6g} |" for f in remaining)
        lines.extend(['', '**The 50-pass limit does not eliminate ungated steps on Swing.** '
                      'Convergence of these runs still does not validate strict gate enforcement.', ''])
    diagnostic = root / 'diagnostics/summary.json'
    if diagnostic.exists():
        lines.extend(['## Supplemental diagnostic (GMRES, M=20, seed 2)', '',
            '| Pass / overlap allowance | First-step passes | Last outcome | Residual / threshold | Total modeled FLOPs |',
            '|---|---:|---|---:|---:|'])
        for row in json.loads(diagnostic.read_text()):
            lines.append(f"| {row['case']} | {row['first_step_passes']} | {row['outcome']} | "
                         f"{row['ratio']:.6f} | {row['work']:.6g} |")
        lines.extend(['', 'With 200 passes allowed and the original overlap cap 110, this case '
            'passes both gates on pass 121. Enlarging the overlap allowance to 1000 alone '
            'does not resolve the first-step failure within 50 passes. Thus the observed '
            'plateau at cap 50 cannot be attributed solely to the overlap bound.', '',
            'Reproduce these separate diagnostics:', '', '```bash',
            'python experiments/swing/run_pass_cap_study.py --output results/cap_study/diagnostics --one 200 AOTD-rebuild 20 2',
            'python results/cap_study/diagnostics/increase_overlap.py', '```', ''])
    lines.extend(['',
        '## Figures and provenance', '',
        '- [Cap comparison](swing_cap_comparison.pdf) and [paired convergence](swing_cap_convergence.pdf).',
        '- Regenerated paper-style cap-50 [convergence](cap50/swing/swing_convergence.pdf) and '
        '[adaptation/work](cap50/swing/swing_adaptation.pdf); tables are in the same directory.',
        '- Fixed FOTD and nonlinear baselines in the paper-style plots/tables are unchanged archived '
        'runs from data/swing.json.gz. All adaptive curves and entries are fresh. Timings combine '
        'different execution conditions and are not a controlled speedup comparison.',
        '- Work uses the repository FLOP model. No runtime or accuracy claims beyond these cases '
        'follow from this experiment. A finite cap still needs explicit failure handling for general use.',
        '- Existing acc_relax=10, cost-only Hessian/merit-gradient implementation, hybrid update, '
        'overlap cap 110 and local solver settings are unchanged. Passing these implemented gates '
        'does not settle the separate manuscript/implementation differences.',
        '- Paired convergence plots use geometric means and one log-standard-deviation bands; '
        'shorter histories are extended with their final residual.',
        '- Fresh cap-6 sketch runs use a controlled per-case stream and may differ from the '
        'archived sketch runs, whose RNG depends on worker call order.', '',
        'Reproduce from the development repo root:', '',
        '```bash', 'python experiments/swing/run_pass_cap_study.py', '```', '',
        'Machine-readable summaries: summary.json and summary.csv. Every ungated step is '
        'listed in ungated_steps.json. Full trajectories are retained under cap6/ and cap50/.', ''])
    (root / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:20]))


if __name__ == '__main__':
    main()

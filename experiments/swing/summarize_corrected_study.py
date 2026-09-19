#!/usr/bin/env python3
"""Audit strict acceptance and report successes and failures without filtering cases."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ('AOTD-rebuild', 'AOTD-sketch')
MS = (4, 10, 20, 50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    records = [json.loads(p.read_text()) for p in sorted((root/'swing/records').glob('*.json'))]
    assert len(records) == 160 and all(r['schema'] == 2 and 'error' not in r for r in records)
    local_checks, applied_steps, rejected_attempts = 0, 0, 0
    for r in records:
        assert r['gauss_newton'] is False
        for t in r['trajectory']:
            for p in t['passes']:
                for check in p['local_solves']:
                    local_checks += 1
                    assert check['converged'] == (check['residual_norm'] <= check['residual_threshold'])
            if t['step_applied']:
                applied_steps += 1
                assert t['accepted'] and t['failure_reason'] is None and t['alpha'] > 0
                assert all(c['converged'] for c in t['passes'][-1]['local_solves'])
                if r['method'] in METHODS:
                    p = t['passes'][-1]
                    assert p['outcome'] == 'accept'
                    assert p['r_norm'] <= p['acc_rhs']
                    assert p['gdir'] <= p['descent_rhs']
            else:
                rejected_attempts += 1
                assert not t['accepted'] and t['failure_reason'] and t['alpha'] == 0
        assert r['converged'] == (r['kkt'] <= 1e-6)
    summary = []
    for method in sorted({r['method_label'] for r in records}):
        for m in MS:
            group = [r for r in records if r['method_label'] == method and r['M'] == m]
            assert len(group) == 5
            success = [r for r in group if r['converged']]
            steps = [t for r in group for t in r['trajectory']]
            summary.append(dict(
                method=method, M=m, converged=len(success), cases=len(group),
                stop_reasons=dict(Counter(r['stop_reason'] for r in group)),
                mean_work_all=float(np.mean([r['flops'] for r in group])),
                mean_work_success=float(np.mean([r['flops'] for r in success])) if success else None,
                applied_steps=sum(t['step_applied'] for t in steps),
                rejected_attempts=sum(not t['step_applied'] for t in steps),
                max_shift=max(t['hessian_shift_max'] for t in steps),
                shifted_attempts=sum(t['hessian_shifted_blocks'] > 0 for t in steps),
                max_last_solve_shift=max(r['trajectory'][-1]['hessian_shift_max'] for r in group),
                max_kkt=max(r['kkt'] for r in group),
            ))
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (root/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    audit = dict(runs=len(records), local_residual_checks=local_checks,
                 applied_steps=applied_steps, rejected_attempts=rejected_attempts,
                 applied_local_accuracy_violations=0, applied_adaptive_gate_violations=0,
                 adaptive_stop_reasons=dict(Counter(r['stop_reason'] for r in records if r['method'] in METHODS)))
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')

    lookup = {(r['method'], r['M']): r for r in summary}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for j, method in enumerate(METHODS):
        cells = [lookup[method, m] for m in MS]
        ax = axes[0, j]
        ax.bar(range(4), [r['converged'] for r in cells], color='#0072b2')
        ax.set(xticks=range(4), xticklabels=MS, ylim=(0, 5.6), yticks=range(6),
               title='GMRES' if j == 0 else 'Sketch', ylabel='Converged cases / 5', xlabel='Subdomains M')
        for i, r in enumerate(cells):
            ax.text(i, r['converged']+.12, f"{r['converged']}/5", ha='center')
        ax.grid(axis='y', alpha=.2)
        ax = axes[1, j]
        ax.bar(range(4), [r['mean_work_all']/1e7 for r in cells], color='#0072b2', alpha=.7)
        for i, m in enumerate(MS):
            group = [r for r in records if r['method'] == method and r['M'] == m]
            for r in group:
                ax.scatter(i, r['flops']/1e7, marker='o' if r['converged'] else 'x',
                           color='black' if r['converged'] else '#d55e00', s=30, zorder=3)
        ax.set(xticks=range(4), xticklabels=MS, ylabel='Work spent (10⁷ modeled FLOPs)',
               xlabel='Subdomains M', title='All cases; × stopped without convergence')
        ax.grid(axis='y', alpha=.2)
    fig.savefig(root/'swing_correctness_summary.pdf')
    fig.savefig(root/'swing_correctness_summary.png', dpi=180)
    plt.close(fig)

    lines = ['# Corrected Swing experiment', '',
        'The direction model now starts from the full Lagrangian Hessian and applies '
        'the stagewise shift sigma_k=max(0, 1e-6-lambda_min(H_k)). The true, unshifted '
        'Hessian differentiates the augmented merit for both descent and Armijo tests. '
        'Gauss–Newton remains an optional direction model, but is disabled in this study.', '',
        'Every local solve is certified using ||Gamma_i d_i-b_i|| <= epsilon_i ||b_i|| '
        'in the original, unpreconditioned coordinates. GMRES candidate checks and '
        'sketch restart checks use this criterion. Certification matvecs are counted. '
        'A failed local solve, exhausted global acceptance loop, or failed line search '
        'stops the algorithm without applying its trial direction.', '',
        'Budgets remain 50 acceptance passes, 100 local solver iterations/matvecs '
        '(GMRES/sketch respectively), overlap 110, and 25 outer iterations. '
        'The existing acc_relax=10 and hybrid overlap/tolerance updates are unchanged.', '',
        f"Audit: {local_checks} local certificates; {applied_steps} applied steps; "
        f"{rejected_attempts} rejected outer attempts. **Zero applied local-residual "
        'violations and zero applied adaptive accuracy/descent violations.**', '',
        '| Solver | M | Converged | Stop reasons | Mean work, all cases (10⁷) | Mean work, successes (10⁷) | Max Hessian shift |',
        '|---|---:|---:|---|---:|---:|---:|']
    for method in METHODS:
        for m in MS:
            r = lookup[method, m]
            work_success = f"{r['mean_work_success']/1e7:.2f}" if r['mean_work_success'] is not None else '—'
            reasons = ', '.join(f'{k}: {v}' for k, v in r['stop_reasons'].items())
            lines.append(f"| {'GMRES' if method == METHODS[0] else 'Sketch'} | {m} | {r['converged']}/5 | "
                         f"{reasons} | {r['mean_work_all']/1e7:.2f} | {work_success} | {r['max_shift']:.3g} |")
    lines.extend(['', 'Failed-case work is work spent before termination, not a cost to convergence. '
        'All attempts and failures are retained in the records and figures. Fixed-method '
        'details, including every success count and stop reason, are in summary.json and summary.csv.', '',
        '## Interpretation and limits', '',
        '- A positive stagewise floor is a sufficient, stronger condition than reduced-Hessian '
        'positivity. It preserves the temporal block structure. It does not prove the paper’s '
        'additional local assumption ||Hhat-H||→0: persistent stagewise shifts are possible '
        'even at a constrained local minimizer. Shift sizes are logged on every attempted outer step.',
        '- Cost-only Gauss–Newton may be a valid solve model under the global assumptions, '
        'but it is not the derivative of the merit and does not generally recover the full '
        'Lagrangian Hessian near the solution.',
        '- These are combined corrections, not an ablation assigning the performance change '
        'to one individual fix. The earlier cap-only study in ../cap_study used the old '
        'derivative, old local stopping rule, and an unconditional cap fallback.',
        '- All 160 Newton-method cases (120 fixed + 40 adaptive) were rerun. Only the 125 '
        'unchanged nonlinear IPOPT/MultiShoot/ADMM/Schwarz baselines were taken from the archive.',
        '- Work is the repository’s dominant-operation model, augmented with the actual '
        'residual-check matvecs and preconditioner applications. It is not a count of every '
        'operation, including Hessian assembly. Concurrent timings are not controlled speedups.',
        '- This experiment checks the implemented criteria. It does not resolve separate '
        'issues involving acc_relax, spectral-norm estimates, theory constants or adaptive laws.', '',
        '## Artifacts', '',
        '- [Success and work summary](swing_correctness_summary.pdf).',
        '- Regenerated [convergence](swing/swing_convergence.pdf) and '
        '[adaptation/work](swing/swing_adaptation.pdf). Failed cases are marked explicitly.',
        '- Updated tables: swing/swing_highlight_table.tex and swing/swing_appendix_table.tex.',
        '- audit.json, summary.json, summary.csv, provenance.json; full traces in swing/records/.', '',
        'Reproduce from the dev repository root:', '', '```bash',
        'PYTHONPATH=src python -m unittest discover -s tests -v',
        'python experiments.py smoke',
        'python experiments/swing/run_corrected_study.py --workers 4', '```', ''])
    (root/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:22]))


if __name__ == '__main__':
    main()

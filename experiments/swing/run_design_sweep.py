#!/usr/bin/env python3
"""Screen adaptation rates, then validate one selection on Swing and fresh seeds.

Accuracy/descent tests, penalty updates, local solver caps and overlap bounds
remain fixed. Selection uses only the six previously failing GMRES cases.
"""

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import newton as N
from newton.baselines.common import initial_guess, pack_traj

MS = (4, 10, 20, 50)
SCREEN = ((20, 2), (20, 3), (50, 1), (50, 2), (50, 4), (50, 5))
CANDIDATES = {f'b{b}_e{e:g}': dict(varrho_b=float(b), varrho_eps=e)
              for b in (4, 8) for e in (.5, 1., 2.)}
SOURCE_HASH = hashlib.sha256(b''.join((ROOT/'src/newton'/name).read_bytes() for name in
    ('AOTD.py', 'AOTDsolver.py', 'ComputeKKT.py', 'CalculateAug.py', 'AOTDNLP.py', 'composition.py'))).hexdigest()


def run_one(job):
    out, label, method, m, seed = job
    cfg = N.AlgorithmConfig(
        M=m, mu=10., gauss_newton=False, xi_H=1e-6,
        max_outer_iters=25, tol_kkt=1e-6, use_preconditioner=True,
        max_overlap=110, adaptive=True, b0=4, eps_i_0=.1,
        eps_i_floor=1e-9, acc_relax=10., adapt_mode='hybrid',
        hybrid_b_min1=True, warm_start=True, max_inner_passes=50,
        nu=2., b_step=4, inner_solver=method, inner_rank_tol=1e-14,
        precond_type='ilu_schur', ilu_drop_tol=1e-2, max_inner_iters=100,
        **CANDIDATES[label],
    )
    method_label = 'AOTD-rebuild' if method == 'gmres_qlp' else 'AOTD-sketch'
    path = Path(out)/label/'records'/f"{method_label.lower().replace('-', '_')}_M{m}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('config') == config and cached.get('source_hash') == SOURCE_HASH:
            return cached
    importlib.import_module('newton.AOTDsolver')._SKETCH_RNG = np.random.default_rng(0)
    prob0 = N.make_swing_problem(1000, mode=1, pm_scale=0.)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=.25) + .15*rng.standard_normal(2*N.N_BUS)
    prob = N.make_swing_problem(1000, mode=1, pm_scale=0., x0=x0)
    xi, ui = initial_guess(prob)
    start = time.perf_counter()
    result = N.run_algorithm(prob, cfg, pack_traj(prob, xi, ui), np.zeros(N.n_lam(prob)))
    row = dict(
        schema=2, design=label, config=config, source_hash=SOURCE_HASH,
        method=method_label, method_label=method_label, N=1000, M=m, b=None, seed=seed,
        max_inner_passes=50, max_inner_iters=100, gauss_newton=False, xi_H=1e-6,
        sketch_rng_seed=0, converged=bool(result['converged']), stop_reason=result['stop_reason'],
        kkt=float(result['final_grad_L_norm']), feas=float(result['final_feas']),
        cost=float(result['final_cost']), flops=float(result['total_flops']),
        outer_iters=result['outer_iters'], inner_iters=result['total_inner_iters'],
        matvecs=result['total_matvecs'], b_list=result['b_list'], eps_i_list=result['eps_i_list'],
        wall_contended=time.perf_counter()-start, trajectory=result['trajectory'],
    )
    for step in row['trajectory']:
        if step['step_applied']:
            p = step['passes'][-1]
            assert p['outcome'] == 'accept' and p['r_norm'] <= p['acc_rhs']
            assert p['gdir'] <= p['descent_rhs']
            assert all(c['converged'] and c['residual_norm'] <= c['residual_threshold']
                       for c in p['local_solves'])
        else:
            assert step['alpha'] == 0
    path.write_text(json.dumps(row, default=float))
    return row


def batch(jobs, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_one, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            print(f"{row['design']:8} {row['method']:12} M={row['M']:2} seed={row['seed']:2} "
                  f"{row['stop_reason']:16} KKT={row['kkt']:.2e} work={row['flops']:.3e}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/design_sweep')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(candidates=CANDIDATES, screening_cases=SCREEN,
        screening_solver='gmres_qlp', selection='maximize successful cases; break ties by mean work over all six cases',
        validation_M=MS, validation_seeds=list(range(1, 6)),
        fresh_seed_M=[20,50], fresh_seeds=list(range(6,11)),
        validation_solvers=['gmres_qlp','sketch'], source_hash=SOURCE_HASH,
        note='Only varrho_b and varrho_eps change. No accuracy threshold or budget is relaxed. '
             'The original 40 cases are benchmark validation, not an independent test set. '
             'Seeds 6–10 are evaluated only after selection; no retuning uses them.')
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    screen = batch([(str(out), label, 'gmres_qlp', m, seed)
                    for label in CANDIDATES for m,seed in SCREEN], args.workers)
    summary = []
    for label in CANDIDATES:
        group = [r for r in screen if r['design'] == label]
        summary.append(dict(design=label, converged=sum(r['converged'] for r in group),
                            cases=len(group), mean_work=float(np.mean([r['flops'] for r in group]))))
    summary.sort(key=lambda r: (-r['converged'], r['mean_work'], r['design']))
    (out/'screening_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    best = summary[0]['design']
    (out/'selection.json').write_text(json.dumps(dict(design=best, **CANDIDATES[best]), indent=2)+'\n')
    print('SELECTED', best, summary[0], flush=True)
    validation = batch([(str(out), best, method, m, seed)
        for method in ('gmres_qlp','sketch') for m in MS for seed in range(1,6)], args.workers)
    fresh = batch([(str(out), best, method, m, seed)
        for method in ('gmres_qlp','sketch') for m in (20,50) for seed in range(6,11)], args.workers)
    (out/'validation.json').write_text(json.dumps(validation, default=float))
    (out/'fresh_seeds.json').write_text(json.dumps(fresh, default=float))
    # Same corrected fixed baselines; adaptive rows come from the selected design.
    baseline = ROOT/'results/corrected_swing/swing'
    work = out/'selected/swing'
    (work/'records').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline/'baselines_sigma06.json', work/'baselines_sigma06.json')
    for path in (baseline/'records').glob('*.json'):
        row = json.loads(path.read_text())
        if row['method'] not in ('AOTD-rebuild','AOTD-sketch'):
            shutil.copyfile(path, work/'records'/path.name)
    for row in validation:
        filename = f"{row['method'].lower().replace('-', '_')}_M{row['M']}_seed{row['seed']}.json"
        (work/'records'/filename).write_text(json.dumps(row, default=float))
    env = dict(os.environ, AOTD_WORK=str(work))
    subprocess.run([sys.executable, str(ROOT/'experiments/swing/plot_results.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT/'experiments/swing/summarize_design_sweep.py'),
                    '--input', str(out)], check=True)


if __name__ == '__main__':
    main()

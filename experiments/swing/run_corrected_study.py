#!/usr/bin/env python3
"""Rerun all Swing Newton methods with corrected derivatives and strict residual tests."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/corrected_swing')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    out = args.output.resolve()
    work = out / 'swing'
    work.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / 'data/swing.json.gz', 'rt') as stream:
        baselines = json.load(stream)['baselines_sigma06.json']
    (work / 'baselines_sigma06.json').write_text(json.dumps(baselines))
    files = [ROOT / 'src/newton' / name for name in (
        'AOTD.py', 'AOTDsolver.py', 'ComputeKKT.py', 'CalculateAug.py')]
    provenance = dict(
        adaptive_runs=40, fixed_newton_runs=120, nonlinear_baselines='archived; unchanged code',
        max_inner_passes=50, max_inner_iters=100, max_overlap=110,
        gauss_newton=False, xi_H=1e-6, sketch_rng_seed_per_run=0,
        workers=args.workers, strict_local_accuracy=True, strict_global_acceptance=True,
        note='Work includes physical residual certification matvecs. Timings are not controlled speedups.',
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    env = dict(os.environ, AOTD_WORK=str(work), AOTD_WORKERS=str(args.workers),
               AOTD_MAX_PASSES='50', PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    for script in ('run_comparison.py', 'run_sketch.py'):
        subprocess.run([sys.executable, str(HERE / script)], env=env, check=True)
    records = [json.loads(p.read_text()) for p in (work / 'records').glob('*.json')]
    if len(records) != 160 or any('error' in r or r['schema'] != 2 for r in records):
        raise RuntimeError('Incomplete corrected Swing study; inspect the records.')
    subprocess.run([sys.executable, str(HERE / 'plot_results.py')], env=env, check=True)
    subprocess.run([sys.executable, str(HERE / 'summarize_corrected_study.py'),
                    '--input', str(out)], env=env, check=True)


if __name__ == '__main__':
    main()

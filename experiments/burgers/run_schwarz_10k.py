#!/usr/bin/env python3
"""Run the resumable Schwarz baseline at N=10000, b=40, on the Burgers sweep seeds."""

import os
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"
import sys
import time
import json
import traceback
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "burgers"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
REC = os.path.join(WORK, "records")
os.makedirs(REC, exist_ok=True)
N_H, MU, B = 10000, 10.0, 40
M_LIST = [10, 50, 250]
SEEDS = [1, 2, 3, 4, 5]
SCHEMA = 3  # Matches the comparison sweep schema.


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "rt", os.path.join(HERE, "run_comparison.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rec_path(M, seed):
    return os.path.join(REC, f"N{N_H}_schwarz_b40_M{M}_seed{seed}.json")


def run_one(job):
    M, seed = job
    out = rec_path(M, seed)
    if os.path.exists(out):
        d = json.loads(Path(out).read_text())
        if "error" not in d and d.get("schema_version", 0) >= SCHEMA:
            return d
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    try:
        import newton as N

        rt = _sweep_module()
        prob = rt.make_problem(N, N_H, seed)
        t = time.perf_counter()
        res = N.solve_schwarz(prob, M=M, b=B, mu_pen=MU, max_outer=30, tol_kkt=1e-6)
        wall = time.perf_counter() - t
        e = res.extra
        # idealized-parallel wall: subproblem solves run on M workers (critical path = max), the rest is serial
        par_wall = float(
            wall - e.get("t_subsolve_serial", 0.0) + e.get("t_subsolve_parallel", 0.0)
        )
        rec = dict(
            schema_version=SCHEMA,
            N=N_H,
            M=M,
            seed=seed,
            b=B,
            method="Schwarz",
            method_label="Schwarz-b40",
            family="baseline",
            converged=bool(res.converged),
            kkt=float(res.final_grad_L_norm),
            feas=float(res.final_feas),
            cost=float(res.final_cost),
            flops=float(e.get("flops", float("nan"))),
            outer_iters=int(res.iters),
            inner_iters=int(e.get("ipopt_iters", -1)),
            wall_contended=float(wall),
            parallel_wall_contended=par_wall,
            t_subsolve_serial=float(e.get("t_subsolve_serial", float("nan"))),
            t_subsolve_parallel=float(e.get("t_subsolve_parallel", float("nan"))),
            history=[float(h) for h in (res.history or [])],
        )
        Path(out).write_text(json.dumps(rec, default=float))
        return rec
    except Exception as ex:
        rec = dict(
            schema_version=SCHEMA,
            N=N_H,
            M=M,
            seed=seed,
            b=B,
            method_label="Schwarz-b40",
            error=f"{type(ex).__name__}: {ex}",
            tb=traceback.format_exc(),
        )
        Path(out).write_text(json.dumps(rec))
        return rec


def main():
    jobs = [(M, s) for M in M_LIST for s in SEEDS]
    w = min(int(os.environ.get("AOTD_WORKERS", "1")), len(jobs), os.cpu_count() or 4)
    print(
        f"=== Burgers N=10k Schwarz b40 (exp10 problem): {len(jobs)} jobs / {w} workers ===",
        flush=True,
    )
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            if "error" in d:
                print(f"[ERR ] M={d['M']} s={d['seed']}: {d['error']}", flush=True)
            else:
                print(
                    f"[done] Schwarz M={d['M']:>3} s={d['seed']} conv={str(d['converged'])[:1]} "
                    f"kkt={d['kkt']:.1e} flops={d['flops']:.2e} out={d['outer_iters']} "
                    f"in={d['inner_iters']} wall={d['wall_contended']:.0f}s",
                    flush=True,
                )
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

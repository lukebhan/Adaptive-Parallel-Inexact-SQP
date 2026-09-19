#!/usr/bin/env python3
"""Run Swing nonlinear baselines with the same initial-condition seeds as AOTD."""

import os
from pathlib import Path
import sys
import json
import time
import traceback

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "swing"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
DATA = os.path.join(WORK, "baselines_sigma06.json")
MU, N_H = 10.0, 1000
M_LIST = [4, 10, 20, 50]
B_SCHWARZ = [1, 10, 40, 75]
SEEDS = [1, 2, 3, 4, 5]


def make_seed_problem(N, seed, kick=0.25):  # identical sigma=0.6 IC to the OTD sweep
    prob0 = N.make_swing_problem(N_H, mode=1, pm_scale=0.0)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=kick) + 0.6 * kick * rng.standard_normal(
        2 * N.N_BUS
    )
    return N.make_swing_problem(N_H, mode=1, pm_scale=0.0, x0=x0)


def base_record(method, M, seed, b, res, wall):
    e = res.extra
    # Parallel time replaces summed local solves with their critical path; IPOPT stays serial.
    t_ser = float(e.get("t_subsolve_serial", wall))
    t_par = float(e.get("t_subsolve_parallel", wall))
    return dict(
        method=method,
        M=M,
        seed=seed,
        b=b,
        family="baseline",
        converged=bool(res.converged),
        kkt=float(res.final_grad_L_norm),
        feas=float(res.final_feas),
        cost=float(res.final_cost),
        wall=float(wall),
        par_wall=float(wall - t_ser + t_par),
        t_subsolve_serial=t_ser,
        t_subsolve_parallel=t_par,
        flops=float(e.get("flops", float("nan"))),
        outer_iters=int(res.iters),
        inner_iters=int(e.get("ipopt_iters", -1)),
        history=[float(h) for h in (res.history or [])],
    )


def run_one(job):
    method, M, b, seed = job
    sys.path.insert(0, SRC)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import newton as N

        prob = make_seed_problem(N, seed)
        t = time.time()
        if method == "IPOPT":
            res = N.solve_ipopt(prob, tol_kkt=1e-6, max_iters=400)
        elif method == "Schwarz":
            res = N.solve_schwarz(prob, M=M, b=b, mu_pen=MU, max_outer=30, tol_kkt=1e-6)
        elif method == "MultiShoot":
            res = N.solve_multishoot(prob, M=M, max_outer=30, tol_kkt=1e-6)
        elif method == "ADMM":
            res = N.solve_admm(prob, M=M, rho=MU, max_outer=80, tol_kkt=1e-6)
        else:
            raise ValueError(method)
        return base_record(method, M, seed, b, res, time.time() - t)
    except Exception as e:
        return dict(
            method=method,
            M=M,
            seed=seed,
            b=b,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        )


def main():
    jobs = [("IPOPT", None, None, s) for s in SEEDS]
    for M in M_LIST:
        for s in SEEDS:
            jobs += [("Schwarz", M, b, s) for b in B_SCHWARZ]
            jobs += [("MultiShoot", M, None, s), ("ADMM", M, None, s)]
    w = min(int(os.environ.get("AOTD_WORKERS", "1")), len(jobs), os.cpu_count() or 4)
    print(
        f"=== NE39 swing baselines @ sigma=0.6: {len(jobs)} runs / {w} workers ===",
        flush=True,
    )
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            rows.append(d)
            if "error" in d:
                print(
                    f"[ERR ] {d['method']:<11} M={str(d['M']):>4} b={str(d['b']):>4} s={d['seed']}: {d['error']}",
                    flush=True,
                )
            else:
                print(
                    f"[done] {d['method']:<11} M={str(d['M']):>4} b={str(d['b']):>4} s={d['seed']} "
                    f"conv={str(d['converged'])[:1]} kkt={d['kkt']:.1e} flops={d['flops']:.2e} wall={d['wall']:.1f}s",
                    flush=True,
                )
    Path(DATA).write_text(
        json.dumps([r for r in rows if "error" not in r], default=float)
    )
    nconv = sum(r.get("converged", False) for r in rows if "error" not in r)
    print(
        f"\nDONE {len(rows)} runs ({nconv} converged) in {(time.time() - t0) / 60:.1f} min -> {os.path.relpath(DATA)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

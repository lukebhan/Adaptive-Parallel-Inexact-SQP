#!/usr/bin/env python3
"""Run the Swing comparison with the custom sketched solver."""

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
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "swing"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
REC = os.path.join(WORK, "records")
os.makedirs(REC, exist_ok=True)
MU, N_H = 10.0, 1000
M_LIST = [4, 10, 20, 50]
SEEDS = [1, 2, 3, 4, 5]
MAX_OUTER, AOTD_MAX_INNER = 25, 100
AOTD_MAX_PASSES = int(os.environ.get("AOTD_MAX_PASSES", "50"))
VB, VE = 4.0, 0.2  # swing headline hybrid rates (match run_comparison.py)
SCHEMA = 2  # True merit gradient, original-system local residuals, strict gates.


def make_seed_problem(N, seed, kick=0.25):
    prob0 = N.make_swing_problem(N_H, mode=1, pm_scale=0.0)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=kick) + 0.6 * kick * rng.standard_normal(
        2 * N.N_BUS
    )  # sigma=0.6
    return N.make_swing_problem(N_H, mode=1, pm_scale=0.0, x0=x0)


def record_path(M, seed):
    return os.path.join(REC, f"aotd_sketch_M{M}_seed{seed}.json")


def run_one(job):
    M, seed = job
    out = record_path(M, seed)
    if os.path.exists(out):
        d = json.loads(Path(out).read_text())
        if (
            "error" not in d
            and d.get("schema", 0) >= SCHEMA
            and d.get("max_inner_passes") == AOTD_MAX_PASSES
        ):
            return d
    sys.path.insert(0, SRC)
    try:
        import newton as N
        importlib.import_module("newton.AOTDsolver")._SKETCH_RNG = np.random.default_rng(0)

        globals()["N"] = N
        from newton.baselines.common import initial_guess, pack_traj

        prob = make_seed_problem(N, seed)
        Xi, Ui = initial_guess(prob)
        z0 = pack_traj(prob, Xi, Ui)
        lam0 = np.zeros(N.n_lam(prob))
        common = dict(
            M=M,
            mu=MU,
            gauss_newton=False,
            xi_H=1e-6,
            max_outer_iters=MAX_OUTER,
            tol_kkt=1e-6,
            use_preconditioner=True,
            max_overlap=110,
        )
        cfg = N.AlgorithmConfig(
            adaptive=True,
            b0=4,
            eps_i_0=1e-1,
            eps_i_floor=1e-9,
            acc_relax=10.0,
            adapt_mode="hybrid",
            hybrid_b_min1=True,
            warm_start=True,
            max_inner_passes=AOTD_MAX_PASSES,
            nu=2.0,
            b_step=4,
            varrho_b=VB,
            varrho_eps=VE,
            inner_solver="sketch",
            inner_rank_tol=1e-14,
            precond_type="ilu_schur",
            ilu_drop_tol=1e-2,
            max_inner_iters=AOTD_MAX_INNER,
            **common,
        )
        t = time.perf_counter()
        r = N.run_algorithm(prob, cfg, z0, lam0)
        wall = time.perf_counter() - t
        traj = [{k: v for k, v in tt.items()} for tt in r["trajectory"]]
        rec = dict(
            schema=SCHEMA,
            max_inner_passes=cfg.max_inner_passes,
            max_inner_iters=cfg.max_inner_iters,
            gauss_newton=cfg.gauss_newton,
            xi_H=cfg.xi_H,
            stop_reason=r['stop_reason'],
            sketch_rng_seed=0,
            method="AOTD-sketch",
            method_label="AOTD-sketch",
            N=N_H,
            M=M,
            b=None,
            seed=seed,
            converged=bool(r["converged"]),
            kkt=float(r["final_grad_L_norm"]),
            feas=float(r["final_feas"]),
            cost=float(r["final_cost"]),
            flops=float(r["total_flops"]),
            outer_iters=int(r["outer_iters"]),
            inner_iters=int(r["total_inner_iters"]),
            matvecs=int(r["total_matvecs"]),
            b_list=list(map(int, r["b_list"])),
            eps_i_list=[float(e) for e in r["eps_i_list"]],
            wall_contended=float(wall),
            trajectory=traj,
        )
        Path(out).write_text(json.dumps(rec, default=float))
        return rec
    except Exception as e:
        rec = dict(
            schema=SCHEMA,
            method="AOTD-sketch",
            method_label="AOTD-sketch",
            N=N_H,
            M=M,
            b=None,
            seed=seed,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        )
        Path(out).write_text(json.dumps(rec))
        return rec


def main():
    jobs = [(M, s) for M in M_LIST for s in SEEDS]
    w = min(int(os.environ.get("AOTD_WORKERS", "1")), len(jobs), os.cpu_count() or 4)
    print(f"=== swing AOTD-sketch: {len(jobs)} jobs / {w} workers ===", flush=True)
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            if "error" in d:
                print(f"[ERR ] M={d['M']} s={d['seed']}: {d['error']}", flush=True)
            else:
                print(
                    f"[done] AOTD-sketch M={d['M']:>2} s={d['seed']} conv={str(d['converged'])[:1]} "
                    f"kkt={d['kkt']:.1e} flops={d['flops']:.2e} out={d['outer_iters']} in={d['inner_iters']}",
                    flush=True,
                )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

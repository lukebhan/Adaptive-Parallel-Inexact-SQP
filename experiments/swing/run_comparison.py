#!/usr/bin/env python3
"""Run resumable Swing AOTD, FOTD-LU, and FOTD-QLP comparisons."""

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
COMBINED = os.path.join(WORK, "tight_rank_eps_sweep.json")

MU, N_H = 10.0, 1000
M_LIST = [4, 10, 20, 50]
B_LIST = [10, 40, 75]
SEEDS = [1, 2, 3, 4, 5]
MAX_OUTER = 25
FOTD_QLP_TOL = 1e-14
FOTD_QLP_MAX_INNER = 100
AOTD_MAX_INNER = 100
VB, VE = 4.0, 0.2  # swing headline hybrid rates
MAX_WORKERS = int(os.environ.get("AOTD_WORKERS", "1"))
SCHEMA = 1


def make_seed_problem(N, seed, kick=0.25):
    prob0 = N.make_swing_problem(N_H, mode=1, pm_scale=0.0)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=kick) + 0.6 * kick * rng.standard_normal(
        2 * N.N_BUS
    )  # sigma=0.6 IC (was 0.15)
    return N.make_swing_problem(N_H, mode=1, pm_scale=0.0, x0=x0)


def method_label(method, b):
    return method if b is None else f"{method}-b{b}"


def record_path(method, b, M, seed):
    return os.path.join(
        REC, f"{method_label(method, b).lower().replace('-', '_')}_M{M}_seed{seed}.json"
    )


def run_one(job):
    method, b, M, seed = job
    out = record_path(method, b, M, seed)
    if os.path.exists(out):
        d = json.loads(Path(out).read_text())
        if "error" not in d and d.get("schema", 0) >= SCHEMA:
            return d
    sys.path.insert(0, SRC)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import newton as N

        globals()["N"] = N
        from newton.baselines.common import initial_guess, pack_traj

        prob = make_seed_problem(N, seed)
        Xi, Ui = initial_guess(prob)
        z0 = pack_traj(prob, Xi, Ui)
        lam0 = np.zeros(N.n_lam(prob))
        common = dict(
            M=M,
            mu=MU,
            gauss_newton=True,
            max_outer_iters=MAX_OUTER,
            tol_kkt=1e-6,
            use_preconditioner=(method != "FOTD-LU"),
            max_overlap=110,
        )
        if method == "FOTD-QLP":
            cfg = N.AlgorithmConfig(
                adaptive=False,
                b0=b,
                eps_i_0=FOTD_QLP_TOL,
                inner_solver="gmres_qlp",
                inner_rank_tol=FOTD_QLP_TOL,
                precond_type="ilu_schur",
                ilu_drop_tol=1e-2,
                max_inner_iters=FOTD_QLP_MAX_INNER,
                **common,
            )
        elif method == "FOTD-LU":
            cfg = N.AlgorithmConfig(
                adaptive=False,
                b0=b,
                eps_i_0=1e-14,
                inner_solver="direct",
                max_inner_iters=400,
                **common,
            )
        else:  # AOTD-rebuild
            cfg = N.AlgorithmConfig(
                adaptive=True,
                b0=4,
                eps_i_0=1e-1,
                eps_i_floor=1e-9,
                acc_relax=10.0,
                adapt_mode="hybrid",
                hybrid_b_min1=True,
                warm_start=True,
                max_inner_passes=6,
                nu=2.0,
                b_step=4,
                varrho_b=VB,
                varrho_eps=VE,
                inner_solver="gmres_qlp",
                inner_rank_tol=1e-14,
                precond_type="ilu_schur",
                ilu_drop_tol=1e-2,
                max_inner_iters=AOTD_MAX_INNER,
                **common,
            )
        t = time.perf_counter()
        r = N.run_algorithm(prob, cfg, z0, lam0)
        wall = time.perf_counter() - t
        traj = [
            {k: v for k, v in tt.items()} for tt in r["trajectory"]
        ]  # keep passes for AOTD adaptation fig
        rec = dict(
            schema=SCHEMA,
            method=method,
            method_label=method_label(method, b),
            N=N_H,
            M=M,
            b=b,
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
            method=method,
            method_label=method_label(method, b),
            N=N_H,
            M=M,
            b=b,
            seed=seed,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        )
        Path(out).write_text(json.dumps(rec))
        return rec


def main():
    jobs = []
    for M in M_LIST:
        for seed in SEEDS:
            for b in B_LIST:
                jobs.append(("FOTD-QLP", b, M, seed))
                jobs.append(("FOTD-LU", b, M, seed))
            jobs.append(("AOTD-rebuild", None, M, seed))
    w = min(MAX_WORKERS, len(jobs), os.cpu_count() or 4)
    print(
        f"=== NE39 swing tight-rank/eps: {len(jobs)} jobs / {w} workers "
        f"(FOTD-QLP 1e-14 cap{FOTD_QLP_MAX_INNER}, FOTD-LU exact, AOTD hybrid) ===",
        flush=True,
    )
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            rows.append(d)
            if "error" in d:
                print(
                    f"[ERR ] {d['method_label']:<14} M={d['M']:>2} s={d['seed']}: {d['error']}",
                    flush=True,
                )
            else:
                print(
                    f"[done] {d['method_label']:<14} M={d['M']:>2} s={d['seed']} conv={str(d['converged'])[:1]} "
                    f"kkt={d['kkt']:.1e} flops={d['flops']:.2e} out={d['outer_iters']:>2} "
                    f"in={d['inner_iters']:>5}",
                    flush=True,
                )
    Path(COMBINED).write_text(
        json.dumps([r for r in rows if "error" not in r], default=float)
    )
    print(
        f"\nDONE {len(rows)} runs in {(time.time() - t0) / 60:.1f} min -> {os.path.relpath(COMBINED)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

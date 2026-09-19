#!/usr/bin/env python3
"""Run resumable Burgers AOTD, FOTD-LU, and FOTD-QLP comparisons."""

import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "burgers"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
RECORDS = os.path.join(WORK, "records")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(WORK, ".matplotlib"))

DT, ALPHA, MU, NU, NX = 0.004, 1e-3, 10.0, 0.02, 24
N_LIST = [10000, 50000]
M_LIST = [10, 50, 250]
SEEDS = [1, 2, 3, 4, 5]
FOTD_B_LIST = [10, 40]
METHODS = ["FOTD-QLP", "FOTD-LU", "AOTD-rebuild"]
FOTD_EPS = 1e-14
RANK_TOL = 1e-14  # AOTD rank tolerance.
# Use the same tight FOTD-QLP tolerance at both horizons.
FOTD_QLP_TOL_BY_N = {10000: 1e-14, 50000: 1e-14}
FOTD_QLP_MAX_INNER = 100
FOTD_MAX_INNER_ITERS = 400  # FOTD-LU is direct -> cap irrelevant; kept for schema
AOTD_MAX_INNER_ITERS = 100
SCHEMA_VERSION = 4  # Full Hessian and strict, unrelaxed acceptance.
WORKERS_BY_N = {10000: 15, 50000: 6}
CHECK_QLP_RESIDUALS = False


def method_label(method, b):
    if b is None:
        return method
    return f"{method}-b{b}"


def record_path(horizon, method, b, M, seed):
    safe = method_label(method, b).lower().replace("-", "_")
    return os.path.join(RECORDS, f"N{horizon}_{safe}_M{M}_seed{seed}.json")


def make_jobs(horizon, M):
    jobs = []
    for seed in SEEDS:
        for b in FOTD_B_LIST:
            if b == 10 and M == 250:  # Excluded from the paper grid.
                continue
            jobs.append((horizon, "FOTD-QLP", b, M, seed))
        for b in FOTD_B_LIST:
            jobs.append((horizon, "FOTD-LU", b, M, seed))
        jobs.append((horizon, "AOTD-rebuild", None, M, seed))
    return jobs


def make_problem(newton_mod, horizon, seed):
    from newton import burgers_setting as bs

    bs.set_dims(NX, nu=NX)
    params = newton_mod.BurgersParams(nu=NU, amp=1.0, alpha=ALPHA, target="zero")
    grid = (
        bs.burgers_grid() if hasattr(bs, "burgers_grid") else newton_mod.burgers_grid()
    )
    rng = np.random.default_rng(seed)
    amp = params.amp * (1.0 + 0.3 * rng.standard_normal())
    perturbation = 0.2 * (
        rng.uniform(-1, 1) * np.sin(2 * np.pi * grid)
        + rng.uniform(-1, 1) * np.sin(3 * np.pi * grid)
    )
    x0 = amp * np.sin(np.pi * grid) + perturbation
    Q = DT * np.eye(NX)
    R = DT * ALPHA * np.eye(NX)
    QN = np.eye(NX)
    target = np.tile(newton_mod.burgers_target(params), (horizon + 1, 1))
    return newton_mod.Problem(horizon, DT, Q, R, QN, x0=x0, x_des=target, params=params)


def make_config(newton_mod, method, b, M, horizon):
    common = dict(
        M=M,
        mu=MU,
        max_outer_iters=12,
        tol_kkt=1e-6,
        max_overlap=6000,
        cache_precond=False,
        freeze_precond=False,
    )
    if method == "FOTD-QLP":
        return newton_mod.AlgorithmConfig(
            adaptive=False,
            b0=int(b),
            eps_i_0=FOTD_QLP_TOL_BY_N[horizon],
            inner_solver="gmres_qlp",
            inner_rank_tol=FOTD_QLP_TOL_BY_N[horizon],
            use_preconditioner=True,
            precond_type="ilu_schur",
            ilu_drop_tol=1e-2,
            max_inner_iters=FOTD_QLP_MAX_INNER,
            **common,
        )
    if method == "FOTD-LU":
        return newton_mod.AlgorithmConfig(
            adaptive=False,
            b0=int(b),
            eps_i_0=FOTD_EPS,
            inner_solver="direct",
            inner_rank_tol=RANK_TOL,
            use_preconditioner=False,
            max_inner_iters=FOTD_MAX_INNER_ITERS,
            **common,
        )
    return newton_mod.AlgorithmConfig(
        adaptive=True,
        b0=4,
        eps_i_0=1e-1,
        eps_i_floor=1e-9,
        warm_start=True,
        max_inner_passes=50,
        nu=2.0,
        varrho_b=10.0,
        varrho_eps=0.1,
        inner_solver="gmres_qlp",
        inner_rank_tol=RANK_TOL,
        use_preconditioner=True,
        precond_type="ilu_schur",
        ilu_drop_tol=1e-2,
        max_inner_iters=AOTD_MAX_INNER_ITERS,
        **common,
    )


def residual_summary(values):
    if not values:
        return dict(minimum=math.nan, median=math.nan, p95=math.nan, maximum=math.nan)
    data = np.asarray(values, dtype=float)
    return dict(
        minimum=float(np.min(data)),
        median=float(np.median(data)),
        p95=float(np.quantile(data, 0.95)),
        maximum=float(np.max(data)),
    )


def current_record(row):
    return "error" not in row and row.get("schema_version", 0) >= SCHEMA_VERSION


def run_one(job):
    horizon, method, b, M, seed = job
    output = record_path(horizon, method, b, M, seed)
    if os.path.exists(output):
        with open(output) as stream:
            existing = json.load(stream)
        if current_record(existing):
            return existing

    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    record = dict(
        schema_version=SCHEMA_VERSION,
        N=horizon,
        M=M,
        seed=seed,
        method=method,
        method_label=method_label(method, b),
        b=b,
    )
    try:
        import newton as newton_mod
        import newton.AOTD as algorithm
        from newton.baselines.common import initial_guess, pack_traj

        problem = make_problem(newton_mod, horizon, seed)
        Xi, Ui = initial_guess(problem)
        z0 = pack_traj(problem, Xi, Ui)
        lam0 = np.zeros(newton_mod.n_lam(problem))
        cfg = make_config(newton_mod, method, b, M, horizon)

        achieved_residuals = []
        reported_cap_hits = 0
        early_misses = 0
        diagnostic_wall = 0.0
        original_solve = algorithm.solve

        if method == "FOTD-QLP" and CHECK_QLP_RESIDUALS:

            def monitored_solve(solver, Gamma, rhs, tol, max_iters, M=None, **kwargs):
                nonlocal diagnostic_wall, reported_cap_hits, early_misses
                solution, info = original_solve(
                    solver, Gamma, rhs, tol=tol, max_iters=max_iters, M=M, **kwargs
                )
                diagnostic_start = time.perf_counter()
                apply_preconditioner = M.__call__ if hasattr(M, "__call__") else M
                transformed_rhs = apply_preconditioner(np.asarray(rhs, dtype=float))
                transformed_residual = apply_preconditioner(Gamma @ solution - rhs)
                denominator = np.linalg.norm(transformed_rhs)
                relative_residual = (
                    float(np.linalg.norm(transformed_residual) / denominator)
                    if denominator
                    else 0.0
                )
                achieved_residuals.append(relative_residual)
                reported_cap_hits += int(bool(info.get("cap_hit", False)))
                early_misses += int(
                    relative_residual > tol and info["iters"] < max_iters
                )
                diagnostic_wall += time.perf_counter() - diagnostic_start
                return solution, info

            algorithm.solve = monitored_solve

        start = time.perf_counter()
        try:
            result = newton_mod.run_algorithm(problem, cfg, z0, lam0)
        finally:
            algorithm.solve = original_solve
        wall = time.perf_counter() - start

        pass_count = sum(len(step.get("passes", [])) for step in result["trajectory"])
        if method == "AOTD-rebuild":
            preconditioner_builds = pass_count * M
        elif method == "FOTD-QLP":
            preconditioner_builds = max(result["outer_iters"] - 1, 0) * M
        else:
            preconditioner_builds = 0

        record.update(
            window=horizon // M,
            inner_solver=cfg.inner_solver,
            eps_i_0=float(cfg.eps_i_0),
            eps_i_floor=float(cfg.eps_i_floor),
            rank_tol=(
                float(cfg.inner_rank_tol) if cfg.inner_solver == "gmres_qlp" else None
            ),
            max_inner_iters=int(cfg.max_inner_iters),
            cache_precond=bool(cfg.cache_precond),
            freeze_precond=bool(cfg.freeze_precond),
            converged=bool(result["converged"]),
            stop_reason=result["stop_reason"],
            kkt=float(result["final_grad_L_norm"]),
            feas=float(result["final_feas"]),
            stationarity=float(result["final_stationarity"]),
            cost=float(result["final_cost"]),
            flops=float(result["total_flops"]),
            outer_iters=int(result["outer_iters"]),
            inner_iters=int(result["total_inner_iters"]),
            matvecs=int(result["total_matvecs"]),
            inner_passes=int(pass_count),
            preconditioner_builds=int(preconditioner_builds),
            b_list=list(map(int, result["b_list"])),
            b_min=int(min(result["b_list"])),
            b_max=int(max(result["b_list"])),
            b_mean=float(np.mean(result["b_list"])),
            eps_i_list=[float(value) for value in result["eps_i_list"]],
            eps_i_min=float(min(result["eps_i_list"])),
            eps_i_max=float(max(result["eps_i_list"])),
            eps_i_mean=float(np.mean(result["eps_i_list"])),
            eta1=float(result["eta1"]),
            eta2=float(result["eta2"]),
            eps_g=float(result["eps_g"]),
            wall_contended=float(wall),
            diagnostic_wall=float(diagnostic_wall),
            wall_excluding_diagnostics_contended=float(wall - diagnostic_wall),
            parallel_wall_contended=float(
                result["t_assemble"]
                + result["t_subsolve_parallel"]
                + result["t_compose"]
                + result["t_linesearch"]
            ),
            timing=dict(
                assemble=float(result["t_assemble"]),
                subsolve_serial=float(result["t_subsolve_serial"]),
                subsolve_parallel=float(result["t_subsolve_parallel"]),
                compose=float(result["t_compose"]),
                linesearch=float(result["t_linesearch"]),
            ),
            round_sub_flops=result["round_sub_flops"],
            round_sub_times=result["round_sub_times"],
            trajectory=result["trajectory"],
        )
        if method == "FOTD-QLP":
            misses = sum(value > FOTD_EPS for value in achieved_residuals)
            record["local_solve_diagnostics"] = dict(
                local_solves=len(achieved_residuals),
                solves_meeting_tol=len(achieved_residuals) - misses,
                solves_missing_tol=misses,
                early_rank_or_breakdown_misses=early_misses,
                reported_cap_hits=reported_cap_hits,
                achieved_preconditioned_relative_residual=residual_summary(
                    achieved_residuals
                ),
                residual_check_enabled=CHECK_QLP_RESIDUALS,
            )
    except Exception as exc:
        record.update(
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )

    with open(output, "w") as stream:
        json.dump(record, stream, default=float)
    return record


def main():
    os.makedirs(RECORDS, exist_ok=True)
    total_start = time.time()
    for horizon in N_LIST:
        for M in M_LIST:
            jobs = make_jobs(horizon, M)
            workers = min(
                int(os.environ.get("AOTD_WORKERS", "1")), len(jobs), os.cpu_count() or 1
            )
            print(
                f"\n=== N={horizon}, M={M}: {len(jobs)} jobs / {workers} workers ===",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, job): job for job in jobs}
                for future in as_completed(futures):
                    row = future.result()
                    if "error" in row:
                        print(
                            f"[ERR ] {row['method_label']:<16} N={row['N']:>5} M={row['M']:>3} "
                            f"seed={row['seed']}: {row['error']}",
                            flush=True,
                        )
                        continue
                    diagnostics = row.get("local_solve_diagnostics")
                    met = ""
                    if (
                        diagnostics
                        and diagnostics.get("residual_check_enabled", True)
                        and diagnostics.get("local_solves", 0)
                    ):
                        met = f" met={diagnostics['solves_meeting_tol']}/{diagnostics['local_solves']}"
                    print(
                        f"[done] {row['method_label']:<16} N={row['N']:>5} M={row['M']:>3} "
                        f"seed={row['seed']} conv={str(row['converged'])[:1]} "
                        f"kkt={row['kkt']:.1e} flops={row['flops']:.3e} "
                        f"out={row['outer_iters']:>2} in={row['inner_iters']:>7}{met}",
                        flush=True,
                    )

    print(f"Finished sweep in {(time.time() - total_start) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the resumable Burgers N=50000 sweep with the custom sketched solver."""

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
import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "burgers"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
REC = os.path.join(WORK, "records")
os.makedirs(REC, exist_ok=True)
N_H = 50000
M_LIST = [10, 50, 250]
SEEDS = [1, 2, 3, 4, 5]
SCHEMA = 3
MAX_WORKERS = int(os.environ.get("AOTD_WORKERS", "1"))


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "rt", os.path.join(HERE, "run_comparison.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rec_path(M, seed):
    return os.path.join(REC, f"N{N_H}_aotd_sketch_M{M}_seed{seed}.json")


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
        from newton.baselines.common import initial_guess, pack_traj

        rt = _sweep_module()
        prob = rt.make_problem(N, N_H, seed)
        Xi, Ui = initial_guess(prob)
        z0 = pack_traj(prob, Xi, Ui)
        lam0 = np.zeros(N.n_lam(prob))
        cfg = dataclasses.replace(
            rt.make_config(N, "AOTD-rebuild", None, M, N_H), inner_solver="sketch"
        )
        t = time.perf_counter()
        r = N.run_algorithm(prob, cfg, z0, lam0)
        wall = time.perf_counter() - t
        pass_count = sum(len(s.get("passes", [])) for s in r["trajectory"])
        rec = dict(
            schema_version=SCHEMA,
            N=N_H,
            M=M,
            seed=seed,
            b=None,
            method="AOTD-sketch",
            method_label="AOTD-sketch",
            converged=bool(r["converged"]),
            kkt=float(r["final_grad_L_norm"]),
            feas=float(r["final_feas"]),
            cost=float(r["final_cost"]),
            flops=float(r["total_flops"]),
            outer_iters=int(r["outer_iters"]),
            inner_iters=int(r["total_inner_iters"]),
            inner_passes=int(pass_count),
            matvecs=int(r["total_matvecs"]),
            b_list=list(map(int, r["b_list"])),
            eps_i_list=[float(e) for e in r["eps_i_list"]],
            wall_contended=float(wall),
            trajectory=r["trajectory"],
        )
        Path(out).write_text(json.dumps(rec, default=float))
        return rec
    except Exception as ex:
        rec = dict(
            schema_version=SCHEMA,
            N=N_H,
            M=M,
            seed=seed,
            b=None,
            method_label="AOTD-sketch",
            error=f"{type(ex).__name__}: {ex}",
            tb=traceback.format_exc(),
        )
        Path(out).write_text(json.dumps(rec))
        return rec


def main():
    jobs = [(M, s) for M in M_LIST for s in SEEDS]
    w = min(MAX_WORKERS, len(jobs), os.cpu_count() or 4)
    print(
        f"=== N=50k Burgers AOTD-sketch: {len(jobs)} jobs / {w} workers ===", flush=True
    )
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            if "error" in d:
                print(f"[ERR ] M={d['M']} s={d['seed']}: {d['error']}", flush=True)
            else:
                print(
                    f"[done] AOTD-sketch M={d['M']:>3} s={d['seed']} conv={str(d['converged'])[:1]} "
                    f"kkt={d['kkt']:.1e} flops={d['flops']:.2e} out={d['outer_iters']} "
                    f"in={d['inner_iters']} wall={d['wall_contended']:.0f}s",
                    flush=True,
                )
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

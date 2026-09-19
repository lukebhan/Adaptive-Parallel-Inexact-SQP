#!/usr/bin/env python3
"""Sweep 16 hybrid adaptation-rate pairs at M=20 over five Swing seeds."""

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
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "rates"))
)
os.makedirs(WORK, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "src")
REC = os.path.join(WORK, "records")
os.makedirs(REC, exist_ok=True)
COMBINED = os.path.join(WORK, "rate_ablation.json")

# Reference configuration.
MU, N_H = 10.0, 1000
M_LIST = [20]
SEEDS = [1, 2, 3, 4, 5]
MAX_OUTER = 25
TOL_KKT = 1e-6
SIGMA = 0.6  # Relative initial-condition noise.
KICK = 0.25
AOTD_MAX_INNER = 100
MAX_OVERLAP = 110
INNER_RANK_TOL = 1e-14
# Rate grid includes the reference pair (4, 0.2).
RB = [1.0, 2.0, 4.0, 10.0]
RE = [0.1, 0.2, 0.3, 1.0]
HEADLINE = (4.0, 0.2)
MAX_WORKERS = int(os.environ.get("AOTD_WORKERS", "1"))
SCHEMA = 1


def make_seed_problem(N, seed):
    prob0 = N.make_swing_problem(N_H, mode=1, pm_scale=0.0)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=KICK) + SIGMA * KICK * rng.standard_normal(
        2 * N.N_BUS
    )
    return N.make_swing_problem(N_H, mode=1, pm_scale=0.0, x0=x0)


def tag(rb, re):
    return f"vb{rb:g}".replace(".", "p") + "_" + f"ve{re:g}".replace(".", "p")


def record_path(rb, re, M, seed):
    return os.path.join(REC, f"aotd_{tag(rb, re)}_M{M}_seed{seed}.json")


def make_cfg(N, M, rb, re):
    """Build the Swing AOTD configuration for the requested adaptation rates."""
    return N.AlgorithmConfig(
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
        varrho_b=rb,
        varrho_eps=re,
        inner_solver="gmres_qlp",
        inner_rank_tol=INNER_RANK_TOL,
        precond_type="ilu_schur",
        ilu_drop_tol=1e-2,
        max_inner_iters=AOTD_MAX_INNER,
        M=M,
        mu=MU,
        gauss_newton=True,
        max_outer_iters=MAX_OUTER,
        tol_kkt=TOL_KKT,
        use_preconditioner=True,
        max_overlap=MAX_OVERLAP,
    )


def run_one(job):
    rb, re, M, seed = job
    out = record_path(rb, re, M, seed)
    if os.path.exists(out):
        d = json.loads(Path(out).read_text())
        if "error" not in d and d.get("schema", 0) >= SCHEMA:
            return d
    sys.path.insert(0, SRC)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import newton as N
        from newton.baselines.common import initial_guess, pack_traj

        prob = make_seed_problem(N, seed)
        Xi, Ui = initial_guess(prob)
        z0 = pack_traj(prob, Xi, Ui)
        lam0 = np.zeros(N.n_lam(prob))
        cfg = make_cfg(N, M, rb, re)
        t = time.perf_counter()
        r = N.run_algorithm(prob, cfg, z0, lam0)
        wall = time.perf_counter() - t
        traj = [{k: v for k, v in tt.items()} for tt in r["trajectory"]]
        b_all = np.array([p["b_used"] for tt in traj for p in tt["passes"]], float)
        e_all = np.array([p["eps_used"] for tt in traj for p in tt["passes"]], float)
        n_updates = sum(int(tt.get("n_acc_fail", 0)) for tt in traj)
        rec = dict(
            schema=SCHEMA,
            method="AOTD-rebuild",
            varrho_b=float(rb),
            varrho_eps=float(re),
            N=N_H,
            M=M,
            seed=seed,
            sigma=SIGMA,
            converged=bool(r["converged"]),
            kkt=float(r["final_grad_L_norm"]),
            feas=float(r["final_feas"]),
            cost=float(r["final_cost"]),
            flops=float(r["total_flops"]),
            outer_iters=int(r["outer_iters"]),
            inner_iters=int(r["total_inner_iters"]),
            matvecs=int(r["total_matvecs"]),
            n_passes=int(b_all.shape[0]),
            n_updates=int(n_updates),
            b_list=list(map(int, r["b_list"])),
            eps_i_list=[float(e) for e in r["eps_i_list"]],
            b_max=float(b_all.max()) if b_all.size else float("nan"),
            eps_min=float(e_all.min()) if e_all.size else float("nan"),
            wall_contended=float(wall),
            trajectory=traj,
        )
        Path(out).write_text(json.dumps(rec, default=float))
        return rec
    except Exception as e:
        rec = dict(
            schema=SCHEMA,
            method="AOTD-rebuild",
            varrho_b=float(rb),
            varrho_eps=float(re),
            N=N_H,
            M=M,
            seed=seed,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        )
        Path(out).write_text(json.dumps(rec))
        return rec


def print_constants():
    print(
        "=== exp12 constants (exp10 algorithm + exp10 settings; only (varrho_b,varrho_eps) swept) ==="
    )
    print(
        f"  problem     : NE39 swing, N={N_H}, mode=1, pm_scale=0, mu={MU}, Gauss-Newton"
    )
    print(
        f"  IC ensemble : x0 = swing_x0(kick={KICK}) + {SIGMA}*{KICK}*xi, xi~N(0,I), seeds {SEEDS}"
    )
    print(f"  outer       : max_outer={MAX_OUTER}, tol_kkt={TOL_KKT:g}, M in {M_LIST}")
    print("  adaptation  : adapt_mode=hybrid, hybrid_b_min1=True, b0=4, b_step=4,")
    print(
        "                eps_i_0=1e-1, eps_i_floor=1e-9, acc_relax=10, nu=2, max_inner_passes=6,"
    )
    print(f"                max_overlap={MAX_OVERLAP}, warm_start=True")
    print(
        f"  inner solve : gmres_qlp, inner_rank_tol={INNER_RANK_TOL:g}, max_inner_iters={AOTD_MAX_INNER},"
    )
    print("                precond=ilu_schur (rebuilt), ilu_drop_tol=1e-2")
    print(
        "  law         : eps_i <- max(eps_i*exp(-ve*(1/sqrt(M)+||r_i||/||r||)), 1e-9)"
    )
    print(
        f"                b_i   <- min(b_i + max(1, ceil(vb*||r_i||/||r||)), {MAX_OVERLAP})"
    )
    print(
        f"  grid        : varrho_b in {[f'{x:g}' for x in RB]} x varrho_eps in {[f'{x:g}' for x in RE]}"
        f"  (headline vb={HEADLINE[0]:g}, ve={HEADLINE[1]:g})",
        flush=True,
    )


def print_detail(rows):
    """Summarize seed-1 trajectories for the reference and extreme rate pairs."""
    want = [(HEADLINE[0], HEADLINE[1], M, 1) for M in M_LIST] + [
        (rb, re, 20, 1) for rb in (RB[0], RB[-1]) for re in (RE[0], RE[-1])
    ]
    idx = {
        (r["varrho_b"], r["varrho_eps"], r["M"], r["seed"]): r
        for r in rows
        if "error" not in r
    }
    for key in dict.fromkeys(want):
        r = idx.get(key)
        if r is None:
            continue
        rb, re, M, seed = key
        print(
            f"\n--- per-outer detail: varrho_b={rb:g} varrho_eps={re:g} M={M} seed={seed} "
            f"(conv={r['converged']}, KKT={r['kkt']:.3e}, FLOPs={r['flops']:.3e}) ---"
        )
        print(
            "  tau  |grad L|   theta_k   eps^tau    eps_i_min  eps_i_max  acc_rhs    "
            "b_min b_max b_mean  passes  in  alpha  acc"
        )
        for t in r["trajectory"]:
            arh = [p["acc_rhs"] for p in t["passes"]]
            print(
                f"  {t['tau']:>3} {t['grad_L_norm']:.3e} {t['theta_k']:.3e} {t['eps_g']:.3e} "
                f"{t['eps_i_min']:.3e} {t['eps_i_max']:.3e} {min(arh):.3e}  "
                f"{t['b_min']:>5.0f} {t['b_max']:>5.0f} {t['b_mean']:>6.1f} "
                f"{t['n_inner_passes']:>6} {t['inner_iters']:>4} {t['alpha']:.2f}  {str(t['accepted'])[:1]}"
            )
        print(
            f"  final b_i : min={min(r['b_list'])} max={max(r['b_list'])} "
            f"mean={np.mean(r['b_list']):.1f} | trajectory b_max over passes = {r['b_max']:.0f}"
        )
        print(
            f"  final eps_i: min={min(r['eps_i_list']):.3e} max={max(r['eps_i_list']):.3e} "
            f"| min over passes = {r['eps_min']:.3e}"
        )
        print(
            f"  b_i (final, per subproblem): {list(map(int, r['b_list']))}", flush=True
        )


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    print(
        "\n=== grid summary: mean over 5 seeds of converged runs (FLOPs, outer/inner, b_max, eps_min) ==="
    )
    for M in M_LIST:
        print(f"\n  M = {M}")
        print(
            "    vb     ve     conv   FLOPs(mean+-std)      out    in     b_max  eps_min   KKT(mean)"
        )
        for rb in RB:
            for re in RE:
                v = [
                    r
                    for r in ok
                    if r["varrho_b"] == rb and r["varrho_eps"] == re and r["M"] == M
                ]
                if not v:
                    continue
                c = [r for r in v if r["converged"]]
                f = (
                    np.array([r["flops"] for r in c], float)
                    if c
                    else np.array([np.nan])
                )
                print(
                    f"    {rb:<6g} {re:<6g} {len(c)}/{len(v)}  "
                    f"{np.mean(f):.3e}+-{np.std(f):.1e}  "
                    f"{np.mean([r['outer_iters'] for r in (c or v)]):>5.1f} "
                    f"{np.mean([r['inner_iters'] for r in (c or v)]):>6.1f} "
                    f"{np.mean([r['b_max'] for r in (c or v)]):>6.1f} "
                    f"{np.mean([r['eps_min'] for r in (c or v)]):.2e}  "
                    f"{np.mean([r['kkt'] for r in v]):.2e}"
                )
        allc = {
            (rb, re): all(
                r["converged"]
                for r in ok
                if r["varrho_b"] == rb and r["varrho_eps"] == re and r["M"] == M
            )
            for rb in RB
            for re in RE
        }
        cheap = sorted(
            (
                (
                    np.mean(
                        [
                            r["flops"]
                            for r in ok
                            if r["varrho_b"] == rb
                            and r["varrho_eps"] == re
                            and r["M"] == M
                        ]
                    ),
                    rb,
                    re,
                )
                for (rb, re), a in allc.items()
                if a
            )
        )
        if cheap:
            fl, rb, re = cheap[0]
            print(
                f"    -> cheapest 5/5-converged pair at M={M}: vb={rb:g} ve={re:g} "
                f"(mean FLOPs {fl:.3e}); headline vb={HEADLINE[0]:g} ve={HEADLINE[1]:g}",
                flush=True,
            )


def main():
    print_constants()
    jobs = [
        (rb, re, M, seed) for M in M_LIST for seed in SEEDS for rb in RB for re in RE
    ]
    w = min(MAX_WORKERS, len(jobs), os.cpu_count() or 4)
    print(f"\n=== exp12 rate ablation: {len(jobs)} jobs / {w} workers ===", flush=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=w) as pool:
        for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
            d = fut.result()
            rows.append(d)
            if "error" in d:
                print(
                    f"[ERR ] vb={d['varrho_b']:<4g} ve={d['varrho_eps']:<4g} M={d['M']:>2} "
                    f"s={d['seed']}: {d['error']}",
                    flush=True,
                )
            else:
                print(
                    f"[done] vb={d['varrho_b']:<4g} ve={d['varrho_eps']:<4g} M={d['M']:>2} s={d['seed']} "
                    f"conv={str(d['converged'])[:1]} kkt={d['kkt']:.1e} flops={d['flops']:.2e} "
                    f"out={d['outer_iters']:>2} in={d['inner_iters']:>5} pass={d['n_passes']:>3} "
                    f"bmax={d['b_max']:>3.0f} emin={d['eps_min']:.1e}",
                    flush=True,
                )
    Path(COMBINED).write_text(
        json.dumps([r for r in rows if "error" not in r], default=float)
    )
    print(
        f"\nDONE {len(rows)} runs in {(time.time() - t0) / 60:.1f} min -> {os.path.relpath(COMBINED)}",
        flush=True,
    )
    summarize(rows)
    print_detail(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sweep initial merit penalties and update factors on the Swing problem."""

import os
import sys
import json
import time
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import newton as N
from newton.baselines.common import initial_guess, pack_traj

OUT = os.environ.get(
    "AOTD_WORK",
    os.path.abspath(os.path.join(HERE, "..", "..", "results", "globalization")),
)
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(OUT, "ablation_eta_data.json")

MU = 10.0
N_H = 1000
SEED = 1  # single IC-noise seed (same convention as exp2)

# Sweep initial penalties and update factors for one seed.
ETA1 = [1e-3, 1e-1, 1e1]
ETA2 = [1e1, 1e2, 1e3]
NU = [1.1, 1.3, 1.5, 2.0]
CELLS = [(e1, e2, nu) for e1 in ETA1 for e2 in ETA2 for nu in NU]

records = []


def save():
    with open(DATA, "w") as fh:
        fh.write("[\n")
        for i, rec in enumerate(records):
            fh.write(json.dumps(rec, default=float))
            fh.write(",\n" if i < len(records) - 1 else "\n")
        fh.write("]\n")


def make_seed_problem(seed, kick=0.25):  # identical to exp2 → same x0
    prob0 = N.make_swing_problem(N_H, mode=1, pm_scale=0.0)
    rng = np.random.default_rng(seed)
    x0 = N.swing_x0(prob0.params, kick=kick) + 0.15 * kick * rng.standard_normal(
        2 * N.N_BUS
    )
    return N.make_swing_problem(N_H, mode=1, pm_scale=0.0, x0=x0)


def common_cfg(M=20):  # exp2/exp3 solver stack + 2 tweaks:
    # Small psi relaxes the accuracy gate; 40 passes allow repeated descent updates.
    return dict(
        M=M,
        mu=MU,
        gauss_newton=True,
        max_outer_iters=25,
        tol_kkt=1e-6,
        tol_step=1e-10,
        inner_solver="gmres_qlp",
        use_preconditioner=True,
        precond_type="ilu_schur",
        ilu_drop_tol=1e-2,
        max_inner_iters=100,
        max_overlap=110,
        b0=4,
        eps_i_0=1e-1,
        eps_i_floor=1e-9,
        b_step=4,
        acc_relax=1.0,
        psi=1e-5,
        max_inner_passes=40,
    )


def extract_firings(result):
    """Collect post-update penalties and pre-update descent margins from failed passes."""
    firings = []
    for t in result["trajectory"]:
        gnorm = float(t["grad_L_norm"])
        for p in t.get("passes", []):
            if p.get("outcome") != "desc_fail":
                continue
            gdir = float(p.get("gdir", np.nan))
            d_rhs = float(p.get("descent_rhs", np.nan))
            firings.append(
                dict(
                    fire=len(firings) + 1,
                    tau=int(t["tau"]),
                    pass_idx=int(p["pass_idx"]),
                    grad_L_norm=gnorm,
                    gdir=gdir,
                    descent_rhs=d_rhs,
                    desc_margin=gdir - d_rhs,  # >0 ⇒ failed the bar
                    eta1=float(p["eta1"]),
                    eta2=float(p["eta2"]),
                    eps_g_after=float(p.get("eps_g_after", np.nan)),
                )
            )
    return firings


def per_outer(result):
    return [
        dict(
            tau=int(t["tau"]),
            grad_L_norm=float(t["grad_L_norm"]),
            eta1=float(t["eta1"]),
            eta2=float(t["eta2"]),
            eps_g=float(t["eps_g"]),
            n_desc_fail=int(t["n_desc_fail"]),
            n_acc_fail=int(t["n_acc_fail"]),
            alpha=float(t["alpha"]),
            b_max=int(t["b_max"]),
        )
        for t in result["trajectory"]
    ]


def main():
    prob = make_seed_problem(SEED)
    Xi, Ui = initial_guess(prob)
    z0 = pack_traj(prob, Xi, Ui)
    lam0 = np.zeros(N.n_lam(prob))
    for e1, e2, nu in CELLS:
        cfg = N.AlgorithmConfig(
            adaptive=True, eta1_0=e1, eta2_0=e2, nu=nu, **common_cfg()
        )
        print(f"[ablation] η1_0={e1:.0e} η2_0={e2:.0e} ν={nu} ...", flush=True)
        t0 = time.time()
        try:
            r = N.run_algorithm(prob, cfg, z0, lam0)
        except Exception:
            traceback.print_exc()
            continue
        firings = extract_firings(r)
        rec = dict(
            eta1_0=e1,
            eta2_0=e2,
            nu=nu,
            seed=SEED,
            M=cfg.M,
            mu=MU,
            converged=bool(r["converged"]),
            stop_reason=r["stop_reason"],
            outer_iters=int(r["outer_iters"]),
            final_kkt=float(r["final_grad_L_norm"]),
            total_flops=float(r["total_flops"]),
            total_matvecs=int(r["total_matvecs"]),
            final_eta1=float(r["eta1"]),
            final_eta2=float(r["eta2"]),
            n_firings=len(firings),
            wall=time.time() - t0,
            firings=firings,
            per_outer=per_outer(r),
        )
        records.append(rec)
        save()
        print(
            f"  -> conv={rec['converged']} outers={rec['outer_iters']} "
            f"firings={rec['n_firings']} final_kkt={rec['final_kkt']:.2e}",
            flush=True,
        )
    print(f"\nwrote {len(records)} runs -> {DATA}", flush=True)


if __name__ == "__main__":
    main()

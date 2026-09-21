"""Centralized multiple-shooting NLP solved with CasADi/Ipopt."""

import time
import numpy as np
from .common import (
    BaselineResult,
    kkt_residual,
    recover_lambda,
    pack_traj,
    initial_guess,
)
from ..ComputeKKT import cost
from .. import burgers_setting as bs
from . import _nlp


def solve_ipopt(prob, tol_kkt=1e-6, max_iters=200, verbose=False, method="IPOPT"):
    N = prob.N
    t0 = time.time()
    opti, x, u = _nlp.make_opti(prob, N)
    opti.subject_to(x[0, :].T == prob.x0)
    obj = _nlp.stage_cost(
        x, u, prob, [prob.x_des[k] for k in range(N)], N
    ) + _nlp.terminal_cost(x, prob, prob.x_des[N], N)
    opti.minimize(obj)
    X0, _ = initial_guess(prob)
    if verbose:
        nvar = (N + 1) * bs.N_X + N * bs.N_U
        print(
            f"  [IPOPT] monolithic solve: N={N} stages, {nvar} vars, tol={tol_kkt}, "
            f"max_iter={max_iters} — Ipopt iteration table follows",
            flush=True,
        )
    xv, uv, info = _nlp.solve_opti(
        opti,
        x,
        u,
        X0,
        np.zeros((N, bs.N_U)),
        tol=tol_kkt,
        max_iters=max_iters,
        verbose=verbose,
    )
    if verbose:
        print(f"  [IPOPT] solve returned after {info['iters']} Ipopt iters", flush=True)
    wall = time.time() - t0
    z = pack_traj(prob, xv, uv)
    lam = recover_lambda(prob, z)
    kkt, feas = kkt_residual(prob, z, lam)
    # centralized: 1 solve ⇒ parallel == serial == wall
    return BaselineResult(
        method,
        kkt <= tol_kkt,
        info["iters"],
        cost(prob, z),
        kkt,
        feas,
        z=z,
        lam=lam,
        extra={
            "stop_reason": "kkt" if kkt <= tol_kkt else "cap",
            "ipopt_iters": info["iters"],
            "flops": info["flops"],
            "n_call": info["n_call"],
            "t_subsolve_serial": wall,
            "t_subsolve_parallel": wall,
            "time": wall,
        },
    )

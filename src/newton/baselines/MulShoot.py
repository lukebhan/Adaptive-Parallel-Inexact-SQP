"""Iterative multiple shooting with pinned endpoints and cached segment NLPs."""

import time
from collections import defaultdict
import numpy as np
from .common import (
    BaselineResult,
    kkt_residual,
    recover_lambda,
    pack_traj,
    initial_guess,
)
from ..ComputeKKT import cost
from ..composition import uniform_knots, core_range
from .. import burgers_setting as bs
from . import _nlp


def _make_build(prob, m1, m2, is_first, is_last):
    nL = m2 - m1

    def build(opti, x, u, P):
        opti.subject_to(x[0, :].T == (prob.x0 if is_first else P["xL"]))
        if not is_last:
            opti.subject_to(x[nL, :].T == P["xR"])  # hard terminal matching
        obj = _nlp.stage_cost(x, u, prob, [prob.x_des[m1 + k] for k in range(nL)], nL)
        if is_last:
            obj = obj + _nlp.terminal_cost(x, prob, prob.x_des[m2], nL)
        return obj

    return build


def solve_multishoot(
    prob,
    M=10,
    max_outer=30,
    tol_kkt=1e-6,
    tol_step=1e-6,
    subproblem_tol=1e-8,
    subproblem_max_iters=200,
    verbose=False,
    method="MultiShoot",
):
    t0 = time.time()
    N = prob.N
    nx = bs.N_X
    knots = uniform_knots(N, M)
    rng = []
    for i in range(M):
        n_i, n_ip1 = core_range(knots, i)
        rng.append((max(n_i - 1, 0), min(n_ip1 + 1, N)))
    pspec = [("xL", nx), ("xR", nx)]
    segs = [
        _nlp.CachedSegment(
            prob,
            m2 - m1,
            pspec,
            _make_build(prob, m1, m2, i == 0, i == M - 1),
            tol=subproblem_tol,
            max_iters=subproblem_max_iters,
        )
        for i, (m1, m2) in enumerate(rng)
    ]
    X, U = initial_guess(prob)
    converged, stop, last_it, tot_it = False, "max_iters", 0, 0
    tot_flops, t_ser, t_par, ncall = 0.0, 0.0, 0.0, defaultdict(int)
    hist = []
    for it in range(max_outer):
        last_it = it
        subs, st = [], []
        for i in range(M):
            m1, m2 = rng[i]
            ts = time.perf_counter()
            xs, us, info = segs[i].solve(
                dict(xL=X[m1], xR=X[m2]), X[m1 : m2 + 1], U[m1:m2]
            )
            st.append(time.perf_counter() - ts)
            tot_it += info["iters"]
            tot_flops += info["flops"]
            for k, v in info["n_call"].items():
                ncall[k] += v
            subs.append((m1, m2, xs, us))
        t_ser += sum(st)
        t_par += max(st)
        Xn, Un = X.copy(), U.copy()
        for i, (m1, m2, xs, us) in enumerate(subs):
            n_i, n_ip1 = core_range(knots, i)
            for k in range(n_i, n_ip1):
                j = k - m1
                Xn[k] = xs[j]
                if k < N:
                    Un[k] = us[j]
            if i == M - 1:
                Xn[N] = xs[-1]
        step = np.linalg.norm(np.concatenate([(Xn - X).ravel(), (Un - U).ravel()]))
        X, U = Xn, Un
        z = pack_traj(prob, X, U)
        lam = recover_lambda(prob, z)
        kkt, feas = kkt_residual(prob, z, lam)
        hist.append(kkt)
        if verbose:
            print(
                f"  [MultiShoot] it={it} |gL|={kkt:.3e} step={step:.3e} feas={feas:.3e}",
                flush=True,
            )
        if kkt <= tol_kkt:
            converged, stop = True, "kkt"
            break
        if step <= tol_step:
            converged, stop = True, "step"
            break
    z = pack_traj(prob, X, U)
    lam = recover_lambda(prob, z)
    kkt, feas = kkt_residual(prob, z, lam)
    return BaselineResult(
        method,
        converged,
        last_it + 1,
        cost(prob, z),
        kkt,
        feas,
        z=z,
        lam=lam,
        history=hist,
        extra={
            "stop_reason": stop,
            "ipopt_iters": tot_it,
            "flops": tot_flops,
            "n_call": dict(ncall),
            "t_subsolve_serial": t_ser,
            "t_subsolve_parallel": t_par,
            "time": time.time() - t0,
        },
    )

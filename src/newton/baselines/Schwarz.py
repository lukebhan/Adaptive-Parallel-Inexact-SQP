"""Overlapping Schwarz with cached NLPs, penalized interfaces, and core composition."""

import time
from collections import defaultdict
import numpy as np
import casadi as ca
from .common import (
    BaselineResult,
    kkt_residual,
    recover_lambda,
    pack_traj,
    initial_guess,
)
from ..ComputeKKT import cost, lam_indices
from ..composition import uniform_knots, subproblem_range, core_range
from .. import burgers_setting as bs
from . import _nlp


def _make_build(prob, m1, m2, is_last, mu):
    nL = m2 - m1
    swing = _nlp._is_swing(prob)
    dt = prob.dt
    if not swing:  # Burgers IMEX interface step
        Mimp, C1 = _nlp._ops(prob)
        Minv = np.linalg.inv(Mimp)
    Qd, Rd = np.diag(prob.Q), np.diag(prob.R)

    def build(opti, x, u, P):
        opti.subject_to(x[0, :].T == P["d1"])
        obj = _nlp.stage_cost(x, u, prob, [prob.x_des[m1 + k] for k in range(nL)], nL)
        if is_last:
            obj = obj + _nlp.terminal_cost(x, prob, prob.x_des[m2], nL)
        else:
            xT = x[nL, :].T
            if swing:  # f_{m2}(x_{m2}, d3) for the Lagrange term
                fterm = _nlp.swing_next_state(prob, xT, P["d3"])
            else:
                fterm = Minv @ (xT - dt * xT * (C1 @ xT) + dt * P["d3"])
            dxT = xT - prob.x_des[m2]
            obj = (
                obj
                + ca.dot(Qd, dxT * dxT)
                + ca.dot(Rd, P["d3"] * P["d3"])
                - ca.dot(P["d4"], fterm)
                + 0.5 * mu * ca.dot(xT - P["d2"], xT - P["d2"])
            )
        return obj

    return build


def solve_schwarz(
    prob,
    M=10,
    b=1,
    mu_pen=10.0,
    max_outer=30,
    tol_kkt=1e-6,
    tol_step=1e-6,
    subproblem_tol=1e-8,
    subproblem_max_iters=200,
    verbose=False,
    method="Schwarz",
):
    t0 = time.time()
    N = prob.N
    nx, nu = bs.N_X, bs.N_U
    knots = uniform_knots(N, M)
    rng = [subproblem_range(knots, i, b, N) for i in range(M)]
    pspec = [("d1", nx), ("d2", nx), ("d3", nu), ("d4", nx)]
    segs = [
        _nlp.CachedSegment(
            prob,
            m2 - m1,
            pspec,
            _make_build(prob, m1, m2, i == M - 1, mu_pen),
            tol=subproblem_tol,
            max_iters=subproblem_max_iters,
        )
        for i, (m1, m2) in enumerate(rng)
    ]
    X, U = initial_guess(prob)
    z = pack_traj(prob, X, U)
    lam = recover_lambda(prob, z)
    converged, stop, last_it, tot_it = False, "max_iters", 0, 0
    tot_flops, t_ser, t_par, ncall = 0.0, 0.0, 0.0, defaultdict(int)
    hist = []
    for it in range(max_outer):
        last_it = it
        subs, st = [], []
        for i in range(M):
            m1, m2 = rng[i]
            is_last = i == M - 1
            # At N, drop downstream coupling but retain the consensus penalty.
            no_right = is_last or m2 >= N
            pv = dict(
                d1=X[m1],
                d2=X[m2],
                d3=(np.zeros(nu) if no_right else U[min(m2, N - 1)]),
                d4=(np.zeros(nx) if no_right else lam[lam_indices(m2 + 1)]),
            )
            ts = time.perf_counter()
            xs, us, info = segs[i].solve(pv, X[m1 : m2 + 1], U[m1:m2])
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
                f"  [Schwarz] it={it} |gL|={kkt:.3e} step={step:.3e} feas={feas:.3e}",
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

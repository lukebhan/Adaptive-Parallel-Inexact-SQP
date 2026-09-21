"""Consensus ADMM with cached segment NLPs, over-relaxation, and adaptive penalties."""

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
from ..ComputeKKT import cost
from ..composition import uniform_knots
from .. import burgers_setting as bs
from . import _nlp


def _make_build(prob, m1, m2, i_first, i_last):
    nL = m2 - m1

    def build(opti, x, u, P):
        if i_first:
            opti.subject_to(x[0, :].T == prob.x0)
        obj = _nlp.stage_cost(x, u, prob, [prob.x_des[m1 + k] for k in range(nL)], nL)
        if i_last:
            obj = obj + _nlp.terminal_cost(x, prob, prob.x_des[m2], nL)
        if not i_first:
            dL = x[0, :].T - P["zL"]
            obj = obj + ca.dot(P["lamL"], dL) + 0.5 * P["rho"] * ca.dot(dL, dL)
        if not i_last:
            dR = x[nL, :].T - P["zR"]
            obj = obj + ca.dot(P["lamR"], dR) + 0.5 * P["rho"] * ca.dot(dR, dR)
        return obj

    return build


def solve_admm(
    prob,
    M=10,
    rho=1.0,
    max_outer=300,
    tol_kkt=1e-6,
    tol_step=1e-3,
    subproblem_tol=1e-9,
    subproblem_max_iters=200,
    alpha_relax=1.6,
    adapt_rho=True,
    mu_bal=10.0,
    tau_inc=2.0,
    verbose=False,
    method="ADMM",
):
    t0 = time.time()
    N = prob.N
    knots = uniform_knots(N, M)
    nik = M - 1
    nx = bs.N_X
    pspec = [("zL", nx), ("lamL", nx), ("zR", nx), ("lamR", nx), ("rho", 1)]
    segs = [
        _nlp.CachedSegment(
            prob,
            knots[i + 1] - knots[i],
            pspec,
            _make_build(prob, knots[i], knots[i + 1], i == 0, i == M - 1),
            tol=subproblem_tol,
            max_iters=subproblem_max_iters,
        )
        for i in range(M)
    ]
    X, U = initial_guess(prob)
    z_cons = [X[knots[i + 1]].copy() for i in range(1, M)]
    lam_left = [np.zeros(nx) for _ in range(nik)]
    lam_right = [np.zeros(nx) for _ in range(nik)]
    converged, stop, last_it, tot_it = False, "max_iters", 0, 0
    tot_flops, t_ser, t_par, ncall = 0.0, 0.0, 0.0, defaultdict(int)
    hist = []
    for it in range(max_outer):
        last_it = it
        sub_x, sub_u, stt = [None] * M, [None] * M, []
        for i in range(M):
            m1, m2 = knots[i], knots[i + 1]
            pv = dict(
                zL=(prob.x0 if i == 0 else z_cons[i - 1]),
                lamL=(np.zeros(nx) if i == 0 else lam_left[i - 1]),
                zR=(np.zeros(nx) if i == M - 1 else z_cons[i]),
                lamR=(np.zeros(nx) if i == M - 1 else lam_right[i]),
                rho=rho,
            )
            ts = time.perf_counter()
            xs, us, info = segs[i].solve(pv, X[m1 : m2 + 1], U[m1:m2])
            stt.append(time.perf_counter() - ts)
            tot_it += info["iters"]
            tot_flops += info["flops"]
            for k, v in info["n_call"].items():
                ncall[k] += v
            sub_x[i], sub_u[i] = xs, us
        t_ser += sum(stt)
        t_par += max(stt)
        z_old = [z.copy() for z in z_cons]
        xhat_r = [
            alpha_relax * sub_x[j][-1] + (1 - alpha_relax) * z_old[j]
            for j in range(nik)
        ]
        xhat_l = [
            alpha_relax * sub_x[j + 1][0] + (1 - alpha_relax) * z_old[j]
            for j in range(nik)
        ]
        for j in range(nik):
            z_cons[j] = 0.5 * (xhat_r[j] + xhat_l[j])
        for j in range(nik):
            lam_right[j] = lam_right[j] + rho * (xhat_r[j] - z_cons[j])
            lam_left[j] = lam_left[j] + rho * (xhat_l[j] - z_cons[j])
        if adapt_rho and nik:
            r_pri = np.sqrt(
                sum(
                    np.sum((sub_x[j][-1] - z_cons[j]) ** 2)
                    + np.sum((sub_x[j + 1][0] - z_cons[j]) ** 2)
                    for j in range(nik)
                )
            )
            s_dual = rho * np.sqrt(
                sum(np.sum((z_cons[j] - z_old[j]) ** 2) for j in range(nik))
            )
            if r_pri > mu_bal * s_dual:
                rho *= tau_inc
            elif s_dual > mu_bal * r_pri:
                rho /= tau_inc
        Xn, Un = X.copy(), U.copy()
        for i in range(M):
            m1, m2 = knots[i], knots[i + 1]
            for k in range(m1, m2 + 1):
                Xn[k] = sub_x[i][k - m1]
            for k in range(m1, m2):
                if k < N:
                    Un[k] = sub_u[i][k - m1]
        step = np.linalg.norm(np.concatenate([(Xn - X).ravel(), (Un - U).ravel()]))
        X, U = Xn, Un
        z = pack_traj(prob, X, U)
        lam = recover_lambda(prob, z)
        kkt, feas = kkt_residual(prob, z, lam)
        hist.append(kkt)
        if verbose:
            print(
                f"  [ADMM] it={it} |gL|={kkt:.3e} step={step:.3e} feas={feas:.3e} rho={rho:.1f}",
                flush=True,
            )
        if kkt <= tol_kkt:
            converged, stop = True, "kkt"
            break
        if step <= tol_step and it > 1:
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

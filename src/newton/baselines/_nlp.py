"""Shared CasADi dynamics and cached Ipopt segment solvers."""

import numpy as np
import casadi as ca
from .. import burgers_setting as bs


def _ops(prob):
    nx = bs.N_X
    Mimp = np.eye(nx) - prob.dt * prob.params.nu * prob.params.D2
    return Mimp, prob.params.C1


def _is_swing(prob):
    return type(prob.params).__name__ == "SwingParams"


def add_dynamics(opti, x, u, prob, n_local):
    """Symbolic dynamics constraints for stages 0..n_local-1 (x:(n_local+1)×N_X).
    Dispatches on the problem: viscous-Burgers IMEX, or NE39 swing (single fixed mode)."""
    if _is_swing(prob):
        _add_swing_dynamics(opti, x, u, prob, n_local)
    else:
        _add_burgers_dynamics(opti, x, u, prob, n_local)


def _add_burgers_dynamics(opti, x, u, prob, n_local):
    Mimp, C1 = _ops(prob)
    dt = prob.dt
    for k in range(n_local):
        xk = x[k, :].T
        conv = xk * (C1 @ xk)  # element-wise x_i (C1 x)_i
        opti.subject_to(Mimp @ x[k + 1, :].T == xk - dt * conv + dt * u[k, :].T)


def swing_next_state(prob, theta_omega, u):
    """Symbolic Euler step for Swing states (theta, omega) and generator controls."""
    p = prob.params
    n = p.n
    dt = prob.dt
    Minv = ca.DM(p.Minv.reshape(-1, 1))
    D = ca.DM(p.D.reshape(-1, 1))
    Pm = ca.DM(p.Pm.reshape(-1, 1))
    K = p.K
    theta = theta_omega[:n]
    omega = theta_omega[n:]
    g = ca.vertcat(
        *[
            sum(K[i, j] * ca.sin(theta[i] - theta[j]) for j in range(n))
            for i in range(n)
        ]
    )
    theta_n = theta + dt * 2.0 * np.pi * omega
    omega_n = omega + dt * (Minv * (Pm - D * omega - u - g))
    return ca.vertcat(theta_n, omega_n)


def _add_swing_dynamics(opti, x, u, prob, n_local):
    for k in range(n_local):
        opti.subject_to(x[k + 1, :].T == swing_next_state(prob, x[k, :].T, u[k, :].T))


def stage_cost(x, u, prob, x_des_rows, n_local):
    """Σ_{k=0}^{n_local-1} (x_k-x_des)ᵀQ(x_k-x_des) + u_kᵀR u_k  (no-½; optimum-equivalent)."""
    Qd = np.diag(prob.Q)
    Rd = np.diag(prob.R)
    c = 0
    for k in range(n_local):
        dxk = x[k, :].T - x_des_rows[k]
        c = c + ca.dot(Qd, dxk * dxk) + ca.dot(Rd, u[k, :].T * u[k, :].T)
    return c


def terminal_cost(x, prob, x_des_last, n_local):
    """(x_N - x_des)ᵀ QN (x_N - x_des) at local stage n_local."""
    QNd = np.diag(prob.QN)
    dxN = x[n_local, :].T - x_des_last
    return ca.dot(QNd, dxN * dxN)


def make_opti(prob, n_local):
    """Opti with x (n_local+1, N_X), u (n_local, N_U) and Burgers dynamics added."""
    opti = ca.Opti()
    x = opti.variable(n_local + 1, bs.N_X)
    u = opti.variable(n_local, bs.N_U)
    add_dynamics(opti, x, u, prob, n_local)
    return opti, x, u


_NCALL = [
    "n_call_nlp_f",
    "n_call_nlp_g",
    "n_call_nlp_grad_f",
    "n_call_nlp_jac_g",
    "n_call_nlp_hess_l",
]


def _info_from_stats(st, n_stages):
    """Ipopt iter count + internal eval counts + a FLOP estimate for the segment."""
    from ..flop_model import ipopt_iter_flops

    it = int(st.get("iter_count", 0))
    ncall = {k: int(st.get(k, 0)) for k in _NCALL}
    flops = it * ipopt_iter_flops(n_stages, bs.N_X, bs.N_U)
    return dict(iters=it, n_call=ncall, flops=flops, n_stages=n_stages)


class CachedSegment:
    """Cache a parametric segment NLP across solves.

    params_spec lists (name, dimension); build_fn adds constraints and returns the objective."""

    def __init__(self, prob, n_local, params_spec, build_fn, tol=1e-9, max_iters=200):
        self.opti = ca.Opti()
        self.x = self.opti.variable(n_local + 1, bs.N_X)
        self.u = self.opti.variable(n_local, bs.N_U)
        add_dynamics(self.opti, self.x, self.u, prob, n_local)
        self.P = {name: self.opti.parameter(dim) for name, dim in params_spec}
        self.opti.minimize(build_fn(self.opti, self.x, self.u, self.P))
        self.opti.solver(
            "ipopt",
            {
                "print_time": False,
                "ipopt": {
                    "print_level": 0,
                    "tol": tol,
                    "max_iter": max_iters,
                    "sb": "yes",
                },
            },
        )
        self.n_local = n_local

    def solve(self, pvals, x_init, u_init):
        for k, v in pvals.items():
            self.opti.set_value(self.P[k], v)
        self.opti.set_initial(self.x, x_init)
        self.opti.set_initial(self.u, u_init)
        try:
            sol = self.opti.solve()
            st = sol.stats()
            X, U = sol.value(self.x), sol.value(self.u)
        except RuntimeError:
            st = self.opti.stats()
            X, U = self.opti.debug.value(self.x), self.opti.debug.value(self.u)
        return (
            np.array(X).reshape(-1, bs.N_X),
            np.array(U).reshape(-1, bs.N_U),
            _info_from_stats(st, self.n_local),
        )


def solve_opti(opti, x, u, x_init, u_init, tol=1e-8, max_iters=200, verbose=False):
    """Solve with Ipopt and return states, controls, iteration counts, and estimated work."""
    n_stages = u_init.shape[0]
    opti.set_initial(x, x_init)
    opti.set_initial(u, u_init)
    iopts = {
        "print_level": (5 if verbose else 0),
        "tol": tol,
        "max_iter": max_iters,
        "sb": "yes",
    }
    if verbose:
        iopts["print_frequency_iter"] = 1
    opti.solver("ipopt", {"print_time": bool(verbose), "ipopt": iopts})
    try:
        sol = opti.solve()
        st = sol.stats()
        return (
            np.array(sol.value(x)).reshape(-1, bs.N_X),
            np.array(sol.value(u)).reshape(-1, bs.N_U),
            _info_from_stats(st, n_stages),
        )
    except RuntimeError:
        st = opti.stats()
        return (
            np.array(opti.debug.value(x)).reshape(-1, bs.N_X),
            np.array(opti.debug.value(u)).reshape(-1, bs.N_U),
            _info_from_stats(st, n_stages),
        )

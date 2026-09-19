"""Viscous Burgers IMEX dynamics with analytic Jacobians and Hessian contractions."""

import numpy as np
from . import burgers_setting as bs
from .eval_stats import bump


def _operators(nx):
    """Central-difference D2 (Laplacian) and C1 (first derivative); h = 1/(nx+1)."""
    h = 1.0 / (nx + 1)
    main = -2.0 * np.ones(nx)
    off = np.ones(nx - 1)
    D2 = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2
    C1 = (np.diag(off, 1) - np.diag(off, -1)) / (2 * h)
    return D2, C1, h


class BurgersParams:
    """Dials: nu (viscosity), amp (Peclet ~ amp*h/nu), alpha (control penalty)."""

    def __init__(self, nu=0.02, amp=1.0, alpha=1e-3, target="zero"):
        nx = bs.N_X
        self.nx = nx
        self.nu = float(nu)
        self.amp = float(amp)
        self.alpha = float(alpha)
        self.target = target
        self.D2, self.C1, self.h = _operators(nx)
        self.Minv = np.linalg.inv(
            np.eye(nx) - self.nu * self.D2 * 0.0
        )  # placeholder; set in make_step_fns
        self.Bm = np.eye(nx)


def burgers_grid():
    """Interior spatial grid x_i = i*h, i=1..nx (Dirichlet endpoints excluded)."""
    nx = bs.N_X
    h = 1.0 / (nx + 1)
    return np.linspace(h, 1 - h, nx)


def burgers_x0(p: BurgersParams):
    """Smooth initial bump amp*sin(pi x)."""
    return p.amp * np.sin(np.pi * burgers_grid())


def burgers_target(p: BurgersParams):
    """Tracking target: 0 (or 0.3*amp*sin(2 pi x) if target != 'zero')."""
    if p.target == "zero":
        return np.zeros(bs.N_X)
    return 0.3 * p.amp * np.sin(2 * np.pi * burgers_grid())


def peclet(p: BurgersParams):
    return p.amp * p.h / p.nu


def make_step_fns(dt: float, p: BurgersParams):
    """Return (F, A, B, Hdyn): the IMEX step, its Jacobians, and the dynamics
    Hessian contraction. Mirrors src/burgers.jl:make_step_fns."""
    nx = bs.N_X
    dt = float(dt)
    Minv = np.linalg.inv(np.eye(nx) - dt * p.nu * p.D2)
    C1 = p.C1
    Bm = p.Bm
    p.Minv = Minv  # cache (Schwarz/baselines reuse it)

    def conv(x):
        return x * (C1 @ x)  # non-conservative u u_x

    def F(x, w, k=None):  # k: stage index, unused (time-invariant)
        bump("F")
        return Minv @ (x + dt * (-conv(x) + Bm @ w))

    def A(x, w, k=None):
        bump("jac_x")
        dN = np.diag(C1 @ x) + (x[:, None] * C1)  # d(conv)/dx
        return Minv @ (np.eye(nx) - dt * dN)

    Bconst = dt * (Minv @ Bm)

    def B(x, w, k=None):
        bump("jac_u")
        return Bconst

    def Hdyn(x, u, lam, k=None):
        # -lambda^T d^2F: closed form. mu = Minv^T lambda; H_xx = dt(mu_a C1[a,b] + mu_b C1[b,a]).
        mu = Minv.T @ lam
        D = mu[:, None] * C1
        Hb = np.zeros((nx + bs.N_U, nx + bs.N_U))
        Hb[:nx, :nx] = dt * (D + D.T)
        return Hb

    return F, A, B, Hdyn

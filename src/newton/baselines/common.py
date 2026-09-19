"""Shared baseline result + KKT-residual helpers (mirror of src/baselines/common.jl)."""

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..ComputeKKT import (
    x_indices,
    u_indices,
    grad_lagrangian,
    constraint_jacobian,
    grad_cost,
)
from .. import burgers_setting as bs


@dataclass
class BaselineResult:
    method: str
    converged: bool
    iters: int
    final_cost: float
    final_grad_L_norm: float
    final_feas: float
    z: np.ndarray = None
    lam: np.ndarray = None
    history: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def pack_traj(prob, X, U):
    """(X:(N+1,nx), U:(N,nu)) → flat z."""
    from ..ComputeKKT import n_z

    z = np.zeros(n_z(prob))
    for k in range(prob.N + 1):
        z[x_indices(k)] = X[k]
        if k < prob.N:
            z[u_indices(k)] = U[k]
    return z


def unpack_traj(prob, z):
    X = np.array([z[x_indices(k)] for k in range(prob.N + 1)])
    U = np.array([z[u_indices(k)] for k in range(prob.N)])
    return X, U


def forward_rollout(prob, U):
    """Nonlinear rollout X[0]=x0, X[k+1]=F(X[k],U[k])."""
    X = np.zeros((prob.N + 1, bs.N_X))
    X[0] = prob.x0
    for k in range(prob.N):
        X[k + 1] = prob.F(X[k], U[k], k)
    return X


def initial_guess(prob):
    """Initialize Burgers by open-loop rollout and Swing at its bounded regulation target."""
    U = np.zeros((prob.N, bs.N_U))
    if type(prob.params).__name__ == "SwingParams":
        X = np.zeros((prob.N + 1, bs.N_X))
        X[0] = prob.x0
        return X, U
    return forward_rollout(prob, U), U


def recover_lambda(prob, z, G=None):
    """Estimate multipliers from (G G.T) lambda = -G grad_cost.

    This avoids an unstable backward adjoint sweep on lightly damped dynamics."""
    if G is None:
        G = constraint_jacobian(prob, z)
    gcost = grad_cost(prob, z)
    GGt = (G @ G.T).tocsc()
    GGt = GGt + 1e-10 * sp.eye(
        GGt.shape[0], format="csc"
    )  # regularize the gauge/null space
    lam = spla.spsolve(GGt, -(G @ gcost))
    return np.asarray(lam).ravel()


def kkt_residual(prob, z, lam=None):
    """Return (‖∇L‖, ‖feas‖). If lam is None, recover it by adjoint sweep."""
    if lam is None:
        lam = recover_lambda(prob, z)
    grad_z, f = grad_lagrangian(prob, z, lam)
    return float(np.linalg.norm(np.concatenate([grad_z, f]))), float(np.linalg.norm(f))

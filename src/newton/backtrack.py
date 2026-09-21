"""Armijo backtracking line search on the augmented-Lagrangian merit"""

from .ComputeKKT import n_z
from .CalculateAug import merit


def armijo(
    p,
    z,
    lam,
    direction,
    eta1,
    eta2,
    beta,
    max_backtracks=30,
    current_merit=None,
    current_grad_dot_dir=None,
    G=None,
):
    """Backtrack alpha = 1, ½, ¼, ... until
        L_eta(z+αd, lam+αdl) <= L0 + beta·α·⟨∇L_eta, d⟩.
    Returns (alpha, n_backtracks, accepted)."""
    nz = n_z(p)
    dz = direction[:nz]
    dl = direction[nz:]
    if current_merit is None:
        current_merit = merit(p, z, lam, eta1, eta2, G=G)
    alpha = 1.0
    for k in range(max_backtracks):
        zn = z + alpha * dz
        lamn = lam + alpha * dl
        if (
            merit(p, zn, lamn, eta1, eta2)
            <= current_merit + beta * alpha * current_grad_dot_dir
        ):
            return alpha, k, True
        alpha *= 0.5
    return alpha, max_backtracks, False

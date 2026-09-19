"""Augmented merit: L + eta1/2 ||grad_lambda L||^2 + eta2/2 ||grad_z L||^2."""

import numpy as np
from .ComputeKKT import grad_lagrangian, lagrangian, constraint_jacobian


def merit(p, z, lam, eta1, eta2, G=None):
    grad_z, grad_lam = grad_lagrangian(p, z, lam, G=G)
    L = lagrangian(p, z, lam)
    return L + 0.5 * eta1 * (grad_lam @ grad_lam) + 0.5 * eta2 * (grad_z @ grad_z)


def grad_merit(p, z, lam, eta1, eta2, H=None, G=None):
    """Uses the *unmodified* Lagrangian Hessian H (modification only affects the solve)."""
    grad_z, grad_lam = grad_lagrangian(p, z, lam, G=G)
    if G is None:
        G = constraint_jacobian(p, z)
    grad_z_eta = grad_z + eta2 * (H @ grad_z) + eta1 * (G.T @ grad_lam)
    grad_lam_eta = eta2 * (G @ grad_z) + grad_lam
    return grad_z_eta, grad_lam_eta


def grad_merit_flat(p, z, lam, eta1, eta2, H=None, G=None):
    gz, gl = grad_merit(p, z, lam, eta1, eta2, H=H, G=G)
    return np.concatenate([gz, gl])

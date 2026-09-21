"""Assemble local KKT systems with left anchors and penalized right interfaces."""

import numpy as np
import scipy.sparse as sp
from . import burgers_setting as bs
from .ComputeKKT import (
    x_indices,
    u_indices,
    lam_indices,
    z_size,
    lam_size,
    assemble_kkt,
)
from .composition import subproblem_range


class Subproblem:
    def __init__(self, i_idx, m1, m2, is_last, mu):
        self.i_idx = i_idx
        self.m1 = m1
        self.m2 = m2
        self.is_last = is_last
        self.mu = float(mu)


def n_local(s):
    return s.m2 - s.m1


def n_z_local(s):
    return z_size(n_local(s))


def n_lam_local(s):
    return lam_size(n_local(s))


def build_subproblems(prob, knots, b_list, mu):
    M = len(knots) - 1
    subs = []
    for i in range(M):
        m1, m2 = subproblem_range(knots, i, b_list[i], prob.N)
        subs.append(Subproblem(i, m1, m2, i == M - 1, mu))
    return subs


def build_local_jacobian_sparse(local_A, local_B, nL):
    """Build the anchored local constraint Jacobian as COO blocks."""
    nx = bs.N_X
    nu = bs.N_U
    xs = np.array([x_indices(j).start for j in range(nL + 1)], dtype=np.int64)
    us = (
        np.array([u_indices(j).start for j in range(nL)], dtype=np.int64)
        if nL
        else np.zeros(0, np.int64)
    )
    ar = np.arange(nx)
    jI = np.arange(nL + 1)  # identity blocks (diagonal), rows 0..nL
    rows = [(jI[:, None] * nx + ar[None, :]).ravel()]
    cols = [(xs[:, None] + ar[None, :]).ravel()]
    vals = [np.ones((nL + 1) * nx)]
    if nL:
        j = np.arange(1, nL + 1)
        A = -np.stack(local_A)
        B = -np.stack(local_B)  # (nL,nx,nx), (nL,nx,nu)
        aru = np.arange(nu)
        rows.append(
            np.broadcast_to(
                (j * nx)[:, None, None] + ar[None, :, None], (nL, nx, nx)
            ).ravel()
        )
        cols.append(
            np.broadcast_to(
                xs[:nL][:, None, None] + ar[None, None, :], (nL, nx, nx)
            ).ravel()
        )
        vals.append(A.ravel())
        rows.append(
            np.broadcast_to(
                (j * nx)[:, None, None] + ar[None, :, None], (nL, nx, nu)
            ).ravel()
        )
        cols.append(
            np.broadcast_to(
                us[:, None, None] + aru[None, None, :], (nL, nx, nu)
            ).ravel()
        )
        vals.append(B.ravel())
    G = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(lam_size(nL), z_size(nL)),
    )
    G.eliminate_zeros()  # Exclude structural zeros from the sparse FLOP count.
    return G


def assemble(sub, prob, H_blocks, A_list, B_list, grad_z, f, lam):
    """Return local KKT, RHS, Hessian blocks, and constraint Jacobian.

    Nonterminal interfaces omit downstream gradient coupling; the Hessian adds mu I."""
    nL = n_local(sub)
    nx = bs.N_X

    # 1) local Hessian blocks (interface gets Q + μI for non-last subproblems)
    local_H = [None] * (nL + 1)
    for j in range(nL):
        local_H[j] = H_blocks[sub.m1 + j]
    # Use terminal QN whenever the extended window reaches N; otherwise add mu I.
    reaches_end = sub.m2 == prob.N
    if reaches_end:
        local_H[nL] = H_blocks[prob.N]
    else:
        Q_m2 = H_blocks[sub.m2][:nx, :nx]
        local_H[nL] = Q_m2 + sub.mu * np.eye(nx)

    local_A = [A_list[k] for k in range(sub.m1, sub.m2)]
    local_B = [B_list[k] for k in range(sub.m1, sub.m2)]
    G_i = build_local_jacobian_sparse(local_A, local_B, nL)

    # 3) rhs = -[∇_ω L_QP; ∇_ζ L_QP]
    rhs_omega = np.zeros(n_z_local(sub))
    for j in range(nL):
        k = sub.m1 + j
        rhs_omega[x_indices(j)] = -grad_z[x_indices(k)]
        rhs_omega[u_indices(j)] = -grad_z[u_indices(k)]
    if reaches_end:
        rhs_omega[x_indices(nL)] = -grad_z[
            x_indices(sub.m2)
        ]  # terminal: QN(x_N-x_des)+λ_N
    else:
        # interface: truncate the downstream coupling −A_m2ᵀλ_{m2+1} (add it back to grad_z)
        rhs_omega[x_indices(nL)] = -(
            grad_z[x_indices(sub.m2)] + A_list[sub.m2].T @ lam[lam_indices(sub.m2 + 1)]
        )

    rhs_zeta = np.zeros(n_lam_local(sub))
    # At m1=0 correct initial-condition drift; other windows anchor the direction to zero.
    if sub.m1 == 0:
        rhs_zeta[lam_indices(0)] = -f[lam_indices(0)]
    for j in range(1, nL + 1):
        rhs_zeta[lam_indices(j)] = -f[lam_indices(sub.m1 + j)]
    rhs = np.concatenate([rhs_omega, rhs_zeta])

    Gamma_i = assemble_kkt(local_H, G_i)
    return Gamma_i, rhs, local_H, G_i


def local_residual(Gamma_i, direction, rhs):
    """r̃_i = Γ_i d − rhs."""
    return Gamma_i @ direction - rhs

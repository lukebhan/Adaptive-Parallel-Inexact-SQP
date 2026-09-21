"""Composition (C) and decomposition (D) operators + knot helpers
Knots use zero-based indices."""

import numpy as np
from .ComputeKKT import x_indices, u_indices, lam_indices, z_size, lam_size


def uniform_knots(N, M):
    """Knots n_0=0 < n_1 < ... < n_M=N, uniformly placed."""
    return [int(round(i * N / M)) for i in range(M + 1)]


def subproblem_range(knots, i, b, N):
    """Extended stage range [max(n_i-b,0), min(n_{i+1}+b,N)] for subproblem i."""
    return max(knots[i] - b, 0), min(knots[i + 1] + b, N)


def core_range(knots, i):
    """Core stages [n_i, n_{i+1}) — what composition keeps from subproblem i."""
    return knots[i], knots[i + 1]


def compose(N, knots, b_list, local_dirs, n_z_global, n_lam_global):
    """Build the global direction from each subproblem's core slice.
    local_dirs[i] = (omega_i, zeta_i) flat local primal/dual vectors."""
    M = len(knots) - 1
    dz = np.zeros(n_z_global)
    dl = np.zeros(n_lam_global)
    for i in range(M):
        m1, _ = subproblem_range(knots, i, b_list[i], N)
        omega, zeta = local_dirs[i]
        n_i, n_ip1 = core_range(knots, i)
        for k in range(n_i, n_ip1):
            j = k - m1
            dz[x_indices(k)] = omega[x_indices(j)]
            dz[u_indices(k)] = omega[u_indices(j)]
            dl[lam_indices(k)] = zeta[lam_indices(j)]
        if i == M - 1:
            m2 = subproblem_range(knots, i, b_list[i], N)[1]
            j = m2 - m1
            dz[x_indices(N)] = omega[x_indices(j)]
            dl[lam_indices(N)] = zeta[lam_indices(j)]
    return dz, dl


def decompose_to_local(dz, dl, m1, m2):
    """Project a global direction (dz, dl) onto subproblem window [m1, m2], returning
    the local KKT vector [ω; ζ] (warm-start initial guess for that subproblem). When
    the window has grown across an adaptation pass, the new boundary stages simply read
    the global direction's values there — a better start than zero in the overlap."""
    nL = m2 - m1
    omega = np.zeros(z_size(nL))
    zeta = np.zeros(lam_size(nL))
    for j in range(nL + 1):
        k = m1 + j
        omega[x_indices(j)] = dz[x_indices(k)]
        zeta[lam_indices(j)] = dl[lam_indices(k)]
        if j < nL:
            omega[u_indices(j)] = dz[u_indices(k)]
    return np.concatenate([omega, zeta])


def decompose_residual_norms(N, knots, r_global, n_z):
    """Per-subproblem core residual norms [‖r̃_i‖]_{i=0}^{M-1}."""
    M = len(knots) - 1
    r_z = r_global[:n_z]
    r_lam = r_global[n_z:]
    norms = []
    for i in range(M):
        s = 0.0
        n_i, n_ip1 = core_range(knots, i)
        for k in range(n_i, n_ip1):
            s += float(r_z[x_indices(k)] @ r_z[x_indices(k)])
            s += float(r_z[u_indices(k)] @ r_z[u_indices(k)])
            s += float(r_lam[lam_indices(k)] @ r_lam[lam_indices(k)])
        if i == M - 1:
            s += float(r_z[x_indices(N)] @ r_z[x_indices(N)])
            s += float(r_lam[lam_indices(N)] @ r_lam[lam_indices(N)])
        norms.append(np.sqrt(s))
    return norms

"""Dominant-operation FLOP estimates, not hardware counters.

Sparse matvec: 2*nnz. GMRES adds 2*iters^2*n for orthogonalization.
Banded factorization: approximately 2*d*w^2. ILU build includes Schur
formation plus a 2*nnz(ILU) factorization proxy; apply adds primal scaling."""


def gmres_qlp_flops(iters, nnz, n):
    return 2.0 * iters * nnz + 2.0 * iters * iters * n


def matvec_flops(iters, nnz):
    return 2.0 * iters * nnz


def banded_factor_flops(d, w):
    return 2.0 * float(d) * float(w) ** 2


def ipopt_iter_flops(n_stages, nx, nu):
    """One KKT factorization for an OCP segment of n_stages dynamics steps."""
    d = (n_stages + 1) * nx + n_stages * nu + (n_stages + 1) * nx  # vars + duals
    w = 2 * nx + nu  # half-bandwidth
    return banded_factor_flops(d, w)

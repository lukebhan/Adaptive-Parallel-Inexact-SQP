"""OCP derivatives with primal layout [x0, u0, ..., xN] and dual blocks [lam0, ..., lamN]."""

import numpy as np
import scipy.sparse as sp
from . import burgers_setting as bs
from .burgers import make_step_fns
from .eval_stats import bump


# index helpers (0-indexed, half-open slices)
def x_indices(k):
    return slice(k * (bs.N_X + bs.N_U), k * (bs.N_X + bs.N_U) + bs.N_X)


def u_indices(k):
    return slice(k * (bs.N_X + bs.N_U) + bs.N_X, (k + 1) * (bs.N_X + bs.N_U))


def lam_indices(k):
    return slice(k * bs.N_X, (k + 1) * bs.N_X)


def z_size(N):
    return (N + 1) * bs.N_X + N * bs.N_U


def lam_size(N):
    return (N + 1) * bs.N_X


class Problem:
    """Discrete-time OCP with dynamics oracles (F, A, B, Hdyn) and global stage dimensions."""

    def __init__(self, N, dt, Q, R, QN, x_des, x0, params, step_fns=None):
        self.N = int(N)
        self.dt = float(dt)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.QN = np.asarray(QN, float)
        self.x_des = np.asarray(x_des, float)  # (N+1, N_X)
        self.x0 = np.asarray(x0, float)
        self.params = params
        if step_fns is None:
            self.F, self.A, self.B, self.Hdyn = make_step_fns(dt, params)
        else:
            self.F, self.A, self.B, self.Hdyn = step_fns


def n_z(p):
    return z_size(p.N)


def n_lam(p):
    return lam_size(p.N)


def n_kkt(p):
    return n_z(p) + n_lam(p)


def xdes_row(p, k):
    return p.x_des[k]


# cost
def cost(p, z):
    bump("cost")
    total = 0.0
    for k in range(p.N):
        dx = z[x_indices(k)] - xdes_row(p, k)
        uk = z[u_indices(k)]
        total += dx @ (p.Q @ dx) + uk @ (p.R @ uk)
    dxN = z[x_indices(p.N)] - xdes_row(p, p.N)
    total += dxN @ (p.QN @ dxN)
    return 0.5 * total  # ½ convention (matches paper/Python)


def grad_cost(p, z):
    bump("grad_cost")
    g = np.zeros(n_z(p))
    for k in range(p.N):
        g[x_indices(k)] = p.Q @ (z[x_indices(k)] - xdes_row(p, k))
        g[u_indices(k)] = p.R @ z[u_indices(k)]
    g[x_indices(p.N)] = p.QN @ (z[x_indices(p.N)] - xdes_row(p, p.N))
    return g


def cost_hess_block_stage(p):
    H = np.zeros((bs.N_X + bs.N_U, bs.N_X + bs.N_U))
    H[: bs.N_X, : bs.N_X] = p.Q
    H[bs.N_X :, bs.N_X :] = p.R
    return H


def cost_hess_block_terminal(p):
    return np.array(p.QN, dtype=float)


# rollout
def rollout(p, x0, u_seq):
    """Forward rollout from x0 with controls u_seq (N, N_U) → flat z."""
    bump("rollout")
    z = np.zeros(n_z(p))
    x = np.array(x0, float)
    z[x_indices(0)] = x
    for k in range(p.N):
        uk = u_seq[k]
        z[u_indices(k)] = uk
        x = p.F(x, uk, k)
        z[x_indices(k + 1)] = x
    return z


# constraints
def constraint_residual(p, z):
    """f: f[0]=x_0 - x0_given; f[k+1]=x_{k+1} - F(x_k,u_k)."""
    bump("constraint")
    f = np.zeros(n_lam(p))
    f[lam_indices(0)] = z[x_indices(0)] - p.x0
    for k in range(p.N):
        xk = z[x_indices(k)]
        uk = z[u_indices(k)]
        f[lam_indices(k + 1)] = z[x_indices(k + 1)] - p.F(xk, uk, k)
    return f


def stage_jacobians(p, z):
    """Per-stage A_k = dF/dx, B_k = dF/dw (k=0..N-1)."""
    A_list = [p.A(z[x_indices(k)], z[u_indices(k)], k) for k in range(p.N)]
    B_list = [p.B(z[x_indices(k)], z[u_indices(k)], k) for k in range(p.N)]
    return A_list, B_list


def constraint_jacobian(p, z, A_list=None, B_list=None):
    """Sparse G = df/dz. Row-block 0: I (initial). Row-block k+1: [I@x_{k+1}, -A_k@x_k, -B_k@u_k]."""
    bump("jacobian")
    if A_list is None:
        A_list, B_list = stage_jacobians(p, z)
    nx = bs.N_X
    rows, cols, vals = [], [], []

    def add(M, r0, c0):
        rr, cc = np.nonzero(M)
        rows.extend((r0 + rr).tolist())
        cols.extend((c0 + cc).tolist())
        vals.extend(M[rr, cc].tolist())

    # initial-condition block: I at x_0
    add(np.eye(nx), 0, x_indices(0).start)
    for k in range(p.N):
        r0 = (k + 1) * nx
        add(np.eye(nx), r0, x_indices(k + 1).start)  # I @ x_{k+1}
        add(-A_list[k], r0, x_indices(k).start)  # -A_k @ x_k
        add(-B_list[k], r0, u_indices(k).start)  # -B_k @ u_k
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_lam(p), n_z(p)))


# Lagrangian gradient / Hessian
def grad_lagrangian(p, z, lam, G=None):
    bump("grad_L")
    f = constraint_residual(p, z)
    if G is None:
        G = constraint_jacobian(p, z)
    grad_z = grad_cost(p, z) + G.T @ lam
    return grad_z, f


def lagrangian(p, z, lam):
    return cost(p, z) + lam @ constraint_residual(p, z)


def hessian_blocks(p, z, lam, gauss_newton=False):
    """Per-stage KKT (1,1) blocks. GN: cost only; else + Hdyn (full Lagrangian)."""
    bump("hessian")
    cost_block = cost_hess_block_stage(p)
    blocks = [None] * (p.N + 1)
    for k in range(p.N):
        if gauss_newton:
            blocks[k] = cost_block.copy()
        else:
            xk = z[x_indices(k)]
            uk = z[u_indices(k)]
            ln = lam[lam_indices(k + 1)]
            blocks[k] = cost_block + p.Hdyn(xk, uk, ln, k)
    blocks[p.N] = cost_hess_block_terminal(p)
    return blocks


def modify_hessian_blocks(blocks, xi_H):
    """Add sigma I to each block so eig_min >= xi_H (σ=0 if already PD enough)."""
    modified, sigmas, n_mods = [], [], 0
    for H in blocks:
        eig_min = np.linalg.eigvalsh(0.5 * (H + H.T)).min()
        sigma = max(0.0, xi_H - eig_min)
        if sigma > 0:
            n_mods += 1
        modified.append(H + sigma * np.eye(H.shape[0]))
        sigmas.append(sigma)
    return modified, sigmas, n_mods, (max(sigmas) if sigmas else 0.0)


def assemble_kkt(H_blocks, G):
    """Assemble [[blockdiag(H), G.T], [G, 0]] using COO triplets.

    Batch equal-sized leading blocks; append any remaining blocks individually."""
    sizes = np.array([b.shape[0] for b in H_blocks], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nz = int(sizes.sum())
    nl = G.shape[0]
    n = nz + nl
    R, C, D = [], [], []
    d0 = sizes[0]
    nuni = int(np.sum(sizes == d0))
    if nuni and (sizes[:nuni] == d0).all():  # vectorize the leading equal-size run
        ar = np.arange(d0)
        blk = np.stack(
            [np.asarray(H_blocks[k], float) for k in range(nuni)]
        )  # (nuni,d0,d0)
        o = offs[:nuni]
        R.append(
            np.broadcast_to(
                o[:, None, None] + ar[None, :, None], (nuni, d0, d0)
            ).ravel()
        )
        C.append(
            np.broadcast_to(
                o[:, None, None] + ar[None, None, :], (nuni, d0, d0)
            ).ravel()
        )
        D.append(blk.ravel())
        rest = range(nuni, len(H_blocks))
    else:
        rest = range(len(H_blocks))
    for k in rest:  # remaining / differently-sized blocks
        b = np.asarray(H_blocks[k], float)
        d = b.shape[0]
        ar = np.arange(d)
        R.append(offs[k] + np.repeat(ar, d))
        C.append(offs[k] + np.tile(ar, d))
        D.append(b.ravel())
    g = G.tocoo()
    R += [nz + g.row, g.col]
    C += [g.col, nz + g.row]
    D += [g.data, g.data]  # G (bottom-left) + Gᵀ (top-right)
    Gamma = sp.csr_matrix(
        (np.concatenate(D), (np.concatenate(R), np.concatenate(C))), shape=(n, n)
    )
    Gamma.eliminate_zeros()  # match reference nnz (dense-H structural zeros)
    return Gamma

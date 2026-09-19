"""KKT solvers and Schur preconditioners. Solvers return a direction and work counters."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import qr, solve_triangular, lu_factor, lu_solve
from . import burgers_setting as bs


# ====================== preconditioners ======================
class SchurApproxPreconditioner:
    """Block-diagonal inverse of diag(H) and regularized stage Schur blocks."""

    def __init__(self, Hinv_diag, Sinv_blocks, n_z, n_lam):
        self.Hinv_diag = Hinv_diag
        self.Sinv_blocks = Sinv_blocks
        self.n_z = n_z
        self.n_lam = n_lam
        self.shape = (n_z + n_lam, n_z + n_lam)
        # per-apply FLOPs: diag(H)⁻¹ scale (n_z) + per-stage dense block mat-vecs (2·blk²);
        # build: per-stage block inversions (≈⅔·blk³).
        self.apply_flops = float(n_z) + sum(2.0 * b.shape[0] ** 2 for b in Sinv_blocks)
        self.build_flops = sum((2.0 / 3.0) * b.shape[0] ** 3 for b in Sinv_blocks)

    def __call__(self, r):
        nx = bs.N_X
        zp = r[: self.n_z] * self.Hinv_diag
        zd = np.empty(self.n_lam)
        rd = r[self.n_z :]
        for s, Sinv in enumerate(self.Sinv_blocks):
            zd[s * nx : (s + 1) * nx] = Sinv @ rd[s * nx : (s + 1) * nx]
        return np.concatenate([zp, zd])

    def matvec(self, r):
        return self(r)

    def as_linop(self):
        n = self.shape[0]
        return spla.LinearOperator((n, n), matvec=self.__call__)


def schur_approx_preconditioner(H_blocks, G, n_z_local, n_lam_local):
    # Preserve product order: near-singularity amplifies roundoff at accuracy gates.
    nx = bs.N_X
    Hd = np.concatenate([np.diag(np.asarray(Hb)) for Hb in H_blocks])
    Hinv = sp.diags(1.0 / Hd)
    Sm = (G @ Hinv @ G.T).tocsr()  # S = G diag(1/Hd) Gᵀ
    nb = n_lam_local // nx
    Sinv_blocks = []
    for s in range(nb):
        blk = Sm[s * nx : (s + 1) * nx, s * nx : (s + 1) * nx].toarray()
        Sinv_blocks.append(np.linalg.inv(blk + 1e-12 * np.eye(nx)))
    return SchurApproxPreconditioner(1.0 / Hd, Sinv_blocks, n_z_local, n_lam_local)


class ILUSchurPreconditioner:
    """Block-diagonal preconditioner using diag(H) and ILU of its Schur complement."""

    def __init__(self, Hinv_diag, ilu, n_z, n_lam, schur_form_flops=0.0):
        self.Hinv_diag = Hinv_diag
        self.ilu = ilu
        self.n_z = n_z
        self.n_lam = n_lam
        self.shape = (n_z + n_lam, n_z + n_lam)
        # per-apply FLOPs: diag(H)⁻¹ scale (n_z) + L,U triangular solves (2·nnz of the ILU factors).
        self.apply_flops = float(n_z) + 2.0 * float(ilu.nnz)
        # Build work includes Schur formation and a 2*nnz(ILU) factorization proxy.
        self.schur_form_flops = float(schur_form_flops)
        self.build_flops = float(schur_form_flops) + 2.0 * float(ilu.nnz)

    def __call__(self, r):
        return np.concatenate(
            [r[: self.n_z] * self.Hinv_diag, self.ilu.solve(r[self.n_z :])]
        )

    def matvec(self, r):
        return self(r)

    def as_linop(self):
        n = self.shape[0]
        return spla.LinearOperator((n, n), matvec=self.__call__)


def ilu_schur_preconditioner(
    H_blocks, G, n_z_local, n_lam_local, drop_tol=1e-2, fill_factor=20
):
    """Build the ILU-Schur preconditioner (mirror of schur_approx_preconditioner's S)."""
    Hd = np.concatenate([np.diag(np.asarray(Hb)) for Hb in H_blocks])
    A = (G @ sp.diags(1.0 / Hd)).tocsr()  # column-scale G: nnz(G) mults
    Sm = (A @ G.T).tocsc()
    # exact FLOPs to form S = A·Gᵀ: Σ_i Σ_{k∈nz(A_i)} nnz((Gᵀ)_k) multiply-adds + the scaling mults
    B = G.T.tocsr()
    nnzB = np.diff(B.indptr)
    schur_form_flops = float(G.nnz) + 2.0 * float(nnzB[A.indices].sum())
    Sreg = Sm + 1e-12 * sp.eye(Sm.shape[0], format="csc")
    ilu = spla.spilu(Sreg, drop_tol=drop_tol, fill_factor=fill_factor)
    return ILUSchurPreconditioner(
        1.0 / Hd, ilu, n_z_local, n_lam_local, schur_form_flops=schur_form_flops
    )


def _build_schur(H_blocks, G):
    """Schur complement S = G diag(H)⁻¹ Gᵀ (csc) and Hinv diagonal — mirror of the ILU build."""
    Hd = np.concatenate([np.diag(np.asarray(Hb)) for Hb in H_blocks])
    Hinv = 1.0 / Hd
    S = (G @ sp.diags(Hinv) @ G.T).tocsc()
    return S, Hinv


class BorderedILUSchur:
    """Reuse an interior ILU and factor a dense Schur border when overlap grows.

    Both endpoint extensions enter the border. The caller rebuilds stale factors."""

    def __init__(self, base_ilu, Hinv, n_z, n_lam, base_m1, base_m2, nx):
        self.base_ilu = base_ilu
        self.Hinv = Hinv  # current full primal diag(H)⁻¹
        self.n_z, self.n_lam = n_z, n_lam  # current sizes
        self.base_m1, self.base_m2 = base_m1, base_m2
        self.m1, self.m2 = base_m1, base_m2  # current window
        self.nx = nx
        self.int_idx = None
        self.bord_idx = None
        self.E = None
        self.H_lu = None  # E (sparse interior×border), H = C − EᵀM_b⁻¹E
        self.shape = (n_z + n_lam, n_z + n_lam)
        self.build_flops = 0.0  # full spilu cost (set by the factory at base build)
        self.last_build_flops = (
            0.0  # work done on the MOST RECENT build/extend this pass
        )
        self.apply_flops = float(n_z) + 2.0 * float(base_ilu.nnz)

    def _partition(self):
        nx = self.nx
        ndual = self.n_lam // nx
        interior, border = [], []
        for j in range(ndual):
            g = self.m1 + j
            (interior if self.base_m1 <= g <= self.base_m2 else border).extend(
                range(j * nx, (j + 1) * nx)
            )
        return np.array(interior, int), np.array(border, int)

    def extend(self, H_blocks, G, n_z, n_lam, m1, m2):
        """Grow to window [m1,m2]: assemble S, partition, and build the border correction."""
        S, self.Hinv = _build_schur(H_blocks, G)
        self.n_z, self.n_lam, self.m1, self.m2 = n_z, n_lam, m1, m2
        self.shape = (n_z + n_lam, n_z + n_lam)
        ii, bb = self._partition()
        self.int_idx, self.bord_idx = ii, bb
        if bb.size == 0:  # window unchanged from base → plain ILU apply
            self.E = self.H_lu = None
            self.apply_flops = float(n_z) + 2.0 * float(self.base_ilu.nnz)
            self.last_build_flops = 0.0
            return
        Scsr = S.tocsr()
        E = Scsr[ii][:, bb]  # interior × border (SPARSE, kept for apply)
        C = Scsr[bb][:, bb].toarray()  # border × border (small dense)
        Gd = self.base_ilu.solve(
            E.toarray()
        )  # M_b⁻¹E — dense, formed ONCE only to build H
        self.E = E.tocsr()
        self.H_lu = lu_factor(
            C - self.E.T @ Gd
        )  # H = C − Eᵀ M_b⁻¹E  (F = Eᵀ, S symmetric)
        nb = bb.size
        # apply: 2 ILU solves + 2 sparse mat-vecs (E, Eᵀ) + a small H solve
        self.apply_flops = (
            float(n_z)
            + 4.0 * float(self.base_ilu.nnz)
            + 4.0 * float(E.nnz)
            + 2.0 * nb * nb
        )
        # border build this pass: nb ILU solves (G) + small dense H factor — the interior ILU is reused
        self.last_build_flops = (
            2.0 * float(self.base_ilu.nnz) * nb + (2.0 / 3.0) * nb**3
        )

    def __call__(self, r):
        xp = r[: self.n_z] * self.Hinv
        rd = r[self.n_z :]
        if self.bord_idx is None or self.bord_idx.size == 0:
            xd = self.base_ilu.solve(rd)
        else:
            rI = rd[self.int_idx]
            rB = rd[self.bord_idx]
            y = self.base_ilu.solve(rI)  # M_b⁻¹ r_I
            z = lu_solve(self.H_lu, rB - self.E.T @ y)  # H⁻¹(r_B − Eᵀy)
            xd = np.empty_like(rd)
            xd[self.int_idx] = self.base_ilu.solve(rI - self.E @ z)  # M_b⁻¹(r_I − Ez)
            xd[self.bord_idx] = z
        return np.concatenate([xp, xd])

    def matvec(self, r):
        return self(r)

    def as_linop(self):
        n = self.shape[0]
        return spla.LinearOperator((n, n), matvec=self.__call__)


def bordered_ilu_preconditioner(
    H_blocks, G, n_z_local, n_lam_local, m1, m2, nx, drop_tol=1e-2, fill_factor=20
):
    """Full base build: spilu of S over window [m1,m2]; returns a BorderedILUSchur to extend."""
    S, Hinv = _build_schur(H_blocks, G)
    Sreg = S + 1e-12 * sp.eye(S.shape[0], format="csc")
    ilu = spla.spilu(Sreg, drop_tol=drop_tol, fill_factor=fill_factor)
    pc = BorderedILUSchur(ilu, Hinv, n_z_local, n_lam_local, m1, m2, nx)
    pc.build_flops = 2.0 * float(ilu.nnz)  # the one full factorization
    pc.last_build_flops = pc.build_flops  # charged on the base-build pass
    return pc


def spectral_norm_2(Gamma, iters=10, seed=0):
    """‖Γ‖₂ via power iteration on ΓᵀΓ (matches Python reference `spectral_norm`)."""
    rng = np.random.default_rng(seed)
    n = Gamma.shape[1]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = Gamma @ (Gamma @ v)  # Γ symmetric ⇒ ΓᵀΓ v = Γ²v
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        v = w / nw
        lam = np.sqrt(nw)
    return lam


# ====================== GMRES-QLP (ported) ======================
def qlp_min_length(H, c, rank_tol=1e-10):
    """Min-length least-squares of min‖c − H y‖ via QLP (pivoted QR → LQ)."""
    p, m = H.shape
    Q1, R1, piv = qr(H, mode="economic", pivoting=True)
    Q2t, L1t = qr(R1.T, mode="economic")
    L1 = L1t.T
    Q2 = Q2t.T
    dL = np.abs(np.diag(L1))
    r = int(np.sum(dL > rank_tol * max(dL.max(), 1e-300)))
    y = np.zeros(m)
    if r == 0:
        return y, 0
    d = Q1.T @ c
    w = np.zeros(m)
    w[:r] = solve_triangular(L1[:r, :r], d[:r], lower=True)
    y_piv = Q2.T @ w
    y[piv] = y_piv
    return y, r


def gmres_qlp_core(
    matvec, b, rtol=1e-6, maxit=0, rank_tol=1e-10, check_every=5, reorth=True
):
    """Solve matvec(x)=b (possibly nonsymmetric/singular). Returns (x, iters, rel)."""
    b = np.asarray(b, float)
    nrm_b = np.linalg.norm(b)
    nloc = b.shape[0]
    if nrm_b == 0.0:
        return np.zeros(nloc), 0, 0.0
    maxit = maxit if maxit else min(nloc, 1000)
    V = np.zeros((nloc, maxit + 1))
    H = np.zeros((maxit + 1, maxit))
    V[:, 0] = b / nrm_b
    x = np.zeros(nloc)
    rel = 1.0
    for j in range(maxit):
        w = matvec(V[:, j])
        for i in range(j + 1):  # modified Gram-Schmidt
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        if reorth:  # one reorthogonalization pass
            corr = V[:, : j + 1].T @ w
            H[: j + 1, j] += corr
            w = w - V[:, : j + 1] @ corr
        hjp = np.linalg.norm(w)
        H[j + 1, j] = hjp
        m = j + 1
        if (m % check_every == 0) or (hjp <= rank_tol) or (j == maxit - 1):
            Hbar = H[: m + 1, :m]
            rhs = np.zeros(m + 1)
            rhs[0] = nrm_b
            y, _ = qlp_min_length(Hbar, rhs, rank_tol=rank_tol)
            x = V[:, :m] @ y
            rel = np.linalg.norm(Hbar @ y - rhs) / nrm_b
            if rel <= rtol:
                return x, m, rel
        if hjp <= rank_tol:  # happy/invariant breakdown
            return x, m, rel
        V[:, j + 1] = w / hjp
    return x, maxit, rel


# ====================== solver wrappers ======================
def _info(iters, matvecs, fac, hist, cap):
    return dict(
        iters=iters,
        matvecs=matvecs,
        factorizations=fac,
        residual_history=hist,
        cap_hit=cap,
    )


def solve_direct(Gamma, rhs, **kw):
    d = spla.spsolve(sp.csc_matrix(Gamma), rhs)
    return d, _info(1, 0, 1, None, False)


def solve_gmres_qlp(Gamma, rhs, tol, max_iters, M=None, x0=None, rank_tol=1e-12, **kw):
    """Solve the left-preconditioned system with GMRES-QLP.

    Warm-start corrections use a rescaled tolerance for the full residual."""
    if M is None:

        def matvec(v):
            return Gamma @ v

        b = np.asarray(rhs, float)
    else:
        applyP = M.__call__ if hasattr(M, "__call__") else M

        def matvec(v):
            return applyP(Gamma @ v)

        b = applyP(np.asarray(rhs, float))
    if x0 is None or not np.any(x0):
        x, it, rel = gmres_qlp_core(
            matvec, b, rtol=tol, maxit=max_iters, rank_tol=rank_tol, check_every=5
        )
        return x, _info(it, it, 0, None, (it >= max_iters and rel > tol))
    # warm start from x0
    nrm_b = np.linalg.norm(b)
    r0 = b - matvec(x0)  # 1 extra matvec
    nrm_r0 = np.linalg.norm(r0)
    if nrm_b == 0.0 or nrm_r0 <= tol * nrm_b:  # x0 already accurate enough
        return np.asarray(x0, float), _info(0, 1, 0, None, False)
    rtol_eff = min(1.0, tol * nrm_b / nrm_r0)  # so ‖M(Γδ)−r0‖ ≤ tol·‖b‖
    delta, it, rel = gmres_qlp_core(
        matvec, r0, rtol=rtol_eff, maxit=max_iters, rank_tol=rank_tol, check_every=5
    )
    rel_true = rel * nrm_r0 / nrm_b  # ‖M(Γx)−b‖/‖b‖
    return x0 + delta, _info(it, it + 1, 0, None, (it >= max_iters and rel_true > tol))


def solve_minres(Gamma, rhs, tol, max_iters, M=None, **kw):
    Mop = M.as_linop() if isinstance(M, SchurApproxPreconditioner) else M
    it = [0]
    sol, _ = spla.minres(
        Gamma,
        rhs,
        M=Mop,
        rtol=tol,
        maxiter=max_iters,
        callback=lambda xk: it.__setitem__(0, it[0] + 1),
    )
    rel = np.linalg.norm(Gamma @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30)
    return sol, _info(it[0], it[0], 0, None, rel > tol)


def solve_gmres(Gamma, rhs, tol, max_iters, M=None, **kw):
    Mop = M.as_linop() if isinstance(M, SchurApproxPreconditioner) else M
    it = [0]
    sol, _ = spla.gmres(
        Gamma,
        rhs,
        M=Mop,
        rtol=tol,
        maxiter=max_iters,
        callback=lambda xk: it.__setitem__(0, it[0] + 1),
        callback_type="pr_norm",
    )
    rel = np.linalg.norm(Gamma @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30)
    return sol, _info(it[0], it[0], 0, None, rel > tol)


_SKETCH_RNG = np.random.default_rng(0)  # module-global: deterministic given call order


def solve_sketch(Gamma, rhs, tol, max_iters, M=None, x0=None, restart=20, **kw):
    """Minimize a Gaussian-sketched residual over restarted power-basis cycles.

    Uses left preconditioning and warm starts; checks the unsketched residual.
    No certified subspace embedding is claimed. Preconditioner work is added by the caller."""
    applyP = (
        (lambda w: w) if M is None else (M.__call__ if hasattr(M, "__call__") else M)
    )

    def matvec(v):
        return applyP(Gamma @ v)

    b = applyP(np.asarray(rhs, float))
    n = b.shape[0]
    nnz = Gamma.nnz
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        info = _info(0, 0, 0, None, False)
        info["algo_flops"] = 0.0
        return np.zeros(n), info
    d = np.array(x0, float) if (x0 is not None and np.any(x0)) else np.zeros(n)
    mv = 0
    algo_flops = 0.0
    while mv < max_iters:
        r0 = b - matvec(d)
        mv += 1
        rho = np.linalg.norm(r0)
        if rho / bnorm <= tol:
            break
        m = min(restart, max_iters - mv)
        if m < 1:
            break
        V = np.empty((n, m))
        AV = np.empty((n, m))
        v = r0 / rho
        for j in range(m):
            V[:, j] = v
            w = matvec(v)
            mv += 1
            AV[:, j] = w
            v = w / (np.linalg.norm(w) + 1e-30)  # next power-basis vector
        s = min(2 * m + 5, n)
        S = _SKETCH_RNG.standard_normal((s, n)) / np.sqrt(
            s
        )  # Dense Gaussian sketch; no certified embedding guarantee.
        y, *_ = np.linalg.lstsq(S @ AV, S @ r0, rcond=None)  # sketched least-squares
        d = d + V @ y
        algo_flops += 2.0 * s * n * m + 2.0 * s * n + 2.0 * s * m * m + 2.0 * n * m
    algo_flops += 2.0 * mv * nnz  # Γ matvecs (precond apply added by caller)
    rel = (
        np.linalg.norm(b - matvec(d)) / bnorm
    )  # diagnostic (uncounted, as in the Julia ref)
    info = _info(mv, mv, 0, None, rel > tol)
    info["algo_flops"] = algo_flops
    return d, info


SOLVER_DISPATCH = {
    "direct": solve_direct,
    "gmres_qlp": solve_gmres_qlp,
    "minres": solve_minres,
    "minres_qlp": solve_minres,  # scipy has no minres_qlp; minres is the symmetric fallback
    "gmres": solve_gmres,
    "sketch": solve_sketch,  # Custom Gaussian-sketched solver.
}


def solve(method, Gamma, rhs, tol, max_iters, M=None, **kw):
    if method not in SOLVER_DISPATCH:
        raise ValueError(f"unknown solver: {method}; options: {list(SOLVER_DISPATCH)}")
    return SOLVER_DISPATCH[method](Gamma, rhs, tol=tol, max_iters=max_iters, M=M, **kw)

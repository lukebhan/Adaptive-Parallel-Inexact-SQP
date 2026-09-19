"""Adaptive and fixed-overlap SQP loops."""

from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp

from . import burgers_setting as bs
from .ComputeKKT import (
    n_z,
    n_lam,
    n_kkt,
    stage_jacobians,
    constraint_jacobian,
    constraint_residual,
    grad_lagrangian,
    hessian_blocks,
    modify_hessian_blocks,
    assemble_kkt,
    cost,
)
from .CalculateAug import merit, grad_merit_flat
from .backtrack import armijo
from .composition import (
    uniform_knots,
    compose,
    decompose_residual_norms,
    decompose_to_local,
)
from .AOTDNLP import build_subproblems, assemble, n_z_local, n_lam_local
from .AOTDsolver import (
    solve,
    schur_approx_preconditioner,
    ilu_schur_preconditioner,
    bordered_ilu_preconditioner,
    spectral_norm_2,
)
from .flop_model import gmres_qlp_flops, banded_factor_flops


@dataclass
class AlgorithmConfig:
    M: int = 4  # number of subproblems
    mu: float = 10.0  # interface penalty
    beta: float = 0.4  # Armijo / ε_max constant
    eta1_0: float = 10.0
    eta2_0: float = 0.5
    eps0: float = 5e-2  # global ε^τ initial
    eps_i_0: float = 1e-1  # per-subproblem inner tol initial
    eps_i_floor: float = 1e-9
    b0: int = 2  # initial overlap
    max_overlap: int = 45  # b cap (Python: N/(2M)+25)
    max_inner_passes: int = 8
    max_outer_iters: int = 15
    tol_kkt: float = 1e-6
    tol_step: float = 1e-8
    inner_solver: str = "gmres_qlp"
    inner_rank_tol: float = (
        1e-12  # GMRES-QLP numerical-rank / Arnoldi-breakdown threshold
    )
    sketch_restart: int = 5  # Maximum power-basis vectors per cycle.
    use_preconditioner: bool = True
    precond_type: str = (
        "schur_approx"  # "schur_approx" (block-diag) | "ilu_schur" | "exact"
    )
    ilu_drop_tol: float = 1e-2  # ILU-Schur drop tolerance (cost vs tol-sensitivity)
    max_inner_iters: int = 250
    gauss_newton: bool = True
    xi_H: float = 0.0
    warm_start: bool = (
        False  # warm-start each pass's inner solves from the previous pass
    )
    cache_precond: bool = False  # reuse assembled Gi + preconditioner across inner passes while the window (m1,m2) is unchanged
    freeze_precond: bool = (
        False  # build the preconditioner ONCE and reuse it across ALL outer iterations
    )
    # (lagged/frozen preconditioner); rebuilt only if Krylov counts degrade past
    # freeze_rebuild_ratio. Meant for FIXED b (constant window ⇒ constant dim).
    freeze_rebuild_ratio: float = 2.0  # rebuild the frozen preconditioner once iters exceed this × the post-build count
    adaptive: bool = True
    nu: float = 2.0  # adaptation rate ν for η-updates (29)-(30)
    eps_update: str = "geometric"  # global ε^τ forcing rule: "geometric" (Eq.30, ε^τ/ν⁴) | "ew" (Eisenstat-Walker Choice 2)
    ew_gamma: float = 0.9  # EW Choice-2 γ
    ew_alpha: float = 1.618  # EW Choice-2 α (golden ratio)
    varrho: float = 0.01  # adaptation rate ϱ for ε_i/b_i updates (26)-(27); demand-mode default for both rates
    adapt_mode: str = "fixed_step"  # Selects the overlap/tolerance update law.
    kappa: float = 2.0  # "simple" mode: fixed tolerance shrink factor κ>1 (ε_i ← ε_i/κ, b_i ← min(b_i+1, b_max)) ∀i
    varrho_b: float = None  # overlap rate ϱ_b for b_i update (None ⇒ uses varrho); "residual" & "demand" modes
    varrho_eps: float = None  # tolerance rate ϱ_ε for ε_i update (None ⇒ uses varrho); "residual" & "demand" modes
    demand_rho: float = (
        None  # demand-mode only: optional EDS rate ρ to scale the b-map by 1/log(1/ρ)
    )
    b_step: int = 3  # fixed_step overlap increment
    hybrid_b_min1: bool = False  # hybrid mode: floor the b_i increment at 1 (b_i ← b_i + max{1, ⌈ϱ_b‖r_i‖/‖r‖⌉})
    acc_relax: float = (
        1.0  # relaxation κ on accuracy condition (24): ‖r‖≤κ·θ·ε^τ·‖∇L‖/(‖Γ‖Ψ)
    )
    # global ε^τ schedule θ (poly/exp); θ0=θ_min=1 recovers the un-scheduled scheme
    theta0: float = 1.0
    theta_min: float = 1.0
    psi: float = 1.0
    upsilon: float = 1.0
    eps_max_floor: float = 1e-12
    linesearch_max_backtracks: int = 30
    verbose: bool = False


def eps_max_value(eta1, eta2, beta, psi, upsilon, floor):
    return max((0.5 - beta) * eta2 / (1.0 + eta1 + eta2), floor)


def run_algorithm(prob, cfg, z0, lam0):
    z = np.array(z0, float)
    lam = np.array(lam0, float)
    nz = n_z(prob)
    knots = uniform_knots(prob.N, cfg.M)
    b_list = [cfg.b0] * cfg.M
    eps_i_list = [cfg.eps_i_0] * cfg.M
    ell_b = [0.0] * cfg.M
    ell_e = [
        0.0
    ] * cfg.M  # demand-mode accumulators for overlap / tolerance (separate rates)
    eta1, eta2 = cfg.eta1_0, cfg.eta2_0
    eps_g = cfg.eps0
    total_matvecs = 0
    total_inner_iters = 0
    total_flops = 0.0
    round_sub_flops = []  # per-barrier list of per-subproblem FLOPs (throughput-wall model)
    round_sub_times = []  # per-barrier list of per-subproblem wall times (throughput-wall model)
    stop_reason = "max_iters"
    grad_L_norm = np.inf
    prev_grad_L_norm = None  # ‖∇L^{τ-1}‖ for the EW forcing ratio
    tau = 0
    trajectory = []  # per-outer records
    # wall-clock phase accumulators (serial) + critical-path (parallel) sub-solve
    T = dict(
        assemble=0.0,
        subsolve=0.0,
        subsolve_par=0.0,
        compose=0.0,
        checks=0.0,
        linesearch=0.0,
    )

    def vlog(s):
        if cfg.verbose:
            print(s, flush=True)

    _frozen_pc = {}  # cfg.freeze_precond: preconditioner reused ACROSS outers
    _frozen_iters0 = {}  # Krylov count right after each frozen build (staleness gauge)

    for tau in range(cfg.max_outer_iters):
        t_o = time.perf_counter()
        A_list, B_list = stage_jacobians(prob, z)
        G = constraint_jacobian(prob, z, A_list, B_list)
        grad_z, f = grad_lagrangian(prob, z, lam, G=G)
        grad_L = np.concatenate([grad_z, f])
        grad_L_norm = np.linalg.norm(grad_L)
        feas_norm = np.linalg.norm(f)
        stat_norm = np.linalg.norm(grad_z)
        vlog(
            f"[AN] outer τ={tau}  |gL|={grad_L_norm:.6g} feas={feas_norm:.4g} "
            f"eps_g={eps_g:.3g} eta1={eta1:.3g} eta2={eta2:.3g} b={min(b_list)}..{max(b_list)}"
        )
        if grad_L_norm <= cfg.tol_kkt:
            stop_reason = "kkt"
            break

        H_blocks = hessian_blocks(prob, z, lam, gauss_newton=cfg.gauss_newton)
        H_mod, _, _, _ = modify_hessian_blocks(H_blocks, cfg.xi_H)
        H_unmod = sp.block_diag([sp.csr_matrix(Hb) for Hb in H_blocks], format="csr")
        Gamma_global = assemble_kkt(H_mod, G)
        Gamma_norm = spectral_norm_2(Gamma_global)
        T["assemble"] += time.perf_counter() - t_o

        # θ schedule (poly): θ0/(τ+1) if scheduled, else 1
        theta_k = (
            max(cfg.theta_min, cfg.theta0 / (tau + 1.0))
            if cfg.theta0 > cfg.theta_min
            else cfg.theta0
        )
        # per-outer work accumulators (serial sum + critical-path max over subproblems)
        ostat = dict(
            matvecs=0,
            inner_iters=0,
            flops=0.0,
            sub_serial=0.0,
            sub_par=0.0,
            compose=0.0,
            n_pass=0,
        )
        _pc_cache = {}  # per-outer cache of (Gi, rhs_i, precond) keyed on (i, m1, m2)
        _bilu_cache = {}  # per-outer bordered-ILU state per subproblem i

        def solve_all(bvec, epsvec, exact, warm=None):
            nonlocal total_matvecs, total_inner_iters, total_flops
            subs = build_subproblems(prob, knots, bvec, cfg.mu)
            local_dirs = []
            sub_flops = []  # per-subproblem FLOPs for this barrier
            sub_times = []
            for i, sub in enumerate(subs):
                t_s = time.perf_counter()
                key = (i, sub.m1, sub.m2)
                reused = cfg.cache_precond and key in _pc_cache
                pc_build_flop = 0.0  # precond-build work charged THIS pass
                if reused:  # window unchanged since last pass this outer
                    Gi, rhs_i, M = _pc_cache[key]
                else:
                    Gi, rhs_i, lH, lG = assemble(
                        sub, prob, H_mod, A_list, B_list, grad_z, f, lam
                    )
                    M = None
                    _ndof = n_z_local(sub) + n_lam_local(sub)
                    frozen = (
                        cfg.freeze_precond
                        and _frozen_pc.get(i) is not None
                        and _frozen_pc[i].shape[0] == _ndof
                    )  # reuse across outers if dim matches
                    if frozen:
                        M = _frozen_pc[i]  # lagged preconditioner: skip the rebuild
                    elif cfg.use_preconditioner and cfg.inner_solver in (
                        "gmres_qlp",
                        "minres",
                        "minres_qlp",
                        "gmres",
                        "sketch",
                    ):
                        if cfg.precond_type == "ilu_schur":
                            M = ilu_schur_preconditioner(
                                lH,
                                lG,
                                n_z_local(sub),
                                n_lam_local(sub),
                                drop_tol=cfg.ilu_drop_tol,
                            )
                            pc_build_flop = M.build_flops
                        elif cfg.precond_type == "bordered_ilu":
                            # keep the base ILU across passes; border it for the grown endpoints
                            # (rebuild fully only when Krylov counts degrade — safeguard below)
                            entry = _bilu_cache.get(i)
                            if entry is None:  # first pass this outer → full base build
                                M = bordered_ilu_preconditioner(
                                    lH,
                                    lG,
                                    n_z_local(sub),
                                    n_lam_local(sub),
                                    sub.m1,
                                    sub.m2,
                                    bs.N_X,
                                    drop_tol=cfg.ilu_drop_tol,
                                )
                                _bilu_cache[i] = {"pc": M, "base_iters": None}
                                pc_build_flop = M.last_build_flops  # full spilu, once
                            else:
                                M = entry["pc"]
                                if (
                                    sub.m1 < M.base_m1 or sub.m2 > M.base_m2
                                ):  # overlap grew → border correction
                                    M.extend(
                                        lH,
                                        lG,
                                        n_z_local(sub),
                                        n_lam_local(sub),
                                        sub.m1,
                                        sub.m2,
                                    )
                                    pc_build_flop = (
                                        M.last_build_flops
                                    )  # cheap: reuse interior ILU
                        else:
                            M = schur_approx_preconditioner(
                                lH, lG, n_z_local(sub), n_lam_local(sub)
                            )
                            pc_build_flop = M.build_flops
                        if (
                            cfg.freeze_precond and M is not None
                        ):  # store for reuse across future outers
                            _frozen_pc[i] = M
                            _frozen_iters0[i] = None
                    if cfg.cache_precond:
                        _pc_cache[key] = (Gi, rhs_i, M)
                tol = cfg.eps_i_0 if exact else epsvec[i]
                # warm start: project the previous composed direction onto this window
                x0 = (
                    decompose_to_local(warm[0], warm[1], sub.m1, sub.m2)
                    if (cfg.warm_start and warm is not None)
                    else None
                )
                di, info = solve(
                    cfg.inner_solver,
                    Gi,
                    rhs_i,
                    tol=tol,
                    max_iters=cfg.max_inner_iters,
                    M=M,
                    x0=x0,
                    restart=cfg.sketch_restart,
                    rank_tol=cfg.inner_rank_tol,
                )
                # Rebuild reused factors if Krylov iterations exceed 1.7 times the fresh-build count.
                if cfg.freeze_precond and i in _frozen_pc:
                    if _frozen_iters0.get(i) is None:
                        _frozen_iters0[i] = info[
                            "iters"
                        ]  # count right after this (re)build
                    elif _frozen_iters0[i] and info[
                        "iters"
                    ] > cfg.freeze_rebuild_ratio * max(_frozen_iters0[i], 1):
                        del _frozen_pc[i]
                        _frozen_iters0.pop(i, None)  # stale → rebuild next outer
                if cfg.precond_type == "bordered_ilu" and i in _bilu_cache:
                    e = _bilu_cache[i]
                    if e["base_iters"] is None:
                        e["base_iters"] = info["iters"]
                    elif e["base_iters"] and info["iters"] > 1.7 * max(
                        e["base_iters"], 1
                    ):
                        del _bilu_cache[i]
                dt_s = time.perf_counter() - t_s
                sub_times.append(dt_s)
                mv = info["matvecs"]
                total_matvecs += mv
                total_inner_iters += info["iters"]
                ostat["matvecs"] += mv
                ostat["inner_iters"] += info["iters"]
                if info.get("factorizations", 0):  # direct solve
                    flop = banded_factor_flops(Gi.shape[0], 2 * bs.N_X + bs.N_U)
                else:  # gmres_qlp / minres / sketch
                    if (
                        "algo_flops" in info
                    ):  # sketch: measured matvec + sketch/LSQ work
                        flop = info["algo_flops"]
                    else:
                        flop = gmres_qlp_flops(info["iters"], Gi.nnz, Gi.shape[0])
                    if M is not None and hasattr(
                        M, "apply_flops"
                    ):  # precond build (this pass) + apply per iter
                        flop += (0.0 if reused else pc_build_flop) + info[
                            "iters"
                        ] * M.apply_flops
                total_flops += flop
                ostat["flops"] += flop
                sub_flops.append(float(flop))
                nzl = n_z_local(sub)
                local_dirs.append((di[:nzl], di[nzl:]))
            round_sub_flops.append(
                sub_flops
            )  # one barrier = one round of M concurrent solves
            round_sub_times.append(list(sub_times))
            ostat["sub_serial"] += sum(sub_times)
            ostat["sub_par"] += (
                max(sub_times) if sub_times else 0.0
            )  # critical path (parallel)
            ostat["n_pass"] += 1
            t_c = time.perf_counter()
            comp = compose(prob.N, knots, bvec, local_dirs, nz, n_lam(prob))
            ostat["compose"] += time.perf_counter() - t_c
            return comp

        pass_log = []  # per-pass diagnostics (this outer)
        if not cfg.adaptive:
            # FOTD: one exact solve per outer
            dz, dl = solve_all(b_list, eps_i_list, exact=True)
            direction = np.concatenate([dz, dl])
        else:
            # AOTD: accuracy (24) + descent (28) gating
            eps_max = eps_max_value(
                eta1, eta2, cfg.beta, cfg.psi, cfg.upsilon, cfg.eps_max_floor
            )
            if (
                cfg.eps_update == "ew"
                and prev_grad_L_norm is not None
                and prev_grad_L_norm > 0
            ):
                # Eisenstat-Walker Choice 2: forcing ∝ (‖∇L^τ‖/‖∇L^{τ-1}‖)^α — loose when
                # progress is slow (avoid oversolving), tight as ‖∇L‖→0 (→ superlinear).
                ratio = grad_L_norm / prev_grad_L_norm
                ew = cfg.ew_gamma * (ratio**cfg.ew_alpha)
                safe = cfg.ew_gamma * (
                    eps_g**cfg.ew_alpha
                )  # safeguard vs over-fast drop
                if safe > 0.1:
                    ew = max(ew, safe)
                eps_g = min(eps_max, max(ew, cfg.eps_max_floor))
            else:
                eps_g = min(eps_g, eps_max)
            prev_grad_L_norm = grad_L_norm
            direction = np.zeros(n_kkt(prob))
            prev_dir = None  # warm-start source (prev pass's composed dir)
            for _pass in range(cfg.max_inner_passes):
                b_used = list(b_list)
                eps_used = list(eps_i_list)  # (b_i, ε_i) that produce this pass
                mv0 = ostat["matvecs"]
                it0 = ostat["inner_iters"]
                fl0 = ostat["flops"]
                dz, dl = solve_all(b_list, eps_i_list, exact=False, warm=prev_dir)
                direction = np.concatenate([dz, dl])
                prev_dir = (dz, dl)
                r = Gamma_global @ direction + grad_L
                r_norm = np.linalg.norm(r)
                acc_rhs = (
                    cfg.acc_relax
                    * theta_k
                    * eps_g
                    * grad_L_norm
                    / max(Gamma_norm * cfg.psi, 1e-30)
                )
                rec = dict(
                    pass_idx=_pass,
                    r_norm=float(r_norm),
                    acc_rhs=float(acc_rhs),
                    pass_matvecs=ostat["matvecs"] - mv0,
                    pass_inner_iters=ostat["inner_iters"] - it0,
                    pass_flops=float(ostat["flops"] - fl0),
                    b_used=b_used,
                    eps_used=eps_used,
                    eps_g=eps_g,
                )
                if r_norm <= acc_rhs:  # (24) accuracy
                    gflat = grad_merit_flat(prob, z, lam, eta1, eta2, H=H_unmod, G=G)
                    gdir = gflat @ direction
                    descent_rhs = -0.5 * eta2 * grad_L_norm**2
                    rec["gdir"] = float(gdir)
                    rec["descent_rhs"] = float(descent_rhs)
                    if gdir <= descent_rhs:  # (28) descent
                        rec["outcome"] = "accept"
                        pass_log.append(rec)
                        vlog(
                            f"    [pass {_pass}] (24)✓ ‖r‖={r_norm:.3e}≤{acc_rhs:.3e}  "
                            f"(28)✓ ∇φ·d={gdir:.3e}≤{descent_rhs:.3e}  ACCEPT"
                        )
                        break
                    eta1 *= cfg.nu**2
                    eta2 /= cfg.nu
                    eps_g = min(
                        eps_max_value(
                            eta1,
                            eta2,
                            cfg.beta,
                            cfg.psi,
                            cfg.upsilon,
                            cfg.eps_max_floor,
                        ),
                        eps_g / cfg.nu**4,
                    )
                    # per-firing trace: η/ε^τ AFTER this descent-(28) firing (Eq.29-30)
                    rec["eta1"] = float(eta1)
                    rec["eta2"] = float(eta2)
                    rec["eps_g_after"] = float(eps_g)
                    rec["outcome"] = "desc_fail"
                    pass_log.append(rec)
                    vlog(
                        f"    [pass {_pass}] (24)✓ ‖r‖={r_norm:.3e}≤{acc_rhs:.3e}  "
                        f"(28)✗ ∇φ·d={gdir:.3e}>{descent_rhs:.3e}  → η1={eta1:.3g} η2={eta2:.3g} "
                        f"ε^τ={eps_g:.3e}"
                    )
                else:  # (24) failed → adapt (b_i, ε_i)
                    if cfg.adapt_mode == "fixed_step":
                        for i in range(cfg.M):
                            eps_i_list[i] = max(
                                eps_i_list[i] / cfg.nu**3, cfg.eps_i_floor
                            )
                            b_list[i] = min(
                                b_list[i] + cfg.b_step, cfg.max_overlap, prob.N
                            )
                    elif cfg.adapt_mode == "simple":
                        # Uniform adaptation: divide tolerances by kappa and grow overlaps by one.
                        for i in range(cfg.M):
                            eps_i_list[i] = max(
                                eps_i_list[i] / cfg.kappa, cfg.eps_i_floor
                            )
                            b_list[i] = min(b_list[i] + 1, cfg.max_overlap, prob.N)
                    elif cfg.adapt_mode == "fixed":
                        # Apply fixed tolerance and overlap updates to every subproblem.
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        for i in range(cfg.M):
                            eps_i_list[i] = max(
                                eps_i_list[i] * np.exp(-min(rho_e, 700.0)),
                                cfg.eps_i_floor,
                            )
                            b_list[i] = min(
                                b_list[i] + int(round(rho_b)), cfg.max_overlap, prob.N
                            )
                    elif cfg.adapt_mode == "relative":
                        # Adapt only subproblems exceeding acc_rhs / sqrt(M).
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        share = acc_rhs / np.sqrt(cfg.M)
                        rec["loc_resid"] = [float(x) for x in loc]
                        rec["share"] = float(share)
                        for i in range(cfg.M):
                            if loc[i] <= share:  # meets its share → freeze
                                continue
                            eps_i_list[i] = max(
                                eps_i_list[i] / cfg.nu**3, cfg.eps_i_floor
                            )
                            b_list[i] = min(
                                b_list[i] + cfg.b_step, cfg.max_overlap, prob.N
                            )
                    elif cfg.adapt_mode == "demand":
                        # Accumulate log gate violations at separate overlap and tolerance rates.
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        bscale = (
                            (1.0 / np.log(1.0 / cfg.demand_rho))
                            if (cfg.demand_rho and cfg.demand_rho > 0.0)
                            else 1.0
                        )
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        share = acc_rhs / np.sqrt(cfg.M)
                        rec["loc_resid"] = [float(x) for x in loc]
                        rec["share"] = float(share)
                        for i in range(cfg.M):
                            d_i = np.log(
                                max(loc[i] / max(share, 1e-30), 1e-30)
                            )  # signed log-demand
                            ell_b[i] = max(0.0, ell_b[i] + rho_b * d_i)
                            ell_e[i] = max(0.0, ell_e[i] + rho_e * d_i)
                            b_list[i] = min(
                                cfg.b0 + int(np.floor(ell_b[i] * bscale)),
                                cfg.max_overlap,
                                prob.N,
                            )
                            eps_i_list[i] = max(
                                cfg.eps_i_0 * np.exp(-ell_e[i]), cfg.eps_i_floor
                            )
                        rec["ell_b"] = [float(x) for x in ell_b]
                        rec["ell_e"] = [float(x) for x in ell_e]
                    elif cfg.adapt_mode == "residual_ceil":
                        # Ceil residuals so any nonzero residual triggers a full tolerance update.
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        rec["loc_resid"] = [float(x) for x in loc]
                        for i in range(cfg.M):
                            eps_i_list[i] = max(
                                eps_i_list[i]
                                * np.exp(-min(rho_e * np.ceil(loc[i]), 700.0)),
                                cfg.eps_i_floor,
                            )
                            b_list[i] = min(
                                b_list[i] + int(np.ceil(rho_b * loc[i])),
                                cfg.max_overlap,
                                prob.N,
                            )
                    elif cfg.adapt_mode == "hybrid":
                        # Hybrid updates add uniform tightening; hybrid_b_min1 also forces overlap growth.
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        tot = max(r_norm, 1e-30)
                        base = 1.0 / np.sqrt(cfg.M)
                        rec["loc_resid"] = [float(x) for x in loc]
                        for i in range(cfg.M):
                            frac = loc[i] / tot
                            eps_i_list[i] = max(
                                eps_i_list[i]
                                * np.exp(-min(rho_e * (base + frac), 700.0)),
                                cfg.eps_i_floor,
                            )
                            step_b = int(np.ceil(rho_b * frac))
                            if cfg.hybrid_b_min1:
                                step_b = max(1, step_b)
                            b_list[i] = min(b_list[i] + step_b, cfg.max_overlap, prob.N)
                    elif cfg.adapt_mode == "residual_rel":
                        # Scale both updates by the local-to-global residual ratio.
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        tot = max(r_norm, 1e-30)  # ||r|| (global residual this pass)
                        rec["loc_resid"] = [float(x) for x in loc]
                        for i in range(cfg.M):
                            frac = loc[i] / tot
                            eps_i_list[i] = max(
                                eps_i_list[i] * np.exp(-min(rho_e * frac, 700.0)),
                                cfg.eps_i_floor,
                            )
                            b_list[i] = min(
                                b_list[i] + int(np.ceil(rho_b * frac)),
                                cfg.max_overlap,
                                prob.N,
                            )
                    else:  # "residual": residual-scaled, per-subproblem (paper Eqs. 26-27)
                        # Paper update laws with SEPARATELY-TUNABLE rates: ε uses ϱ_ε, b uses ϱ_b
                        # (both default to ϱ). ε_i ← ε_i·exp(−ϱ_ε‖r̃_i‖); b_i ← b_i + ⌈ϱ_b‖r̃_i‖⌉.
                        rho_e = (
                            cfg.varrho_eps if cfg.varrho_eps is not None else cfg.varrho
                        )
                        rho_b = cfg.varrho_b if cfg.varrho_b is not None else cfg.varrho
                        loc = decompose_residual_norms(prob.N, knots, r, nz)
                        rec["loc_resid"] = [float(x) for x in loc]
                        for i in range(cfg.M):
                            eps_i_list[i] = max(
                                eps_i_list[i] * np.exp(-min(rho_e * loc[i], 700.0)),
                                cfg.eps_i_floor,
                            )
                            b_list[i] = min(
                                b_list[i] + int(np.ceil(rho_b * loc[i])),
                                cfg.max_overlap,
                                prob.N,
                            )
                    rec["outcome"] = "acc_fail"
                    pass_log.append(rec)
                    vlog(
                        f"    [pass {_pass}] (24)✗ ‖r‖={r_norm:.3e}>{acc_rhs:.3e}  → adapt "
                        f"b={min(b_list)}..{max(b_list)} ε_i={min(eps_i_list):.2e}..{max(eps_i_list):.2e}"
                    )

        # augmented-Lagrangian line search
        t_ls = time.perf_counter()
        cur_merit = merit(prob, z, lam, eta1, eta2, G=G)
        gflat = grad_merit_flat(prob, z, lam, eta1, eta2, H=H_unmod, G=G)
        grad_dot_dir = gflat @ direction
        alpha, nbt, accepted = armijo(
            prob,
            z,
            lam,
            direction,
            eta1,
            eta2,
            cfg.beta,
            max_backtracks=cfg.linesearch_max_backtracks,
            current_merit=cur_merit,
            current_grad_dot_dir=grad_dot_dir,
        )
        z = z + alpha * direction[:nz]
        lam = lam + alpha * direction[nz:]
        step_norm = alpha * np.linalg.norm(direction)
        dt_ls = time.perf_counter() - t_ls
        T["subsolve"] += ostat["sub_serial"]
        T["subsolve_par"] += ostat["sub_par"]
        T["compose"] += ostat["compose"]
        T["linesearch"] += dt_ls
        trajectory.append(
            dict(
                tau=tau,
                grad_L_norm=grad_L_norm,
                feas=feas_norm,
                stationarity=stat_norm,
                cost=cost(prob, z),
                eta1=eta1,
                eta2=eta2,
                eps_g=eps_g,
                theta_k=theta_k,
                b_min=min(b_list),
                b_max=max(b_list),
                b_mean=float(np.mean(b_list)),
                eps_i_min=min(eps_i_list),
                eps_i_max=max(eps_i_list),
                eps_i_mean=float(np.mean(eps_i_list)),
                b_list=list(b_list),
                eps_i_list=list(eps_i_list),
                passes=pass_log,
                n_acc_fail=sum(1 for p in pass_log if p["outcome"] == "acc_fail"),
                n_desc_fail=sum(1 for p in pass_log if p["outcome"] == "desc_fail"),
                alpha=alpha,
                n_backtracks=nbt,
                accepted=accepted,
                n_inner_passes=ostat["n_pass"],
                inner_iters=ostat["inner_iters"],
                matvecs=ostat["matvecs"],
                flops=ostat["flops"],
                t_subsolve=ostat["sub_serial"],
                t_subsolve_par=ostat["sub_par"],
                t_compose=ostat["compose"],
                t_linesearch=dt_ls,
                t_outer=time.perf_counter() - t_o,
            )
        )
        vlog(
            f"[AN] outer τ={tau} done  α={alpha:.3g} ‖step‖={step_norm:.4g} accepted={accepted}"
        )
        if step_norm <= cfg.tol_step:
            stop_reason = "step"
            tau += 1
            break

    # critical-path (parallel) wall = serial total minus the serialized sub-solve, plus
    # the per-outer max-over-subproblems (the M subproblems run concurrently).
    parallel_subsolve = T["subsolve_par"]
    return {
        "stop_reason": stop_reason,
        "converged": stop_reason == "kkt",
        "outer_iters": tau + 1,
        "total_matvecs": total_matvecs,
        "total_inner_iters": total_inner_iters,
        "total_flops": total_flops,
        "final_grad_L_norm": grad_L_norm,
        "final_feas": float(np.linalg.norm(constraint_residual(prob, z))),
        "final_stationarity": float(np.linalg.norm(grad_lagrangian(prob, z, lam)[0])),
        "final_cost": cost(prob, z),
        "eta1": eta1,
        "eta2": eta2,
        "eps_g": eps_g,
        "t_assemble": T["assemble"],
        "t_subsolve_serial": T["subsolve"],
        "t_subsolve_parallel": parallel_subsolve,
        "t_compose": T["compose"],
        "t_linesearch": T["linesearch"],
        "z": z,
        "lam": lam,
        "b_list": b_list,
        "eps_i_list": eps_i_list,
        "trajectory": trajectory,
        "round_sub_flops": round_sub_flops,
        "round_sub_times": round_sub_times,
    }

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
    max_overlap: int = 45  # Maximum allowed overlap.
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
    xi_H: float = 1e-6  # Positive stagewise eigenvalue floor for the direction model.
    warm_start: bool = (
        False  # warm-start each pass's inner solves from the previous pass
    )
    cache_precond: bool = False  # reuse assembled Gi + preconditioner across inner passes while the window (m1,m2) is unchanged
    freeze_precond: bool = (
        False  # Reuse factors across outer iterations while their dimensions match.
    )
    # Rebuild frozen factors when Krylov counts exceed
    # freeze_rebuild_ratio times the fresh-build count.
    freeze_rebuild_ratio: float = 2.0  # rebuild the frozen preconditioner once iters exceed this × the post-build count
    adaptive: bool = True
    nu: float = 2.0  # adaptation rate ν for η-updates (29)-(30)
    varrho_b: float = 32.0
    varrho_eps: float = 0.5
    psi: float = 1.0
    upsilon: float = 1.0
    linesearch_max_backtracks: int = 30
    verbose: bool = False

    def __post_init__(self):
        for name in (
            "M",
            "max_inner_passes",
            "max_outer_iters",
            "max_inner_iters",
            "max_overlap",
            "sketch_restart",
            "linesearch_max_backtracks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "mu",
            "eta1_0",
            "eta2_0",
            "eps0",
            "eps_i_0",
            "eps_i_floor",
            "xi_H",
            "varrho_b",
            "varrho_eps",
            "psi",
            "upsilon",
            "inner_rank_tol",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.beta < 0.5 or not np.isfinite(self.nu) or self.nu <= 1:
            raise ValueError("Require 0 < beta < .5 and nu > 1")
        if not isinstance(self.b0, int) or not 0 <= self.b0 <= self.max_overlap:
            raise ValueError("Require integer 0 <= b0 <= max_overlap")


def eps_max_value(eta1, eta2, beta, psi, upsilon):
    # Do not enlarge this bound with a numerical floor.
    return (0.5 - beta) * eta2 / ((1.0 + eta1 + eta2) * (psi * upsilon) ** 2)


def run_algorithm(prob, cfg, z0, lam0):
    if cfg.max_inner_passes < 1 or cfg.max_inner_iters < 1:
        raise ValueError("Inner pass and solver budgets must be positive")
    z = np.array(z0, float)
    lam = np.array(lam0, float)
    nz = n_z(prob)
    knots = uniform_knots(prob.N, cfg.M)
    b_list = [cfg.b0] * cfg.M
    eps_i_list = [cfg.eps_i_0] * cfg.M
    eta1, eta2 = cfg.eta1_0, cfg.eta2_0
    eps_g = cfg.eps0
    total_matvecs = 0
    total_inner_iters = 0
    total_flops = 0.0
    round_sub_flops = []  # per-barrier list of per-subproblem FLOPs (throughput-wall model)
    round_sub_times = []  # per-barrier list of per-subproblem wall times (throughput-wall model)
    stop_reason = "max_iters"
    grad_L_norm = np.inf
    tau = 0
    trajectory = []
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

    _frozen_pc = {}  # Frozen preconditioners persist across outer iterations.
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

        H_true_blocks = hessian_blocks(prob, z, lam, gauss_newton=False)
        H_true = sp.block_diag(H_true_blocks, format="csr")
        H_mod, hessian_shifts, hessian_nmods, hessian_max_shift = modify_hessian_blocks(
            H_true_blocks, cfg.xi_H
        )
        Gamma_global = assemble_kkt(H_mod, G)
        Gamma_norm = spectral_norm_2(Gamma_global)
        T["assemble"] += time.perf_counter() - t_o

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
        local_checks = []

        def solve_all(bvec, epsvec, exact, warm=None):
            nonlocal total_matvecs, total_inner_iters, total_flops, local_checks
            local_checks = []
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
                        M = _frozen_pc[i]
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
                            if entry is None:
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
                                pc_build_flop = M.last_build_flops
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
                                    pc_build_flop = M.last_build_flops
                        else:
                            M = schur_approx_preconditioner(
                                lH, lG, n_z_local(sub), n_lam_local(sub)
                            )
                            pc_build_flop = M.build_flops
                        if cfg.freeze_precond and M is not None:
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
                local_checks.append(
                    dict(
                        subproblem=i,
                        tolerance=float(tol),
                        residual_norm=info["residual_norm"],
                        rhs_norm=info["rhs_norm"],
                        residual_threshold=info["residual_threshold"],
                        relative_residual=info["relative_residual"],
                        converged=info["converged"],
                        cap_hit=info["cap_hit"],
                        iters=info["iters"],
                        matvecs=info["matvecs"],
                    )
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
                if info.get("factorizations", 0):
                    flop = banded_factor_flops(Gi.shape[0], 2 * bs.N_X + bs.N_U)
                    flop += 2.0 * info["matvecs"] * Gi.nnz
                else:
                    if (
                        "algo_flops" in info
                    ):  # sketch: measured matvec + sketch/LSQ work
                        flop = info["algo_flops"]
                    else:
                        flop = gmres_qlp_flops(info["iters"], Gi.nnz, Gi.shape[0])
                    if M is not None and hasattr(
                        M, "apply_flops"
                    ):  # precond build (this pass) + apply per iter
                        flop += (0.0 if reused else pc_build_flop) + info.get(
                            "preconditioner_applications", info["iters"]
                        ) * M.apply_flops
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

        pass_log = []
        if not cfg.adaptive:
            # FOTD: one exact solve per outer
            dz, dl = solve_all(b_list, eps_i_list, exact=True)
            direction = np.concatenate([dz, dl])
            pass_log.append(
                dict(
                    pass_idx=0,
                    b_used=list(b_list),
                    eps_used=list(eps_i_list),
                    local_solves=local_checks,
                    outcome="fixed_solve"
                    if all(c["converged"] for c in local_checks)
                    else "local_fail",
                    pass_flops=float(ostat["flops"]),
                    pass_matvecs=ostat["matvecs"],
                    pass_inner_iters=ostat["inner_iters"],
                )
            )
        else:
            # AOTD: accuracy (24) + descent (28) gating
            eps_max = eps_max_value(eta1, eta2, cfg.beta, cfg.psi, cfg.upsilon)
            # Algorithm 1: only an epsilon above the cap is reset, with
            # the strict-interior margin 1/nu. Do not shrink it every outer
            # iteration, or alter an epsilon already at/below the cap.
            if eps_g > eps_max:
                eps_before = eps_g
                eps_g = eps_max / cfg.nu
                vlog(
                    f"    [clip] eps_g={eps_before:.6g} > cap={eps_max:.6g} "
                    f"→ eps_g=cap/nu={eps_g:.6g}"
                )
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
                acc_rhs = eps_g * grad_L_norm / max(Gamma_norm * cfg.psi, 1e-30)
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
                    local_solves=local_checks,
                )
                if not all(c["converged"] for c in local_checks):
                    for check in local_checks:
                        if not check["converged"]:
                            vlog(
                                f"    [pass {_pass}] local accuracy ✗ subproblem={check['subproblem']} "
                                f"residual={check['residual_norm']:.3e}>"
                                f"{check['residual_threshold']:.3e} STOP"
                            )
                    rec["outcome"] = "local_fail"
                    pass_log.append(rec)
                    break
                if r_norm <= acc_rhs:  # (24) accuracy
                    gflat = grad_merit_flat(prob, z, lam, eta1, eta2, H=H_true, G=G)
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
                    loc = decompose_residual_norms(prob.N, knots, r, nz)
                    rec["loc_resid"] = [float(x) for x in loc]
                    for i in range(cfg.M):
                        frac = loc[i] / max(r_norm, 1e-30)
                        eps_i_list[i] = max(
                            eps_i_list[i]
                            * np.exp(
                                -min(
                                    cfg.varrho_eps * (1 / np.sqrt(cfg.M) + frac), 700.0
                                )
                            ),
                            cfg.eps_i_floor,
                        )
                        step_b = max(1, int(np.ceil(cfg.varrho_b * frac)))
                        b_list[i] = min(b_list[i] + step_b, cfg.max_overlap, prob.N)
                    rec["outcome"] = "acc_fail"
                    pass_log.append(rec)
                    vlog(
                        f"    [pass {_pass}] (24)✗ ‖r‖={r_norm:.3e}>{acc_rhs:.3e}  → adapt "
                        f"b={min(b_list)}..{max(b_list)} ε_i={min(eps_i_list):.2e}..{max(eps_i_list):.2e}"
                    )

        if not cfg.adaptive:
            failures = [c for c in local_checks if not c["converged"]]
            vlog(
                f"    [fixed solve] local accuracy {'FAIL' if failures else 'PASS'} "
                f"failed_subproblems={len(failures)}"
            )
        # A failed budget is a failed solve, never permission to apply a step.
        failure_reason = None
        if pass_log[-1]["outcome"] == "local_fail":
            failure_reason = "local_accuracy"
        elif cfg.adaptive and pass_log[-1]["outcome"] != "accept":
            failure_reason = "accuracy_budget"

        # augmented-Lagrangian line search, using the exact merit derivative
        t_ls = time.perf_counter()
        cur_merit = merit(prob, z, lam, eta1, eta2, G=G)
        gflat = grad_merit_flat(prob, z, lam, eta1, eta2, H=H_true, G=G)
        grad_dot_dir = gflat @ direction
        if failure_reason is None:
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
            if not accepted:
                failure_reason = "line_search"
                alpha = 0.0
        else:
            alpha, nbt, accepted = 0.0, 0, False
        if accepted:
            z = z + alpha * direction[:nz]
            lam = lam + alpha * direction[nz:]
        step_norm = alpha * np.linalg.norm(direction) if accepted else 0.0
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
                hessian_model="full_lagrangian",
                hessian_shift_max=hessian_max_shift,
                hessian_shifted_blocks=hessian_nmods,
                hessian_shifts=hessian_shifts,
                merit_slope=float(grad_dot_dir),
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
                step_applied=accepted,
                failure_reason=failure_reason,
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
        if failure_reason is not None:
            vlog(f"[AN] STOP {failure_reason}; no step applied")
            stop_reason = failure_reason
            break
        if step_norm <= cfg.tol_step:
            stop_reason = "step"
            tau += 1
            break

    # critical-path (parallel) wall = serial total minus the serialized sub-solve, plus
    # the per-outer max-over-subproblems (the M subproblems run concurrently).
    parallel_subsolve = T["subsolve_par"]
    final_gz, final_f = grad_lagrangian(prob, z, lam)
    grad_L_norm = float(np.linalg.norm(np.concatenate([final_gz, final_f])))
    if stop_reason in ("max_iters", "step") and grad_L_norm <= cfg.tol_kkt:
        stop_reason = "kkt"
    return {
        "stop_reason": stop_reason,
        "converged": stop_reason == "kkt",
        "outer_iters": tau + 1,
        "total_matvecs": total_matvecs,
        "total_inner_iters": total_inner_iters,
        "total_flops": total_flops,
        "final_grad_L_norm": grad_L_norm,
        "final_feas": float(np.linalg.norm(final_f)),
        "final_stationarity": float(np.linalg.norm(final_gz)),
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

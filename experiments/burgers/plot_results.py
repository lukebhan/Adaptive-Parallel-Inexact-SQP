#!/usr/bin/env python3
"""Render Burgers convergence, adaptation, and comparison tables."""

import os
from pathlib import Path
import json
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "burgers"))
)
os.makedirs(WORK, exist_ok=True)
REC = os.path.join(WORK, "records")
N_ROWS = [10000, 50000]
M_COLS = [10, 50, 250]
CONV = 1e-6
ASO, LUC, QLPC, PUR = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#7E2F8E",
)  # AOTD / FOTD-LU / FOTD-QLP / Schwarz
MCOL = {10: "#d62728", 50: "#2ca02c", 250: "#1f77b4"}
R = [json.loads(Path(f).read_text()) for f in glob.glob(os.path.join(REC, "*.json"))]
R = [r for r in R if "error" not in r]
# Schwarz records exist only at N=10000; align their initial residual with AOTD.
for r in R:
    r["N"], r["M"] = int(r["N"]), int(r["M"])

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 9,
    }
)


def set_size(width, subplots=(1, 1)):
    g = (5**0.5 - 1) / 2
    w = width / 72.27
    return (w, w * g * (subplots[0] / subplots[1]))


def _sci_label(v, pos=None):
    if v <= 0:
        return ""
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10.0**e
    mr = round(m)
    if abs(m - mr) < 1e-6:
        return rf"$10^{{{e}}}$" if mr == 1 else rf"${mr}{{\times}}10^{{{e}}}$"
    return rf"${m:.1f}{{\times}}10^{{{e}}}$"


def log_yticks_min4(ax, target=4):
    """Place at least target labeled log ticks, including sub-decade values."""
    lo, hi = ax.get_ylim()
    dlo, dhi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    for subs in [(1, 2, 3, 5), (1, 2, 4, 6, 8), (1, 2, 3, 4, 5, 6, 7, 8, 9)]:
        ticks = sorted(
            m * 10.0**d
            for d in range(dlo, dhi + 1)
            for m in subs
            if lo * (1 - 1e-9) <= m * 10.0**d <= hi * (1 + 1e-9)
        )
        if len(ticks) >= target:
            break
    ax.set_yticks(ticks)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_sci_label))


def runs(label, N, M):
    return [r for r in R if r["method_label"] == label and r["N"] == N and r["M"] == M]


def _pad(cs):
    L = max(len(c) for c in cs)
    return np.array([list(c) + [c[-1]] * (L - len(c)) for c in cs])


def conv_curve(r):
    if (
        r.get("family") == "baseline"
    ):  # Schwarz: prepend matching AOTD init for x=0 alignment
        a = [x for x in runs("AOTD-rebuild", r["N"], r["M"]) if x["seed"] == r["seed"]]
        init = (
            float(a[0]["trajectory"][0]["grad_L_norm"]) if a else float(r["history"][0])
        )
        return [init] + [float(x) for x in r["history"]]
    return [float(t["grad_L_norm"]) for t in r["trajectory"]] + [float(r["kkt"])]


def conv_agg(label, N, M):
    rs = runs(label, N, M)
    if not rs:
        return None
    curves = [conv_curve(r) for r in rs]
    A = _pad(curves)
    lA = np.log10(np.clip(A, 1e-30, None))
    med = 10 ** lA.mean(axis=0)
    ls = lA.std(axis=0)
    lo, hi = 10 ** (np.log10(med) - ls), 10 ** (np.log10(med) + ls)
    hit = np.where(med <= CONV)[0]
    k = hit[0] + 1 if len(hit) else len(med)
    return med[:k], lo[:k], hi[:k]


def adapt_passes(N, M):
    rs = runs("AOTD-rebuild", N, M)
    Bc, Ec = [], []
    for r in rs:
        P = [p for t in r["trajectory"] for p in t["passes"]]
        if not P:
            continue
        Bc.append(np.array([max(p["b_used"]) for p in P], float))
        Ec.append(np.clip([min(p["eps_used"]) for p in P], 1e-9, None))
    if not Bc:
        return None
    B = _pad(Bc)
    E = _pad(Ec)
    lE = np.log10(E)
    bm, bsd = B.mean(0), B.std(0)
    em, esd = 10 ** lE.mean(0), lE.std(0)
    return (
        bm,
        np.clip(bm - bsd, 0, None),
        bm + bsd,
        em,
        10 ** (np.log10(em) - esd),
        10 ** (np.log10(em) + esd),
    )


def flops_stats(label, N, M):
    rs = runs(label, N, M)
    conv = [r["flops"] for r in rs if r["kkt"] < CONV]
    return (
        (np.mean(conv), np.std(conv), len(conv) < len(rs))
        if conv
        else (np.nan, 0.0, True)
    )


# convergence (2x3)
def fig_convergence():
    w, h = set_size(505, subplots=(1, 3))
    fig, ax = plt.subplots(2, 3, figsize=(w, h * 1.7 * 2), sharey=True)
    # Schwarz b=40 only has N=10k records -> conv_agg returns None at N=50k and it is silently skipped.
    series = [
        ("AOTD-rebuild", "AOTD", ASO, "-", 4),
        ("FOTD-LU-b40", r"FOTD (LU) $b{=}40$", LUC, "--", 3),
        ("FOTD-LU-b10", r"FOTD (LU) $b{=}10$", QLPC, "--", 2),
        ("Schwarz-b40", r"Schwarz $b{=}40$", PUR, "-.", 1),
    ]
    for i, N in enumerate(N_ROWS):
        for a, M in zip(ax[i], M_COLS):
            for label, lab, c, ls, z in series:
                agg = conv_agg(label, N, M)
                if agg is None:
                    continue
                m, lo, hi = agg
                x = np.arange(len(m))
                a.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0, zorder=z - 0.5)
                a.semilogy(x, m, ls, color=c, lw=1.5, label=lab, zorder=z)
            a.axhline(CONV, color="k", ls=":", lw=0.9)
            a.set_title(rf"$N{{=}}{N:,}$, $M={M}$")
            a.minorticks_off()
            a.grid(alpha=0.3, which="major", lw=0.4)
            a.set_xlabel(r"outer iteration $\tau$")
        ax[i, 0].set_ylabel(r"KKT residual $\|\nabla\mathcal{L}\|$")
    hh, ll = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        hh,
        ll,
        loc="lower center",
        ncol=4,
        frameon=False,
        labelcolor="black",
        handlelength=2.0,
        bbox_to_anchor=(0.5, -0.03),
        columnspacing=1.6,
    )
    p = os.path.join(WORK, "burgers_convergence.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote ->", os.path.basename(p))


# adaptation (2x3: rows=N)
def fig_adaptation():
    w, h = set_size(505, subplots=(1, 3))
    fig, ax = plt.subplots(2, 3, figsize=(w, h * 1.7 * 2))
    fseries = [
        ("AOTD-rebuild", "AOTD", ASO, "o"),
        ("FOTD-LU-b40", r"FOTD-LU $b{=}40$", LUC, "s"),
        ("FOTD-QLP-b40", r"FOTD-QLP $b{=}40$", QLPC, "^"),
        ("Schwarz-b40", r"Schwarz $b{=}40$", PUR, "v"),
    ]
    for i, N in enumerate(N_ROWS):
        ab, ae, af = ax[i]
        for M in M_COLS:
            ap = adapt_passes(N, M)
            if ap is None:
                continue
            bm, blo, bhi, em, elo, ehi = ap
            x = np.arange(len(bm))
            ab.fill_between(x, blo, bhi, color=MCOL[M], alpha=0.10, lw=0, zorder=1)
            ab.plot(x, bm, "-", color=MCOL[M], lw=1.8, label=rf"$M={M}$", zorder=3)
            ae.fill_between(x, elo, ehi, color=MCOL[M], alpha=0.10, lw=0, zorder=1)
            ae.semilogy(x, em, "-", color=MCOL[M], lw=1.8, zorder=3)
        ab.set_ylabel(r"overlap $\max_i b_i$")
        ab.set_ylim(0, None)
        ab.set_title(rf"({'abcdef'[3 * i + 0]}) overlap adaptation ($N{{=}}{N:,}$)")
        ab.grid(alpha=0.3, lw=0.4)
        ab.legend(frameon=False, loc="upper left", labelcolor="black")
        ae.set_ylabel(r"inner tol $\min_i\varepsilon_i$")
        ae.set_title(rf"({'abcdef'[3 * i + 1]}) tolerance adaptation ($N{{=}}{N:,}$)")
        ae.grid(alpha=0.3, which="both", lw=0.4)
        log_yticks_min4(ae)

        # Group methods by M; whiskers span all five seeds.
        def conv_flops(label, M):
            return [r["flops"] for r in runs(label, N, M) if r["kkt"] < CONV]

        present = [s for s in fseries if any(conv_flops(s[0], M) for M in M_COLS)]
        n = len(present)
        span = 0.66
        centers = np.linspace(-span / 2, span / 2, n) if n > 1 else np.array([0.0])
        bw = 0.78 * span / (n - 1) if n > 1 else 0.3
        for gi in range(
            len(M_COLS)
        ):  # alternating bands => each M group reads as one cluster
            if gi % 2:
                af.axvspan(gi - 0.5, gi + 0.5, color="0.5", alpha=0.08, lw=0, zorder=0)
        for (label, lab, c, mk), dx in zip(present, centers):
            for gi, M in enumerate(M_COLS):
                vals = conv_flops(label, M)
                if not vals:
                    continue
                bp = af.boxplot(
                    vals,
                    positions=[gi + dx],
                    widths=bw,
                    whis=(0, 100),
                    patch_artist=True,
                    manage_ticks=False,
                    showfliers=False,
                    zorder=3,
                )
                for b in bp["boxes"]:
                    b.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.0)
                for ln in bp["whiskers"] + bp["caps"]:
                    ln.set(color=c, lw=1.0)
                for md in bp["medians"]:
                    md.set(color=c, lw=1.7)
        af.set_yscale("log")
        af.set_xticks(range(len(M_COLS)))
        af.set_xticklabels(M_COLS)
        af.set_xlim(-0.6, len(M_COLS) - 0.4)
        af.set_ylabel(r"FLOPs")
        af.set_title(rf"({'abcdef'[3 * i + 2]}) comp.\ cost vs $M$ ($N{{=}}{N:,}$)")
        af.grid(alpha=0.3, which="both", axis="y", lw=0.4)
        log_yticks_min4(
            af
        )  # per-panel legend removed -> single shared legend below the column
        for a in (ab, ae, af):
            a.set_xlabel(r"inner iteration (pass)" if a is not af else r"$M$")
    fig.tight_layout()
    # one shared box legend centred under the (c)/(f) cost column (below panel (f)'s "M" label)
    box_leg = [
        ("AOTD", ASO),
        ("FOTD (LU)", LUC),
        ("FOTD (GMRES)", QLPC),
        ("Schwarz", PUR),
    ]
    hs = [mpatches.Patch(facecolor=cc, alpha=0.35, edgecolor=cc) for _, cc in box_leg]
    pos = af.get_position()
    fig.legend(
        hs,
        [ll for ll, _ in box_leg],
        loc="upper center",
        bbox_to_anchor=(pos.x0 + pos.width / 2, pos.y0 - 0.062),
        ncol=2,
        frameon=False,
        labelcolor="black",
        handlelength=1.3,
        columnspacing=1.4,
        fontsize=8,
    )
    p = os.path.join(WORK, "burgers_adaptation.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote ->", os.path.basename(p))


# tables
def _pwall(x):
    """Estimate parallel wall time: serial time minus summed solves plus critical-path solves."""
    if x.get("parallel_wall_contended") is not None:
        return float(x["parallel_wall_contended"])
    tr = x.get("trajectory")
    if tr:
        return float(
            sum(t["t_outer"] - t["t_subsolve"] + t["t_subsolve_par"] for t in tr)
        )
    return float("nan")


def cell(label, N, M):
    v = runs(label, N, M)
    if not v:
        return None
    kkt = float(np.mean([x["kkt"] for x in v]))
    pw = [_pwall(x) for x in v]
    return dict(
        kkt=float(np.mean([x["kkt"] / 1e-7 for x in v])),
        flops=float(np.mean([x["flops"] for x in v])),
        flops_sd=float(np.std([x["flops"] for x in v])),
        pwall=float(np.nanmean(pw)),
        pwall_sd=float(np.nanstd(pw)),
        outer=float(np.mean([x["outer_iters"] for x in v])),
        inner=float(np.mean([x["inner_iters"] for x in v])),
        conv=kkt < CONV,
    )


# rows: (label_tex, method_label, kind); Schwarz (N=10k only), LU, QLP, then AOTD
ROWS = [
    ("Schwarz", None, "head"),
    (r"\quad $b{=}40$", "Schwarz-b40", "row"),
    ("FOTD (LU)", None, "head"),
    (r"\quad $b{=}10$", "FOTD-LU-b10", "row"),
    (r"\quad $b{=}40$", "FOTD-LU-b40", "row"),
    ("FOTD (GMRES)", None, "head"),
    (r"\quad $b{=}10$", "FOTD-QLP-b10", "row"),
    (r"\quad $b{=}40$", "FOTD-QLP-b40", "row"),
    ("AOTD (GMRES)", "AOTD-rebuild", "mcol"),
    ("AOTD (sketch GMRES)", "AOTD-sketch", "mcol"),
]


def _has(label, N):
    return label is not None and any(runs(label, N, M) for M in M_COLS)


def block(N, banner, value_fmt, bestkey, show_kkt=True):
    # drop a head + its rows when the whole group has no data at this N (e.g. Schwarz at N=50k)
    skip = [False] * len(ROWS)
    for i, spec in enumerate(ROWS):
        if spec[2] == "head":
            j = i + 1
            grp = []
            while j < len(ROWS) and ROWS[j][2] == "row":
                grp.append(ROWS[j])
                j += 1
            skip[i] = not any(_has(s[1], N) for s in grp)
        elif not _has(spec[1], N):
            skip[i] = True
    specs = [s for s, sk in zip(ROWS, skip) if not sk]
    data = [
        (spec, None if spec[2] == "head" else {M: cell(spec[1], N, M) for M in M_COLS})
        for spec in specs
    ]
    best = {}
    for M in M_COLS:
        vals = [d[M][bestkey] for spec, d in data if d and d[M] and d[M]["conv"]]
        best[M] = min(vals) if vals else None
    ncol = 9 if show_kkt else 6  # data columns (3 groups w/ KKT, else 2)
    out = [rf"\multicolumn{{{ncol + 1}}}{{l}}{{\emph{{{banner}}}}} \\", r"\midrule"]
    for spec, d in data:  # no inter-method rules (heads alone separate the groups)
        label = spec[0]
        if d is None:
            out.append(label + " & " + " & ".join([""] * ncol) + r" \\")
            continue
        vv = [value_fmt(d[M], M, best) for M in M_COLS]
        it = [
            (
                NA
                if (d[M] is None or not d[M]["conv"])
                else f"{d[M]['outer']:.0f}/{d[M]['inner']:.0f}"
            )
            for M in M_COLS
        ]
        kk = [
            (NA if (d[M] is None or not d[M]["conv"]) else f"{d[M]['kkt']:.2f}")
            for M in M_COLS
        ]
        cols = (kk + vv + it) if show_kkt else (vv + it)
        out.append(f"{label} & " + " & ".join(cols) + r" \\")
    return out


NA = "--"
THICKRULE = r"\midrule[\heavyrulewidth]"  # heavy separator between the two horizons


def fmt_flops(c, M, best):
    if c is None or not c["conv"]:
        return NA
    s = rf"${c['flops'] / 1e9:.1f}{{\pm}}{c['flops_sd'] / 1e9:.1f}$"
    return (
        rf"{{\boldmath {s}}}"
        if best[M] is not None and abs(c["flops"] - best[M]) < 1e-9 * best[M]
        else s
    )


def fmt_pwall(c, M, best):
    if c is None or not c["conv"]:
        return NA
    s = rf"${c['pwall']:.1f}{{\pm}}{c['pwall_sd']:.1f}$"
    return (
        rf"{{\boldmath {s}}}"
        if best[M] is not None and abs(c["pwall"] - best[M]) < 1e-6 * best[M]
        else s
    )


def write_highlight():
    CAP = (
        r"Viscous Burgers OCP under the \emph{tight-rank/$\varepsilon$} sweep (exp10): AOTD (adaptive "
        r"inexact, rebuilt ILU--Schur) vs FOTD with the local Newton system solved to high accuracy two "
        r"ways -- \textbf{FOTD-LU} (direct sparse factorization) and \textbf{FOTD-QLP} (GMRES--QLP; "
        r"$\varepsilon_i{=}$rank-tol${=}10^{-14}$, inner iterations capped at 100 -- the cap bounds the "
        r"$\mathcal{O}(K^2)$ null-space chase and is consistent across $N/M$). "
        r"KKT residual in units of $10^{-7}$ (converged iff $<\!10$); FLOPs in $10^{9}$, mean$\pm$std over "
        r"5 IC-noise seeds under the FLOP model that counts the Schur-complement formation; "
        r"out/in $=$ outer / total-inner iterations. Columns are subproblem count $M$ (window $N/M$); "
        r"FOTD at fixed overlap $b$, AOTD adapts $b_i$. Per column/horizon the best converged KKT and "
        r"FLOPs are bold. FOTD-QLP $b{=}10$ omitted at $M{=}250$ (does not converge). Overlapping "
        r"Schwarz ($b{=}40$) is included at $N{=}10{,}000$ as a domain-decomposition baseline. "
        r"\emph{AOTD (sketch)} replaces the inner GMRES--QLP solve with randomized sketched-GMRES "
        r"(Nakatsukasa--Tropp) at otherwise identical settings -- a solver-independence check."
    )
    L = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\caption{" + CAP + r"}",
        r"\label{tab:exp10_highlight}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|rrr|rrr|rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{KKT residual ($\times 10^{-7}$)} & \multicolumn{3}{c|}{FLOPs ($\times 10^{9}$)} "
        r"& \multicolumn{3}{c}{iters (out/in)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        (r"Method" + r" & $M{=}10$ & $M{=}50$ & $M{=}250$" * 3 + r" \\"),
        r"\midrule",
    ]
    L += block(
        10000, r"$N = 10{,}000$ \quad (windows $1000/200/40$)", fmt_flops, "flops"
    )
    L += [THICKRULE]
    L += block(
        50000, r"$N = 50{,}000$ \quad (windows $5000/1000/200$)", fmt_flops, "flops"
    )
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(os.path.join(WORK, "burgers_highlight_table.tex")).write_text(
        "\n".join(L) + "\n"
    )
    _doc("burgers_highlight_table")
    print("wrote -> burgers_highlight_table.tex")


def write_appendix():
    CAP = (
        r"Timing companion to Table~\ref{tab:exp10_highlight} (exp10 tight-rank/$\varepsilon$ sweep), "
        r"Burgers OCP. \emph{Parallel wall} $T_{\parallel}$ $=$ the idealized $M$-worker wall-clock: the "
        r"$M$ subproblem solves run concurrently (critical path $=$ the slowest one) while the global "
        r"phases -- full-problem Jacobian/Hessian/KKT assembly, composition, and line search -- stay "
        r"serial; $T_{\parallel} = t_{\text{asm}} + \max_i t_{\text{sub},i} + t_{\text{comp}} + "
        r"t_{\text{ls}}$ summed over outer iterations. (This replaces the earlier wall$/M$, which "
        r"wrongly divided the serial global phases by $M$ too.) out/in $=$ outer / total-inner "
        r"iterations; mean$\pm$std over 5 IC-noise seeds; `--' $=$ not converged. Per column/horizon the "
        r"best converged $T_{\parallel}$ is bold. \textbf{Caveat:} walls were measured CPU-contended, so "
        r"$T_{\parallel}$ is indicative -- FLOPs (Table~\ref{tab:exp10_highlight}) remains the fair, "
        r"machine-independent cost axis. Overlapping Schwarz ($b{=}40$, $N{=}10{,}000$) is a "
        r"domain-decomposition baseline; \emph{AOTD (sketch GMRES)} swaps the inner GMRES--QLP for "
        r"randomized sketched-GMRES."
    )
    L = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\caption{" + CAP + r"}",
        r"\label{tab:exp10_appendix}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|rrr|rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{parallel wall $T_{\parallel}$ (s)} & \multicolumn{3}{c}{iters (out/in)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        (r"Method" + r" & $M{=}10$ & $M{=}50$ & $M{=}250$" * 2 + r" \\"),
        r"\midrule",
    ]
    for k, (N, banner) in enumerate(
        [
            (10000, r"$N = 10{,}000$ \quad (windows $1000/200/40$)"),
            (50000, r"$N = 50{,}000$ \quad (windows $5000/1000/200$)"),
        ]
    ):
        if k:
            L += [THICKRULE]
        L += block(N, banner, fmt_pwall, "pwall", show_kkt=False)
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(os.path.join(WORK, "burgers_appendix_table.tex")).write_text(
        "\n".join(L) + "\n"
    )
    _doc("burgers_appendix_table")
    print("wrote -> burgers_appendix_table.tex")


def _doc(stem):
    doc = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.5in,landscape]{geometry}",
            r"\usepackage{booktabs,amsmath,amssymb,graphicx}",
            r"\begin{document}\thispagestyle{empty}",
            rf"\input{{{stem}.tex}}",
            r"\end{document}",
        ]
    )
    Path(os.path.join(WORK, stem + "_doc.tex")).write_text(doc + "\n")


fig_convergence()
fig_adaptation()
write_highlight()
write_appendix()

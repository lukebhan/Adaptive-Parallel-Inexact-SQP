#!/usr/bin/env python3
"""Render Swing convergence, adaptation, and comparison tables."""

import os
from pathlib import Path
import json
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "swing"))
)
os.makedirs(WORK, exist_ok=True)
REC = os.path.join(WORK, "records")
M_COLS = [4, 10, 20, 50]
SEEDS = [1, 2, 3, 4, 5]
CONV = 1e-6
NA, STALL = "--", "---"
ASO, LUC, QLPC, PUR = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#7E2F8E",
)  # AOTD / FOTD-LU / FOTD-QLP / Schwarz
MCOL = {4: "#d62728", 10: "#ff7f0e", 20: "#2ca02c", 50: "#1f77b4"}

# baselines (IPOPT/MultiShoot/ADMM/Schwarz) at the SAME sigma=0.6 IC (run_baselines.py)
D = [
    r
    for r in json.loads(Path(os.path.join(WORK, "baselines_sigma06.json")).read_text())
    if r["method"] in ("IPOPT", "MultiShoot", "ADMM", "Schwarz")
]
# this sweep's records: remap method_label -> (method, b), wall_contended -> wall, family newton
for f in glob.glob(os.path.join(REC, "*.json")):
    r = json.loads(Path(f).read_text())
    if "error" in r:
        continue
    lab = r["method_label"]
    r["method"] = (
        "AOTD" if lab == "AOTD-rebuild" else lab.rsplit("-b", 1)[0]
    )  # FOTD-LU / FOTD-QLP
    r["family"] = "newton"
    r["wall"] = r.get("wall_contended", float("nan"))
    D.append(r)
idx = {(r["method"], r["b"], r["M"], r["seed"]): r for r in D}

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


def runs(method, b, M):
    return [r for r in D if r["method"] == method and r["b"] == b and r["M"] == M]


def _pad(cs):
    L = max(len(c) for c in cs)
    return np.array([list(c) + [c[-1]] * (L - len(c)) for c in cs])


def conv_curve(method, b, M, s):
    r = idx[(method, b, M, s)]
    if r.get("family") == "newton":
        return [float(t["grad_L_norm"]) for t in r["trajectory"]] + [float(r["kkt"])]
    init = idx[("AOTD", None, M, s)]["trajectory"][0][
        "grad_L_norm"
    ]  # baseline: prepend AOTD init
    return [float(init)] + [float(x) for x in r.get("history", [])]


def conv_agg(method, b, M):
    cs = [conv_curve(method, b, M, s) for s in SEEDS if (method, b, M, s) in idx]
    if not cs:
        return None
    A = _pad(cs)
    lm, ls = (
        np.log10(np.clip(A, 1e-30, None)).mean(0),
        np.log10(np.clip(A, 1e-30, None)).std(0),
    )
    return 10**lm, 10 ** (lm - ls), 10 ** (lm + ls)


def adapt_passes(M):
    Bc, Ec = [], []
    for s in SEEDS:
        r = idx.get(("AOTD", None, M, s))
        if not r:
            continue
        P = [p for t in r["trajectory"] for p in t.get("passes", [])]
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


def flops_stats(method, b, M):
    v = runs(method, b, M)
    conv = [r["flops"] for r in v if r["kkt"] < CONV]
    return (
        (np.mean(conv), np.std(conv), len(conv) < len(v))
        if conv
        else (np.nan, 0.0, True)
    )


# convergence: AOTD, FOTD (LU) b40, Schwarz b40
def fig_convergence():
    w, h = set_size(505, subplots=(1, len(M_COLS)))
    fig, ax = plt.subplots(1, len(M_COLS), figsize=(w, h * 1.7), sharey=True)
    series = [
        ("AOTD", None, "AOTD", ASO, "-", 3),
        ("FOTD-LU", 40, r"FOTD (LU) $b{=}40$", LUC, "--", 2),
        ("FOTD-LU", 10, r"FOTD (LU) $b{=}10$", QLPC, ":", 2),
        ("Schwarz", 40, r"Schwarz $b{=}40$", PUR, "-.", 1),
    ]
    for a, M in zip(ax, M_COLS):
        for method, b, lab, c, ls, z in series:
            agg = conv_agg(method, b, M)
            if agg is None:
                continue
            m, lo, hi = agg
            x = np.arange(len(m))
            a.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0, zorder=z - 0.5)
            a.semilogy(x, m, ls, color=c, lw=1.5, label=lab, zorder=z)
        a.axhline(CONV, color="k", ls=":", lw=0.9)
        a.set_title(rf"$M={M}$")
        a.minorticks_off()
        a.grid(alpha=0.3, which="major", lw=0.4)
        a.set_xlabel(r"outer iteration $\tau$")
        a.set_ylim(3e-8, 3e1)
        a.set_yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e1])  # every-other decade (5 ticks)
    ax[0].set_ylabel(r"KKT residual $\|\nabla\mathcal{L}\|$")
    hh, ll = ax[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(
        hh,
        ll,
        loc="lower center",
        ncol=4,
        frameon=False,
        labelcolor="black",
        handlelength=2.0,
        bbox_to_anchor=(0.5, -0.08),
        columnspacing=1.6,
    )
    p = os.path.join(WORK, "swing_convergence.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote ->", os.path.basename(p))


# adaptation: (a) AOTD b, (b) AOTD eps, (c) FLOPs vs M (AOTD, FOTD (LU), Schwarz)
def fig_adaptation():
    M_ADAPT = [4, 20, 50]  # panels (a),(b): only M=4,20,50
    w, h = set_size(505, subplots=(1, 3))
    fig, (ab, ae, af) = plt.subplots(1, 3, figsize=(w, h * 1.7))
    ZO = {4: 6, 20: 4, 50: 2}  # M=4 most prominent (top zorder); uniform linewidth
    emins = []
    for M in M_ADAPT:
        ap = adapt_passes(M)
        if ap is None:
            continue
        bm, blo, bhi, em, elo, ehi = ap
        x = np.arange(len(bm))
        ab.fill_between(x, blo, bhi, color=MCOL[M], alpha=0.15, lw=0, zorder=ZO[M] - 1)
        ab.plot(x, bm, "-", color=MCOL[M], lw=1.6, label=rf"$M={M}$", zorder=ZO[M])
        ae.fill_between(x, elo, ehi, color=MCOL[M], alpha=0.15, lw=0, zorder=ZO[M] - 1)
        ae.semilogy(x, em, "-", color=MCOL[M], lw=1.6, zorder=ZO[M])
        emins.append(float(np.min(elo)))
    ab.set_ylabel(r"overlap $\max_i b_i$")
    ab.set_ylim(0, None)
    ab.set_title(r"(a) overlap adaptation")
    ab.grid(alpha=0.3, lw=0.4)
    ab.legend(frameon=False, loc="upper left")
    ab.set_xlabel(r"inner iteration (pass)")
    ab.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ae.set_ylabel(r"inner tol $\min_i\varepsilon_i$")
    ae.set_title(r"(b) tolerance adaptation")
    ae.grid(alpha=0.3, which="both", lw=0.4)
    ae.set_xlabel(r"inner iteration (pass)")
    lo = (
        10 ** np.floor(np.log10(min(emins))) if emins else 1e-3
    )  # round the min down to a decade (not too low)
    ae.set_ylim(lo, 2e-1)
    ae.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    # (c) grouped box-and-whisker of FLOPs vs M, mirroring the Burgers cost panels: categorical x,
    # methods present spaced evenly within each M group, whiskers = min..max over the 5 seeds.
    cseries = [
        ("AOTD", None, "AOTD", ASO),
        ("FOTD-LU", 40, r"FOTD (LU)", LUC),
        ("FOTD-QLP", 40, r"FOTD (GMRES)", QLPC),
        ("Schwarz", 40, r"Schwarz", PUR),
    ]

    def conv_flops(method, b, M):
        return [
            r["flops"]
            for r in runs(method, b, M)
            if r["kkt"] < CONV and np.isfinite(r.get("flops", np.nan))
        ]

    present = [s for s in cseries if any(conv_flops(s[0], s[1], M) for M in M_COLS)]
    nP = len(present)
    span = 0.68
    centers = np.linspace(-span / 2, span / 2, nP) if nP > 1 else np.array([0.0])
    bw = 0.80 * span / (nP - 1) if nP > 1 else 0.3
    for gi in range(
        len(M_COLS)
    ):  # alternating bands => each M group reads as one cluster
        if gi % 2:
            af.axvspan(gi - 0.5, gi + 0.5, color="0.5", alpha=0.08, lw=0, zorder=0)
    for (method, b, lab, c), dx in zip(present, centers):
        for gi, M in enumerate(M_COLS):
            vals = conv_flops(method, b, M)
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
            for bx in bp["boxes"]:
                bx.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.0)
            for ln in bp["whiskers"] + bp["caps"]:
                ln.set(color=c, lw=1.0)
            for md in bp["medians"]:
                md.set(color=c, lw=1.7)
    af.set_yscale("log")
    af.set_xticks(range(len(M_COLS)))
    af.set_xticklabels(M_COLS)
    af.set_xlim(-0.6, len(M_COLS) - 0.4)
    af.set_ylabel(r"FLOPs")
    af.set_xlabel(r"$M$")
    af.set_title(r"(c) comp.\ cost vs $M$")
    af.grid(alpha=0.3, which="both", axis="y", lw=0.4)
    af.yaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 3.0), numticks=12)
    )  # >=4 ticks over ~2 decades
    fig.tight_layout()
    # shared box legend under panel (c) (mirrors the Burgers cost-column legend)
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
        bbox_to_anchor=(pos.x0 + pos.width / 2, pos.y0 - 0.11),
        ncol=2,
        frameon=False,
        labelcolor="black",
        handlelength=1.3,
        columnspacing=1.4,
        fontsize=8,
    )
    p = os.path.join(WORK, "swing_adaptation.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote ->", os.path.basename(p))


# tables
def _pwall(x):
    """Estimate parallel wall time from critical-path solves; global phases stay serial."""
    if x.get("family") == "newton":
        tr = x.get("trajectory")
        if tr:
            return float(
                sum(t["t_outer"] - t["t_subsolve"] + t["t_subsolve_par"] for t in tr)
            )
        return float("nan")
    return float(x.get("par_wall", x.get("wall", float("nan"))))


def cell(method, b, M):
    v = [
        r
        for r in D
        if r["method"] == method and r["b"] == b and (M is None or r["M"] == M)
    ]
    if not v:
        return None
    kkt = float(np.mean([x["kkt"] for x in v]))
    pw = [_pwall(x) for x in v]
    return dict(
        kkt=kkt / 1e-7,
        flops=float(np.mean([x["flops"] for x in v])),
        flops_sd=float(np.std([x["flops"] for x in v])),
        pwall=float(np.nanmean(pw)),
        pwall_sd=float(np.nanstd(pw)),
        outer=float(np.mean([x["outer_iters"] for x in v])),
        inner=float(np.mean([x["inner_iters"] for x in v])),
        conv=kkt < CONV,
    )


B_SCHWARZ = [1, 10, 40, 75]
B_FOTD = [10, 40, 75]
ROWS = (
    [
        ("IPOPT", "IPOPT", "span"),
        ("MultiShoot", "MultiShoot", "mcol"),
        (r"ADMM ($\rho{=}10$)", "ADMM", "mcol"),
        ("Schwarz", "Schwarz", "head"),
    ]
    + [(rf"\quad $b{{=}}{b}$", "Schwarz", "brow", b) for b in B_SCHWARZ]
    + [("FOTD (LU)", "FOTD-LU", "head")]
    + [(rf"\quad $b{{=}}{b}$", "FOTD-LU", "brow", b) for b in B_FOTD]
    + [("FOTD (GMRES)", "FOTD-QLP", "head")]
    + [(rf"\quad $b{{=}}{b}$", "FOTD-QLP", "brow", b) for b in B_FOTD]
    + [("AOTD (GMRES)", "AOTD", "mcol"), ("AOTD (sGMRES)", "AOTD-sketch", "mcol")]
)
COMPET = {"MultiShoot", "ADMM", "Schwarz", "FOTD-LU", "FOTD-QLP", "AOTD", "AOTD-sketch"}
ITERS = {"FOTD-LU", "FOTD-QLP", "AOTD", "AOTD-sketch"}


def build_table(
    value_fmt, bestkey, header_val, caption, label, stem, unit=1e7, show_kkt=True
):
    data = []
    for spec in ROWS:
        method, kind = spec[1], spec[2]
        b = spec[3] if len(spec) > 3 else None
        if kind == "span":
            data.append((spec, {M: cell(method, None, None) for M in M_COLS}))
        elif kind == "head":
            data.append((spec, None))
        else:
            data.append((spec, {M: cell(method, b, M) for M in M_COLS}))
    best = {
        M: min(
            (
                round(d[M][bestkey] / (M if bestkey == "wall" else 1), 3)
                for spec, d in data
                if d and d[M] and d[M]["conv"] and spec[1] in COMPET
            ),
            default=None,
        )
        for M in M_COLS
    }

    def kf(c, M):  # Leave KKT values unbolded.
        if c is None:
            return NA
        return STALL if not c["conv"] else f"{c['kkt']:.2f}"

    def vf(c, M):
        if c is None:
            return NA
        if not c["conv"]:
            return STALL
        return value_fmt(c, M, best)

    def itf(spec, c):
        if c is None or spec[1] not in ITERS:
            return NA
        return STALL if not c["conv"] else f"{c['outer']:.0f}/{c['inner']:.0f}"

    mh = " & ".join(rf"$M{{=}}{M}$" for M in M_COLS)
    ng = 3 if show_kkt else 2  # column groups (KKT | value | iters) or (value | iters)
    colspec = "l|" + "rrrr|" * (ng - 1) + "rrrr"
    if show_kkt:
        head_mc = (
            rf"& \multicolumn{{4}}{{c|}}{{KKT residual ($\times 10^{{-7}}$)}} "
            rf"& \multicolumn{{4}}{{c|}}{{{header_val}}} & \multicolumn{{4}}{{c}}{{iters (out/in)}} \\"
        )
        cmid = r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}"
    else:
        head_mc = rf"& \multicolumn{{4}}{{c|}}{{{header_val}}} & \multicolumn{{4}}{{c}}{{iters (out/in)}} \\"
        cmid = r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"
    L = [
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + r"}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        head_mc,
        cmid,
        r"Method & " + " & ".join([mh] * ng) + r" \\",
        r"\midrule",
    ]
    for spec, d in data:
        label_r, kind = spec[0], spec[2]
        if kind == "head":
            L.append(rf"{label_r} " + "& " * (4 * ng) + r"\\")
        elif kind == "span":
            c = d[M_COLS[0]]
            ok = c is not None and c["conv"]
            fv = value_fmt(c, 1, {1: None}, span=True) if ok else STALL
            kk = f"{c['kkt']:.2f}" if ok else STALL
            parts = ([rf"\multicolumn{{4}}{{c|}}{{{kk}}}"] if show_kkt else []) + [
                rf"\multicolumn{{4}}{{c|}}{{{fv}}}",
                rf"\multicolumn{{4}}{{c}}{{{NA}}}",
            ]
            L.append(rf"{label_r} & " + " & ".join(parts) + r" \\")
        else:
            cols = []
            if show_kkt:
                cols.append(" & ".join(kf(d[M], M) for M in M_COLS))
            cols += [
                " & ".join(vf(d[M], M) for M in M_COLS),
                " & ".join(itf(spec, d[M]) for M in M_COLS),
            ]
            L.append(rf"{label_r} & " + " & ".join(cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    tex = "\n".join(L)
    tex = tex.replace("MultiShoot &", r"\midrule" + "\nMultiShoot &", 1).replace(
        "AOTD (GMRES) &", r"\midrule" + "\nAOTD (GMRES) &", 1
    )
    Path(os.path.join(WORK, stem + ".tex")).write_text(tex + "\n")
    doc = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[a4paper,margin=0.4in,landscape]{geometry}",
            r"\usepackage{booktabs,amsmath,amssymb,lmodern,graphicx}",
            r"\begin{document}\thispagestyle{empty}",
            rf"\input{{{stem}.tex}}",
            r"\end{document}",
        ]
    )
    Path(os.path.join(WORK, stem + "_doc.tex")).write_text(doc + "\n")
    print("wrote ->", stem + ".tex")


def fmt_flops(c, M, best, span=False):
    s = rf"${c['flops'] / 1e7:.1f}{{\pm}}{c['flops_sd'] / 1e7:.1f}$"
    if span:
        return s
    return (
        rf"{{\boldmath {s}}}"
        if best[M] is not None
        and abs(c["flops"] / 1e7 - best[M] / 1e7) < 1e-3 * (best[M] / 1e7)
        else s
    )


def fmt_pwall(c, M, best, span=False):
    s = rf"${c['pwall']:.1f}{{\pm}}{c['pwall_sd']:.1f}$"
    if span:
        return s  # IPOPT monolithic: T_par = wall
    return (
        rf"{{\boldmath {s}}}"
        if best[M] is not None and round(c["pwall"], 3) == best[M]
        else s
    )


HL_CAP = (
    r"Comparison on the NE39 swing OCP ($N{=}1000$, 5 IC-noise seeds with the larger $\sigma{=}0.6$ "
    r"initial-condition perturbation $x_0{=}x_0^{\mathrm{nom}}(0.25){+}0.6\cdot0.25\,\xi$, $\xi\sim\mathcal N(0,I)$, mean). AOTD uses the hybrid "
    r"law ($\varrho_b{=}4$, $\varrho_\varepsilon{=}0.2$). The two accurate FOTD baselines solve each local "
    r"Newton system by a direct sparse factorization -- \textbf{FOTD (LU)} -- or GMRES--QLP to "
    r"$\varepsilon_i{=}$rank-tol${=}10^{-14}$ with inner iterations capped at 100 -- \textbf{FOTD (GMRES)} "
    r"(both give the same convergence; FLOPs differ). KKT residual in units of $10^{-7}$ (converged iff "
    r"$<\!10$); FLOPs in $10^{7}$, mean$\pm$std, under the FLOP model that counts the Schur-complement "
    r"formation. `---' $=$ did not converge; `--' $=$ metric not defined (out/in apply only to FOTD/AOTD). "
    r"Columns are $M$; FOTD/Schwarz at fixed overlap $b$, AOTD adapts $b_i$. Per column the best converged "
    r"KKT and FLOPs among the decomposed methods are bold."
)
AP_CAP = (
    r"Timing companion to Table~\ref{tab:swing_exp10_highlight}, NE39 swing. \emph{Parallel wall} "
    r"$T_{\parallel}$ $=$ the idealized $M$-worker wall-clock: the $M$ subproblem solves run concurrently "
    r"(critical path $=$ the slowest one) while the global phases (assembly, composition/consensus, line "
    r"search) stay serial, i.e.\ $T_{\parallel}=$ wall $-\,t_{\text{sub}}^{\text{serial}}+t_{\text{sub}}"
    r"^{\parallel}$ (IPOPT is monolithic $\Rightarrow T_{\parallel}=$ wall). This replaces the earlier "
    r"wall$/M$, which wrongly divided the serial global phases by $M$ too. out/in $=$ outer / total-inner "
    r"iterations; mean$\pm$std over 5 seeds; `---' $=$ not converged. \textbf{Caveat:} AOTD/FOTD are "
    r"unoptimized Python while IPOPT/Schwarz/MultiShoot/ADMM use compiled CasADi/Ipopt, and walls are "
    r"CPU-contended, so $T_{\parallel}$ is indicative; FLOPs (Table~\ref{tab:swing_exp10_highlight}) is the "
    r"fair, machine-independent axis. Per column the best $T_{\parallel}$ among decomposed methods is bold."
)

fig_convergence()
fig_adaptation()
build_table(
    fmt_flops,
    "flops",
    r"FLOPs ($\times 10^{7}$)",
    HL_CAP,
    "tab:swing_exp10_highlight",
    "swing_highlight_table",
)
build_table(
    fmt_pwall,
    "pwall",
    r"parallel wall $T_{\parallel}$ (s)",
    AP_CAP,
    "tab:swing_exp10_appendix",
    "swing_appendix_table",
    show_kkt=False,
)

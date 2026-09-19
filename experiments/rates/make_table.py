#!/usr/bin/env python3
"""Tabulate the M=20 rate sweep as seed means and standard deviations.

Overlap updates count failed accuracy passes; extrema use final parameter vectors."""

import os
from pathlib import Path
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "rates"))
)
os.makedirs(WORK, exist_ok=True)
REC = os.path.join(WORK, "records")
RB = [1.0, 2.0, 4.0, 10.0]
RE = [0.1, 0.2, 0.3, 1.0]
HEADLINE = (4.0, 0.2)
M_COLS = [20]  # all M in the sweep (markdown diagnostic covers every one)
M_TEX = [20]  # M values that get a LaTeX table
SEEDS = [1, 2, 3, 4, 5]
CONV = 1e-6

D = [
    json.loads(Path(f).read_text())
    for f in sorted(glob.glob(os.path.join(REC, "*.json")))
]
D = [r for r in D if "error" not in r]
if not D:
    raise SystemExit("no records -- run run_sweep.py first")


def update_count(record):
    """Count accuracy failures, with a trajectory fallback for archived records."""
    if "n_updates" in record:
        return record["n_updates"]
    return sum(
        int(
            step.get(
                "n_acc_fail",
                sum(p.get("outcome") == "acc_fail" for p in step.get("passes", [])),
            )
        )
        for step in record.get("trajectory", [])
    )


def cell(rb, re, M):
    v = [r for r in D if r["varrho_b"] == rb and r["varrho_eps"] == re and r["M"] == M]
    if not v:
        return None
    nc = int(sum(1 for r in v if r["kkt"] < CONV))
    out = dict(nconv=nc, ntot=len(v), conv=(nc == len(v)))
    for name, vals in [
        ("kkt", [r["kkt"] for r in v]),
        ("flops", [r["flops"] for r in v]),
        ("outer", [r["outer_iters"] for r in v]),
        ("nupdate", [update_count(r) for r in v]),
        ("krylov", [r["inner_iters"] for r in v]),
        ("bmax", [max(r["b_list"]) for r in v]),
        ("emin", [min(r["eps_i_list"]) for r in v]),
    ]:
        a = np.array(vals, float)
        out[name], out[name + "_sd"] = float(a.mean()), float(a.std())
    return out


CELLS = {(rb, re, M): cell(rb, re, M) for rb in RB for re in RE for M in M_COLS}


# Eight-column LaTeX table.
def sci(v, p=1):
    if v is None or not np.isfinite(v) or v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10**e
    if round(abs(m), p) >= 10:  # rounding carry: 9.98e8 -> 1.0e9, not 10.0e8
        m /= 10
        e += 1
    return rf"${m:.{p}f}\times10^{{{e}}}$"


def pw(x):
    return rf"${x:g}$"


def fixed(v, unit, p=1):
    """Plain number in a FIXED unit (column header carries the power of ten)."""
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v / unit:.{p}f}"


def ms(m, sd, unit=1.0, p=1):
    """mean $\\pm$ std in a FIXED unit (column header carries the power of ten)."""
    if m is None or not np.isfinite(m):
        return "--"
    return rf"${m / unit:.{p}f}\pm{sd / unit:.{p}f}$"


def ms_sci(m, sd, p=1):
    """Format mean and standard deviation with a shared power of ten."""
    if m is None or not np.isfinite(m) or m == 0:
        return "--"
    e = int(np.floor(np.log10(abs(m))))
    a, b = m / 10**e, sd / 10**e
    if round(a, p) >= 10:  # rounding carry: 9.98 -> 1.0, bump exponent
        a /= 10
        b /= 10
        e += 1
    q = 0 if b >= 10 else p  # a std that dwarfs the mean stays readable
    return rf"$({a:.{p}f}\pm{b:.{q}f})\times10^{{{e}}}$"


CAP = (
    r"AOTD adaptation-rate ablation $(\varrho_b,\varrho_\varepsilon)$ under the \emph{hybrid} "
    r"update law with the tight-rank configuration of Table~\ref{tab:swing_exp10_highlight} on the "
    r"NE39 swing OCP ($N=1000$, $M=%d$). Law: "
    r"$\varepsilon_i\!\leftarrow\!\max\{\varepsilon_i\exp(-\varrho_\varepsilon(1/\sqrt{M}+\|r_i\|/\|r\|)),10^{-9}\}$, "
    r"$b_i\!\leftarrow\!\min\{b_i+\max\{1,\lceil\varrho_b\|r_i\|/\|r\|\rceil\},110\}$, i.e.\ the "
    r"overlap step is floored at $1$ so $b_i$ grows on every failing pass. Inner solves are "
    r"GMRES--QLP with rank tolerance $10^{-14}$ and a rebuilt ILU--Schur preconditioner. "
    r"Every entry is the mean $\pm$ one standard deviation over five $\sigma=0.6$ IC-noise seeds; "
    r"all 16 rate pairs converge on all five seeds. "
    r"KKT residual in units of $10^{-7}$, FLOPs in units of $10^{7}$. "
    r"\# overlap updates counts failed accuracy passes on which the local overlaps and tolerances are updated; $b_i^{\max}$ and $\varepsilon_i^{\min}$ are the "
    r"extremes of the final per-subproblem vectors."
)

blocks = []
for M in M_TEX:
    L = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + (CAP % M) + r"}",
        rf"\label{{tab:rate_ablation_M{M}}}",
        r"\tablesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{cc r r cc c c}",
        r"\toprule",
        r"$\varrho_b$ & $\varrho_\varepsilon$ & "
        r"\shortstack{$\|\nabla\mathcal{L}\|$\\($\times 10^{-7}$)} & "
        r"\shortstack{FLOPs\\($\times 10^{7}$)} & "
        r"\shortstack{\# overlap\\updates} & \shortstack{\# outer\\iterations} & "
        r"$b_i^{\max}$ & $\varepsilon_i^{\min}$ \\",
        r"\midrule",
    ]
    for rb in RB:
        for re in RE:
            c = CELLS[(rb, re, M)]
            if c is None:
                continue
            L.append(
                rf"{pw(rb)} & {pw(re)} & {ms(c['kkt'], c['kkt_sd'], 1e-7)} & "
                rf"{ms(c['flops'], c['flops_sd'], 1e7)} & "
                rf"{ms(c['nupdate'], c['nupdate_sd'])} & {ms(c['outer'], c['outer_sd'])} & "
                rf"{ms(c['bmax'], c['bmax_sd'])} & "
                rf"{ms_sci(c['emin'], c['emin_sd'])} \\"
            )
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    blocks.append("\n".join(L))
tex = "\n\n".join(blocks)
Path(os.path.join(WORK, "rate_ablation_table.tex")).write_text(tex + "\n")
doc = "\n".join(
    [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.6in]{geometry}",
        r"\usepackage{booktabs,amsmath,amssymb,graphicx}",
        r"\providecommand{\tablesize}{\footnotesize}",
        r"\begin{document}\thispagestyle{empty}",
        r"\input{rate_ablation_table.tex}",
        r"\end{document}",
    ]
)
Path(os.path.join(WORK, "rate_ablation_table_doc.tex")).write_text(doc + "\n")
print(
    f"wrote -> rate_ablation_table.tex ({len(M_TEX)} table(s), M={M_TEX}) (+ _doc.tex)"
)


# markdown diagnostic
hdr = (
    "| $M$ | $\\varrho_b$ | $\\varrho_\\varepsilon$ | seeds conv | $\\|\\nabla L\\|$ | FLOPs | "
    "\\#outer | \\#overlap updates | total Krylov | $b_i^{\\max}$ | $\\varepsilon_i^{\\min}$ |"
)
lines = [
    "# exp12 hybrid-law rate ablation (exp10 algorithm + settings), NE39 swing $N=1000$",
    "",
    "AOTD-rebuild, `adapt_mode=hybrid`, `hybrid_b_min1=True`, GMRES-QLP rank-tol $10^{-14}$, "
    "rebuilt ILU-Schur, $\\sigma=0.6$ IC seeds 1-5. Every entry is mean$\\pm$std over the "
    "5 seeds.",
    "",
    hdr,
    "|" + "---|" * 11,
]
for M in M_COLS:
    for rb in RB:
        for re in RE:
            c = CELLS[(rb, re, M)]
            if c is None:
                continue
            star = " *(headline)*" if (rb, re) == HEADLINE else ""
            lines.append(
                f"| {M} | {rb:g}{star} | {re:g} | {c['nconv']}/{c['ntot']} | "
                f"{c['kkt']:.2e}$\\pm${c['kkt_sd']:.1e} | "
                f"{c['flops']:.3e}$\\pm${c['flops_sd']:.1e} | "
                f"{c['outer']:.1f}$\\pm${c['outer_sd']:.1f} | "
                f"{c['nupdate']:.1f}$\\pm${c['nupdate_sd']:.1f} | "
                f"{c['krylov']:.0f}$\\pm${c['krylov_sd']:.0f} | "
                f"{c['bmax']:.1f}$\\pm${c['bmax_sd']:.1f} | "
                f"{c['emin']:.2e}$\\pm${c['emin_sd']:.1e} |"
            )
lines += [
    "",
    "## Cheapest all-seeds-converged rate pair per $M$",
    "",
    "| $M$ | best $(\\varrho_b,\\varrho_\\varepsilon)$ | FLOPs | headline $(4,0.2)$ FLOPs | ratio |",
    "|---|---|---|---|---|",
]
for M in M_COLS:
    cand = [
        ((rb, re), CELLS[(rb, re, M)])
        for rb in RB
        for re in RE
        if CELLS[(rb, re, M)] and CELLS[(rb, re, M)]["conv"]
    ]
    if not cand:
        lines.append(f"| {M} | none converged 5/5 | -- | -- | -- |")
        continue
    (rb, re), c = min(cand, key=lambda t: t[1]["flops"])
    h = CELLS[(HEADLINE[0], HEADLINE[1], M)]
    hs = f"{h['flops']:.3e}$\\pm${h['flops_sd']:.1e}" if h and h["conv"] else "not 5/5"
    ratio = f"{h['flops'] / c['flops']:.2f}x" if h and h["conv"] else "--"
    lines.append(
        f"| {M} | ({rb:g}, {re:g}) | {c['flops']:.3e}$\\pm${c['flops_sd']:.1e} | "
        f"{hs} | {ratio} |"
    )
Path(os.path.join(WORK, "rate_ablation_table.md")).write_text("\n".join(lines) + "\n")
print("wrote -> rate_ablation_table.md")
for M in M_COLS:
    cand = [
        ((rb, re), CELLS[(rb, re, M)])
        for rb in RB
        for re in RE
        if CELLS[(rb, re, M)] and CELLS[(rb, re, M)]["conv"]
    ]
    if not cand:
        print(f"  M={M:>2}: no rate pair converged on all 5 seeds")
        continue
    (rb, re), c = min(cand, key=lambda t: t[1]["flops"])
    h = CELLS[(HEADLINE[0], HEADLINE[1], M)]
    hs = f"{h['flops']:.3e}" if h and h["conv"] else "not 5/5"
    print(
        f"  M={M:>2}: cheapest 5/5 = (vb={rb:g}, ve={re:g}) {c['flops']:.3e} FLOPs; "
        f"headline (4,0.2) = {hs}"
    )

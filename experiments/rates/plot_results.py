#!/usr/bin/env python3
"""Plot M=20 adaptation traces with seed means and one-standard-deviation bands."""

import os
from pathlib import Path
import sys
import json
import glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "AOTD_WORK", os.path.abspath(os.path.join(HERE, "..", "..", "results", "rates"))
)
os.makedirs(WORK, exist_ok=True)
REC = os.path.join(WORK, "records")
RB = [1.0, 2.0, 4.0, 10.0]
RE = [0.1, 0.2, 0.3, 1.0]
FIX_B, FIX_E = 4.0, 0.2  # Reference rates.
M_PANEL = 20
SEEDS = [1, 2, 3, 4, 5]
QUAD = [
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#7E2F8E",
]  # blue / green / vermillion / purple
COLB = {r: c for r, c in zip(RB, QUAD)}
LWB = {r: w for r, w in zip(RB, [3.0, 2.2, 1.6, 1.1])}
COLE = {r: c for r, c in zip(RE, QUAD)}
LWE = {r: w for r, w in zip(RE, [3.0, 2.2, 1.6, 1.1])}

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "axes.labelsize": 9,
        "font.size": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 9,
    }
)

D = []
for f in sorted(glob.glob(os.path.join(REC, "*.json"))):
    r = json.loads(Path(f).read_text())
    if "error" not in r:
        D.append(r)
if not D:
    sys.exit("no records -- run run_sweep.py first")
IDX = {(r["varrho_b"], r["varrho_eps"], r["M"], r["seed"]): r for r in D}


def set_size(width, subplots=(1, 1)):
    g = (5**0.5 - 1) / 2
    w = width / 72.27
    return (w, w * g * (subplots[0] / subplots[1]))


def _pad(cs):
    L = max(len(c) for c in cs)
    return np.array([list(c) + [c[-1]] * (L - len(c)) for c in cs], float)


def pass_agg(rb, re, M, key, log=False):
    """Seed-aggregated per-inner-pass series of max_i b_i (key='b_used') or min_i eps_i
    (key='eps_used'); returns (mean, lo, hi) padded to the longest seed."""
    cs = []
    for s in SEEDS:
        r = IDX.get((rb, re, M, s))
        if r is None:
            continue
        P = [p for t in r["trajectory"] for p in t.get("passes", [])]
        if not P:
            continue
        cs.append([min(p[key]) if log else max(p[key]) for p in P])
    if not cs:
        return None
    A = _pad(cs)
    if log:
        A = np.clip(A, 1e-9, None)
        lm, ls = np.log10(A).mean(0), np.log10(A).std(0)
        return 10**lm, 10 ** (lm - ls), 10 ** (lm + ls)
    m, sd = A.mean(0), A.std(0)
    return m, np.clip(m - sd, 0, None), m + sd


def panel_series(ax, rates, fixed, which):
    """(a)/(b): per-pass adaptation traces at M=M_PANEL."""
    for rt in rates:
        rb, re = (rt, fixed) if which == "b" else (fixed, rt)
        agg = pass_agg(
            rb,
            re,
            M_PANEL,
            "b_used" if which == "b" else "eps_used",
            log=(which == "e"),
        )
        if agg is None:
            continue
        m, lo, hi = agg
        x = np.arange(len(m))
        c = COLB[rt] if which == "b" else COLE[rt]
        lw = LWB[rt] if which == "b" else LWE[rt]
        ax.fill_between(x, lo, hi, color=c, alpha=0.16, lw=0)
        ax.plot(x, m, color=c, lw=lw)


fig, (aL, aR) = plt.subplots(1, 2, figsize=set_size(505.89, subplots=(1, 2)))

panel_series(aL, RB, FIX_E, "b")
aL.set_xlabel(r"inner iteration (pass)")
aL.set_ylabel(r"overlap $\max_i b_i$")
aL.set_title(
    rf"(a) vary $\varrho_b$ \ ($\varrho_\varepsilon{{=}}{FIX_E:g}$, $M{{=}}{M_PANEL}$)"
)
aL.set_xlim(left=0)
aL.set_ylim(bottom=0)
aL.grid(alpha=0.3, lw=0.4)
aL.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
aL.legend(
    handles=[
        plt.Line2D([], [], color=COLB[r], lw=LWB[r], label=rf"$\varrho_b{{=}}{r:g}$")
        for r in RB
    ],
    frameon=False,
    loc="upper right",
    ncol=2,
    columnspacing=1.1,
)

panel_series(aR, RE, FIX_B, "e")
aR.set_yscale("log")
aR.set_ylim(5e-10, 3e-1)
aR.set_xlabel(r"inner iteration (pass)")
aR.set_ylabel(r"inner tol $\min_i\varepsilon_i$")
aR.set_title(
    rf"(b) vary $\varrho_\varepsilon$ \ ($\varrho_b{{=}}{FIX_B:g}$, $M{{=}}{M_PANEL}$)"
)
aR.set_xlim(left=0)
aR.grid(alpha=0.3, which="both", lw=0.4)
aR.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
aR.legend(
    handles=[
        plt.Line2D(
            [], [], color=COLE[r], lw=LWE[r], label=rf"$\varrho_\varepsilon{{=}}{r:g}$"
        )
        for r in RE
    ],
    frameon=False,
    loc="lower left",
    ncol=2,
    columnspacing=1.1,
)

fig.tight_layout()
p = os.path.join(WORK, "rate_ablation.pdf")
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print("wrote ->", os.path.relpath(p))
print(
    f"({len(D)} records over {len(set((r['varrho_b'], r['varrho_eps']) for r in D))} rate pairs; "
    f"M={M_PANEL} slice, seed-mean +- 1 std over {len(SEEDS)} seeds)"
)

#!/usr/bin/env python3
"""Plot penalty updates, KKT convergence, and descent-failure counts."""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.7,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def set_size(width, fraction=1, subplots=(1, 1)):
    """Figure dimensions (in) from a LaTeX width in points, so fonts don't scale.
    (J. Walton, "Embed your matplotlib figures the right way".)"""
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get(
    "AOTD_WORK",
    os.path.abspath(os.path.join(HERE, "..", "..", "results", "globalization")),
)
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(OUT, "ablation_eta_data.json")
NU_COLOR = {1.1: "#b2182b", 1.3: "#e08214", 1.5: "#1b7837", 2.0: "#2166ac"}


def main():
    with open(DATA) as fh:
        runs = json.load(fh)

    fig, ax = plt.subplots(2, 2, figsize=set_size(505.89, subplots=(2, 2)))
    axA, axB, axC, axD = ax.ravel()
    any_stall = False

    # Use the median penalty initialization as the representative curve for each nu.
    E1_REP, E2_REP = 1e-1, 1e2
    rep_c = {
        r["nu"]: r for r in runs if r["eta1_0"] == E1_REP and r["eta2_0"] == E2_REP
    }

    for r in runs:
        col = NU_COLOR.get(r["nu"], "#777777")
        f = r["firings"]
        if f:
            k = [0] + [p["fire"] for p in f]
            axA.plot(
                k,
                [r["eta1_0"]] + [p["eta1"] for p in f],
                color=col,
                lw=1.2,
                alpha=0.75,
                marker=".",
                ms=4,
            )
            axB.plot(
                k,
                [r["eta2_0"]] + [p["eta2"] for p in f],
                color=col,
                lw=1.2,
                alpha=0.75,
                marker=".",
                ms=4,
            )
        if rep_c.get(r["nu"]) is r:
            po = r["per_outer"]
            # The converged outer iteration is not logged; append its final residual.
            ctau = [o["tau"] for o in po] + [len(po)]
            ckkt = [o["grad_L_norm"] for o in po] + [r["final_kkt"]]
            axC.semilogy(ctau, ckkt, color=col, lw=1.4, alpha=0.9, marker=".", ms=4)
        axD.plot(
            r["nu"],
            r["n_firings"],
            marker="o",
            ms=10,
            mfc=(col if r["converged"] else "white"),
            mec=col,
            mew=1.8,
            alpha=0.85,
            ls="none",
            zorder=3,
        )
        any_stall |= not r["converged"]

    # fit k ≈ c / ln(ν) on panel D
    nu = np.array([r["nu"] for r in runs])
    kf = np.array([r["n_firings"] for r in runs])
    c = float(np.mean(kf * np.log(nu)))  # least-squares-ish on k·ln ν
    xs = np.linspace(min(NU_COLOR) - 0.03, max(NU_COLOR) + 0.05, 200)
    axD.plot(xs, c / np.log(xs), "k--", lw=1.4, zorder=2)

    axA.set(
        title=r"(a)   $\eta_1$ after each firing  ($\times\nu^2$ per firing)",
        xlabel="cumulative descent firing  $k$",
        ylabel=r"$\eta_1$",
        yscale="log",
    )
    axB.set(
        title=r"(b)   $\eta_2$ after each firing  ($/\nu$ per firing)",
        xlabel="cumulative descent firing  $k$",
        ylabel=r"$\eta_2$",
        yscale="log",
    )
    axC.set(
        title=r"(c)   KKT $\|\nabla L\|$ vs outer iteration",
        xlabel=r"outer iteration  $\tau$",
        ylabel=r"$\|\nabla L\|$",
    )
    axC.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axC.axhline(1e-6, color="k", lw=0.8, ls=":")
    axC.text(
        0.03,
        0.04,
        r"tol $=10^{-6}$",
        transform=axC.transAxes,
        ha="left",
        va="bottom",
        color="0.4",
        fontsize=7,
    )
    axD.set(
        title=r"(d)   descent firings vs $\nu$",
        xlabel=r"$\nu$",
        ylabel="# descent firings",
    )
    axD.set_xticks(sorted(NU_COLOR))
    axD.margins(x=0.12)

    handles = [
        Line2D([0], [0], color=c_, lw=2.2, marker=".", label=rf"$\nu={nu_:g}$")
        for nu_, c_ in NU_COLOR.items()
    ]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"ablation_eta_init.{ext}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("wrote", p)

    md = [
        "# Descent-firing ablation — NE39 swing OCP (seed 1)\n",
        f"Firing count is a function of ν only (η1, η2, seed invariant): "
        f"k ≈ ln D / ln ν with D ≈ {np.exp(c):.1f}. The η-ratchet is geometric "
        f"(η1=η1_0·ν^2k, η2=η2_0·ν^-k).\n",
        "| η1_0 | η2_0 | ν | firings | converges | outer iters | final KKT |",
        "|---:|---:|---:|---:|:--:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda x: (x["nu"], x["eta1_0"], x["eta2_0"])):
        md.append(
            f"| {r['eta1_0']:.0e} | {r['eta2_0']:.0e} | {r['nu']:g} "
            f"| {r['n_firings']} | {'✓' if r['converged'] else '✗'} "
            f"| {r['outer_iters']} | {r['final_kkt']:.2e} |"
        )
    with open(os.path.join(OUT, "ablation_eta_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", os.path.join(OUT, "ablation_eta_results.md"))


if __name__ == "__main__":
    main()

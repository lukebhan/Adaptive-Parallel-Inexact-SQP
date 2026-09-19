#!/usr/bin/env python3
"""Plot penalty updates, KKT convergence, and descent-failure counts."""

import os
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


NU_COLOR = {
    1.01: "#762a83",
    1.1: "#b2182b",
    1.3: "#e08214",
    1.5: "#1b7837",
    2.0: "#2166ac",
}


def render(records, output, config):
    OUT = str(output)
    runs = []
    for row in records:
        firings = []
        for step in row["trajectory"]:
            for p in step["passes"]:
                if p["outcome"] == "desc_fail":
                    firings.append(
                        dict(fire=len(firings) + 1, eta1=p["eta1"], eta2=p["eta2"])
                    )
        runs.append(
            dict(
                eta1_0=row["config"]["eta1_0"],
                eta2_0=row["config"]["eta2_0"],
                nu=row["config"]["nu"],
                converged=row["converged"],
                n_firings=len(firings),
                firings=firings,
                per_outer=row["trajectory"],
                final_kkt=row["kkt"],
                outer_iters=row["outer_iters"],
            )
        )
    nu_values = sorted({r["nu"] for r in runs})
    if all(nu in NU_COLOR for nu in nu_values):
        colors = {nu: NU_COLOR[nu] for nu in nu_values}
    else:
        # New parameter grids need distinct colors, not the same gray fallback.
        palette = ["#b2182b", "#e08214", "#1b7837", "#2166ac", "#762a83"]
        colors = {
            nu: (
                palette[i]
                if len(nu_values) <= len(palette)
                else plt.get_cmap("turbo")(0.08 + 0.84 * i / (len(nu_values) - 1))
            )
            for i, nu in enumerate(nu_values)
        }
    total_firings = sum(r["n_firings"] for r in runs)
    count_labels = True

    fig, ax = plt.subplots(2, 2, figsize=set_size(505.89, subplots=(2, 2)))
    axA, axB, axC, axD = ax.ravel()
    any_stall = False

    # Use the median penalty initialization as the representative curve for each nu.
    e1_values = sorted({r["eta1_0"] for r in runs})
    e2_values = sorted({r["eta2_0"] for r in runs})
    E1_REP, E2_REP = e1_values[len(e1_values) // 2], e2_values[len(e2_values) // 2]
    rep_c = {
        r["nu"]: r for r in runs if r["eta1_0"] == E1_REP and r["eta2_0"] == E2_REP
    }

    for r in runs:
        col = colors[r["nu"]]
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
            # Count applied steps, not a failed attempt or terminal KKT check.
            applied = [o for o in po if o.get("step_applied", True)]
            ctau = list(range(len(applied) + 1))
            ckkt = [o["grad_L_norm"] for o in applied] + [r["final_kkt"]]
            axC.semilogy(ctau, ckkt, color=col, lw=1.4, alpha=0.9, marker=".", ms=4)
        if not count_labels:
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

    if count_labels:
        from collections import Counter

        groups = Counter((r["nu"], r["n_firings"], r["converged"]) for r in runs)
        for (nu, k, converged), count in sorted(groups.items()):
            col = colors[nu]
            axD.plot(
                nu,
                k,
                marker="o",
                ms=13,
                mfc=col if converged else "white",
                mec=col,
                mew=1.5,
                ls="none",
                zorder=3,
            )
            axD.text(
                nu,
                k,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if converged else col,
                zorder=4,
            )
        axD.text(
            0.97,
            0.97,
            "Numbers: initializations",
            transform=axD.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

    if not total_firings:
        # Zero updates are data, not missing traces. Show the initial penalties
        # at k=0 and do not imply a fitted firing-count law.
        for axis, values in ((axA, e1_values), (axB, e2_values)):
            axis.plot([0] * len(values), values, "o", color="0.25", ms=4)
            axis.set_xlim(-0.25, 0.25)
            axis.set_xticks([0])
            axis.text(
                0.98,
                0.06,
                "No descent updates",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )
        axD.set_ylim(-0.25, 1)
        axD.set_yticks([0, 1])

    if any_stall:
        axD.legend(
            handles=[
                Line2D([], [], marker="o", color="0.3", ls="none", label="Converged"),
                Line2D(
                    [],
                    [],
                    marker="o",
                    mfc="white",
                    color="0.3",
                    ls="none",
                    label="Stopped",
                ),
            ],
            loc="upper right",
            fontsize=7,
            frameon=False,
        )
        if all(not r["converged"] for r in runs if r["n_firings"]):
            axA.text(
                0.97,
                0.04,
                "All updated runs stopped",
                transform=axA.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
            )
        axC.text(
            0.97,
            0.96,
            rf"$\eta_1^0={E1_REP:g},\ \eta_2^0={E2_REP:g}$",
            transform=axC.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

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
    if not total_firings:
        axA.set_yticks(e1_values)
        axB.set_yticks(e2_values)
    axC.set(
        title=r"(c)   KKT $\|\nabla L\|$ vs outer iteration",
        xlabel=r"outer iteration  $\tau$",
        ylabel=r"$\|\nabla L\|$",
    )
    axC.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axC.axhline(config["algorithm"]["tol_kkt"], color="k", lw=0.8, ls=":")
    axC.text(
        0.03,
        0.04,
        f"tol = {config['algorithm']['tol_kkt']:.0e}",
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
    axD.set_xticks(sorted(colors))
    axD.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axD.set_xticklabels([f"{nu:g}" for nu in sorted(colors)], rotation=30, ha="right")
    axD.margins(x=0.12, y=0.15)

    handles = [
        Line2D([0], [0], color=c_, lw=2.2, marker=".", label=rf"$\nu={nu_:g}$")
        for nu_, c_ in colors.items()
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
        f"# Descent-firing ablation — NE39 swing OCP (seed {config['seeds'][0]})\n",
        f"{sum(r['converged'] for r in runs)}/{len(runs)} runs converged; "
        f"{total_firings} descent-test failures were recorded. "
        f"Panel (c) uses eta1={E1_REP:g}, eta2={E2_REP:g}. "
        "When no updates fire, panels (a)/(b) show initial values at k=0 and "
        "panel (d) shows zero counts; these runs do not test penalty adaptation.\n",
        "| η1_0 | η2_0 | ν | firings | converges | outer iters | final KKT |",
        "|---:|---:|---:|---:|:--:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda x: (x["nu"], x["eta1_0"], x["eta2_0"])):
        md.append(
            f"| {r['eta1_0']:g} | {r['eta2_0']:g} | {r['nu']:g} "
            f"| {r['n_firings']} | {'✓' if r['converged'] else '✗'} "
            f"| {r['outer_iters']} | {r['final_kkt']:.2e} |"
        )
    with open(os.path.join(OUT, "ablation_eta_results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote", os.path.join(OUT, "ablation_eta_results.md"))

    plt.close(fig)

"""Table 4 from the full configured rate grid."""

import numpy as np


def render(rates, destination, config):
    RB, RE = config["grid"]["varrho_b"], config["grid"]["varrho_eps"]
    seeds = set(config["seeds"])
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{AOTD adaptation-rate ablation $(\varrho_b,\varrho_\epsilon)$ on the NE39 swing OCP ($N=1000$, $M=20$, mean$\pm$stdev over five seeds), using the corrected implementation with $\nu=1.01$. The \# overlap updates column counts failed accuracy-loop passes on which local tolerances and overlaps are updated, summed over all outer iterations. Outer iterations count applied SQP steps. The local tolerance floor is $10^{-9}$ for every setting. FLOPs are shown only if all five runs converge; otherwise --- is shown.}",
        r"\label{tab:rate_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cc r r cc c c}",
        r"\toprule",
        r"$\varrho_b$ & $\varrho_\epsilon$ & \shortstack{$\|\nabla\mathcal{L}\|$\\($\times 10^{-7}$)} & \shortstack{FLOPs\\($\times 10^{7}$)} & \shortstack{\# overlap\\updates} & \shortstack{\# outer\\iterations} & $b_i^{\max}$ & $\epsilon_i^{\min}$ \\",
        r"\midrule",
    ]
    floor = config["algorithm"]["eps_i_floor"]
    exponent = int(np.floor(np.log10(floor)))
    floor_tex = rf"{floor / 10.0**exponent:g}\times10^{{{exponent}}}"
    lines[2] = (
        lines[2]
        .replace("N=1000", f"N={config['problem']['N']}")
        .replace("M=20", f"M={config['M'][0]}")
        .replace("five", str(len(seeds)))
        .replace("1.01", f"{config['algorithm']['nu']:g}")
        .replace(
            r"10^{-9}",
            floor_tex if floor / 10.0**exponent != 1 else rf"10^{{{exponent}}}",
        )
    )
    summaries = []
    for b in RB:
        for e in RE:
            group = [
                r
                for r in rates
                if r["config"]["varrho_b"] == b and r["config"]["varrho_eps"] == e
            ]
            assert len(group) == len(seeds) and {r["seed"] for r in group} == seeds
            stats = [
                [r["kkt"] / 1e-7 for r in group],
                [r["flops"] / 1e7 for r in group],
                [sum(t["n_acc_fail"] for t in r["trajectory"]) for r in group],
                [sum(t["step_applied"] for t in r["trajectory"]) for r in group],
                [max(r["b_list"]) for r in group],
                [min(r["eps_i_list"]) for r in group],
            ]
            cells = [f"${np.mean(v):.1f}\\pm{np.std(v):.1f}$" for v in stats]
            avg, sd = np.mean(stats[-1]), np.std(stats[-1])
            exponent = int(np.floor(np.log10(avg)))
            scale = 10.0**exponent
            cells[-1] = (
                f"$({avg / scale:.1f}\\pm{sd / scale:.1f})\\times10^{{{exponent}}}$"
            )
            n = sum(r["converged"] for r in group)
            if n != len(seeds):
                cells[1] = "---"
            lines.append(f"${b:g}$ & ${e:g}$ & " + " & ".join(cells) + r" \\")
            summaries.append(
                dict(
                    varrho_b=b,
                    varrho_epsilon=e,
                    converged=n,
                    mean_flops=float(np.mean(stats[1]) * 1e7),
                    std_flops=float(np.std(stats[1]) * 1e7),
                )
            )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (destination / "rate_ablation_table.tex").write_text("\n".join(lines) + "\n")
    return summaries

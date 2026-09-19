# Adaptive Overlapping Temporal Decomposition

Corrected IEEE 39-bus Swing experiments for *An Adaptive, Parallel, and Inexact
Newton Method for Large-scale Nonlinear Optimal Control*.

## Install and inspect the configuration

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python experiments/swing/run_main.py --parallel --dry-run
```

Figures need LaTeX, `dvipng`, and Computer Modern fonts. Standalone table PDFs
also need `latexmk` (Ubuntu packages: `latexmk texlive-latex-extra
texlive-fonts-recommended cm-super dvipng`).

## Three studies

| Config | Runner in `experiments/swing/` | Scenarios |
|---|---|---:|
| `configs/swing.json` | `run_main.py` | 285: 40 AOTD, 120 FOTD, 125 nonlinear baselines |
| `configs/rate_ablation.json` | `run_rate_ablation.py` | 80: 16 rate pairs × 5 seeds |
| `configs/eta_ablation.json` | `run_eta_ablation.py` | 36: 9 penalty initializations × 4 ν values |

Each config contains the complete problem and solver settings. The main and
rate studies use `(varrho_b, varrho_eps)=(32,0.5)` as their reference. Figure 7
uses `(32,0.6)`, beta=0.0001, eta1={12,13,14}, eta2={0.15,0.20,0.25}, and
nu={1.075,1.1,1.15,1.2}. This is the previously successful penalty study;
the separate historical `(32,0.5)` penalty diagnostic includes failures.

The rate grid is varrho_b={1,2,4,32}, varrho_eps={0.1,0.25,0.5,1}.
All studies use N=1000. Main/rate seeds are 1–5; eta uses seed 1.
FOTD retains its own explicit penalty settings for comparability.

Choose one execution mode:

```bash
# Throughput: independent scenarios in separate processes; no reportable timings.
python experiments/swing/run_main.py --parallel --workers 4
python experiments/swing/run_rate_ablation.py --parallel --workers 4
python experiments/swing/run_eta_ablation.py --parallel --workers 4

# Timing: sequential scenarios, one BLAS thread. Run on an otherwise idle machine.
python experiments/swing/run_main.py --uncontended
```

All three runners accept both modes. `--parallel` parallelizes **scenarios**,
not the SQP subproblems. `--uncontended` excludes other running studies launched
by these runners; it cannot control unrelated machine load. Timings include
logging overhead. Estimated parallel wall time uses the critical path of local
solves plus serial phases; it is not measured distributed execution.

Defaults write `results/production/<study>/<mode>/`. Use `--output` for another
directory inside this dev repository's `results/`. `--dry-run` lists scenarios;
`--limit 2` runs a small subset, explicitly marked partial. Figure/table scripts
reject partial datasets. Convergence failures remain in a complete study;
exceptions instead mark the study failed and produce a nonzero exit status.

Use `--resume` to reuse verified results. Identity includes the full scenario,
problem, solver source hash, complete study config, execution environment, and execution mode.
Unsuccessful numerical solves are retained; exceptions are retried. Parallel jobs
are submitted in bounded batches, so interruption does not drain the entire queued sweep. A changed
configuration or source requires a new output directory. Contended results
cannot be reused as uncontended timings.

## Live logging

Every output directory contains:

- `progress.log`: scenario START/CACHED/DONE and completed/total counts.
- `logs/<scenario>.log`: configuration/source hashes, execution mode, each outer
  KKT/feasibility residual, epsilon clipping, every accuracy/descent pass,
  overlap/tolerance updates, line-search outcome, and final work/timing summary.
- `records/<scenario>.json`: detailed numerical trace and acceptance certificates.
- `manifest.json`: atomic progress snapshots, config, environment, all results,
  convergence count, and completion status.

Python and native Ipopt output are captured per scenario. Schwarz, ADMM, and
multiple shooting log outer KKT residuals; IPOPT logs its native per-iteration
primal/dual diagnostics and the final independently evaluated KKT residual.

```bash
tail -f results/production/swing/parallel/progress.log
# Pick a scenario filename printed by START:
tail -f results/production/swing/parallel/logs/<scenario>.log
```

`outer_iters` counts applied SQP steps; `outer_checks` also includes the terminal
KKT check. This avoids counting a terminal check as an applied step.

## Seven artifact generators

Each script renders exactly one figure or table from a complete manifest.
Table scripts produce both LaTeX fragments and standalone numbered PDFs;
`--tex-only` skips PDF compilation. Figures retain the agreed scales, labels,
seed bands, and legend placement. Output defaults to `dev_figures/production/`;
`--output` may select another folder inside `dev_figures/`.

```bash
python result_scripts/swing_highlight_table.py --input results/production/swing/parallel
python result_scripts/swing_appendix_table.py --input results/production/swing/uncontended
python result_scripts/swing_adaptation.py --input results/production/swing/parallel
python result_scripts/swing_convergence.py --input results/production/swing/parallel
python result_scripts/rate_ablation.py --input results/production/rate_ablation/parallel
python result_scripts/ablation_eta_init.py --input results/production/eta_ablation/parallel
python result_scripts/rate_ablation_table.py --input results/production/rate_ablation/parallel
```

The timing table **requires uncontended data**. Work and timing table cells show
`---` unless every seed converges. Cost boxplots retain unsuccessful outcomes
with crosses; the M=50 FOTD boxes remain omitted as requested, with those results
still present in tables and raw records. Each artifact has a provenance JSON
sidecar. Original manuscript files and figures are never output targets.

## Numerical contract

The Newton system uses the full Lagrangian Hessian with a stagewise eigenvalue
floor. The merit derivative always uses the unshifted true Hessian. Local
solves must satisfy the original, unpreconditioned residual criterion, including
warm starts and sketch solves. Certification work remains included in FLOPs.

There is one overlap/tolerance update law: exponential tightening with the
uniform `1/sqrt(M)` term and overlap growth of at least one stage. Removed options
include relaxed accuracy multipliers, Gauss–Newton direction selection,
alternative update laws, EW forcing, and theta schedules. Unknown options fail
at configuration loading. Epsilon clips to `epsilon_max/nu` only above the cap;
the cap includes `(Psi*Upsilon)^2` and cannot be enlarged by a numerical floor.

**Psi=Upsilon=1 are experimental constants**, not computed theoretical bounds.
The overlap cap (110), local tolerance floor (1e-9), 50-pass acceptance budget,
100-iteration local budget, and 25-iteration outer budget are explicit practical
limits. Budget exhaustion or failed acceptance stops without applying a step.
These experiments do not establish the manuscript's theoretical bounds merely
by converging.

## Layout and historical material

Shared runner/configuration/reporting code lives in `src/newton/studies/`;
solver code lives in `src/newton/`. There are exactly three IEEE39 runners,
three study configs, and seven artifact entry points. The production checkout
excludes `data/`, `tests/`, `dev_figures/`, and `results/`; the runners and
renderers create their output directories as needed.

Historical datasets, validation tests, and generated artifacts may be retained
locally in the ignored directories. They are not required to run a fresh study.
Old exploratory Swing/rate/eta entry points and the multipurpose `experiments.py`
dispatcher were removed; historical reports may reference those retired scripts.

Burgers entry points remain under `experiments/burgers/`. They now use the strict
core and invalidate earlier cached solver records; their old numerical results
have not been revalidated by this IEEE39 cleanup.

Code is released under the [MIT License](LICENSE). University logos retain
separate ownership; see [asset sources](assets/README.md).

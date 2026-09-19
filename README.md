<p align="center">
  <a href="https://ucsd.edu/"><img src="assets/uc-san-diego.png" alt="UC San Diego" width="200"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.gatech.edu/"><img src="assets/georgia-tech.png" alt="Georgia Tech" width="180"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.berkeley.edu/"><img src="assets/uc-berkeley.svg" alt="UC Berkeley" width="200"></a>
</p>

<h1 align="center">Adaptive Overlapping Temporal Decomposition (AOTD)</h1> 

![Burgers PDE results: overlap and tolerance adaptation, with computational cost across temporal decompositions.](assets/burgers-results.png)

*Burgers PDE control at 10,000 and 50,000 time steps: adaptive overlaps and
local solve tolerances (left, middle), and computational work across methods
(right). Figure 5, using the archived five-seed experiments.*

## About this repository

This repository contains all the code for reproducing the experiments in the paper titled: **An Adaptive, Parallel, and Inexact Newton
Method for Large-scale Nonlinear Optimal Control**. For any issues, or questions please make a Github issue or contact the authors at lbhan@ucsd.edu.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python experiments.py smoke
```

Plotting also requires LaTeX. On Debian/Ubuntu:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended cm-super dvipng
```

## Generate data

```bash
python experiments.py run --study swing --workers 1
```

Studies: `swing`, `burgers`, `rates`, `globalization`, or `all`. Data goes to
`results/runs/<study>/`; use `--output` to change the root. Settings are in
`experiments/<study>/run_*.py`. Burgers runs can take hours and tens of GB of RAM.

## Figures and tables

After generating all four studies:

```bash
python experiments.py plot --input results/runs --compile-tables
```

Or use the bundled data without running the solvers:

```bash
python experiments.py plot --archived --compile-tables
```

Outputs go to `results/paper/`. Bundled records are in `data/*.json.gz`.
Plots cover Figures 2–7 and Tables 1, 2, 4, 5, 6; Figure 1 and Table 3 are
manuscript-only material, not experimental outputs.
New runs use a 100-iteration inner cap. Use a fresh output root after changing
settings. Bundled data represents the earlier implementation and is retained
for historical comparisons; it does not validate the corrected solver.

### Corrected Hessian and residual checks

Swing now uses the **full Lagrangian Hessian** with a stagewise eigenvalue
floor of `1e-6` for the Newton system. The true, unshifted Hessian differentiates
the augmented merit in both the descent test and Armijo slope. Gauss–Newton
remains an optional direction model; it never supplies the merit derivative.

Local convergence requires `norm(Gamma_i @ d_i - rhs_i) <= eps_i * norm(rhs_i)`
in the original coordinates, including for warm starts and the sketch solver.
Residual checks count toward work. Failed local solves, acceptance budgets,
and line searches stop without applying the rejected direction. Swing keeps
50 accuracy/descent passes, local iteration cap 100, and overlap cap 110.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python experiments.py smoke
python experiments/swing/run_corrected_study.py --workers 4
```

The corrected study reruns all 160 Swing Newton-method cases (120 fixed and
40 adaptive), and reuses only the unchanged nonlinear baselines. Outputs and
a full acceptance audit are in `results/corrected_swing/`; start with
`report.md`. Sketch RNG seed zero is reset per case, independently of scheduling.
The smoke check validates a converged corrected case and its local/global
certificates, rather than comparing work against the old implementation.

At these budgets, each adaptive inner solver converges on 14/20 cases:
5/5 for M=4 and M=10, 3/5 for M=20, and 1/5 for M=50. The other cases stop at
the acceptance-pass budget. No applied step violates its required local or
adaptive global checks. Every recorded Hessian shift is zero in this Swing
study; a vanishing modification is not guaranteed for general problems by a
stagewise positive-definiteness rule.

The prior cap-only experiment is preserved in `results/cap_study/`: changing
six passes to 50 under the old implementation reduced ungated steps from
77/188 to 15/179 but did not eliminate them. That historical experiment and
its `run_pass_cap_study.py` driver require the pre-correction solver source.

Solver code is in `src/newton/`. `python experiments.py --help`
lists all commands, including verification and a quick solver check.

## Citation

Placeholder—publication details will be added before the final release.

```bibtex
@misc{bhan_aotd,
  author = {Bhan, Luke and Mahoney, Michael W. and Na, Sen},
  title = {An Adaptive, Parallel, and Inexact Newton Method for
           Large-scale Nonlinear Optimal Control},
  note = {Citation placeholder: venue, year, and DOI to be added}
}
```

## License

Code is released under the [MIT License](LICENSE). University logos remain the
property of their respective institutions and are not covered by this license;
their display does not imply endorsement. See [asset sources](assets/README.md).

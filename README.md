# AOTD

Python solvers and experiments for **An Adaptive, Parallel, and Inexact Newton
Method for Large-scale Nonlinear Optimal Control**. Generate your own results or
use the bundled paper data.

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
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
New runs use a 100-iteration inner cap; archived runs used 120. Use a fresh output
root after changing settings to avoid reusing older records.

Solver code is in `src/newton/`. `python experiments.py --help`
lists all commands, including verification and a quick solver check.

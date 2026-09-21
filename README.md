# Adaptive Overlapping Temporal Decomposition

Reference implementation and reproducible experiments for *An Adaptive,
Parallel, and Inexact Newton Method for Large-scale Nonlinear Optimal Control*.

<p align="center">
  <img src="assets/uc-san-diego.png" alt="UC San Diego" width="180">&nbsp;&nbsp;&nbsp;
  <img src="assets/georgia-tech.png" alt="Georgia Tech" width="180">&nbsp;&nbsp;&nbsp;
  <img src="assets/uc-berkeley.svg" alt="UC Berkeley" width="180">
</p>

![Burgers convergence results for AOTD, FOTD (LU), and Schwarz](assets/burgers-results.png)

<p align="center"><sub>Convergence on the viscous Burgers optimal control problem for horizons N = 10,000 and 50,000, with M = 10, 50, and 250 temporal subproblems. Curves compare AOTD, FOTD (LU) with fixed overlaps b = 10 and 50, and Schwarz with b = 50 (N = 10,000 only). Lower KKT residuals indicate closer satisfaction of the optimality conditions; the dotted line marks the convergence tolerance.</sub></p>

## Installation

The reference results were generated with Python 3.10; numerical dependencies
are pinned in `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Reference data

The four compressed manifests in `data/reference/burgers/` and
`data/reference/ieee39/` contain all 596 measured scenarios, configurations, and
solver trajectories.

## Rerun the experiments

Use either `--parallel X` to run up to X independent scenario processes or
`--uncontended` to run one scenario at a time. Every scenario uses one BLAS
thread. The suite runs Swing, the rate ablation, the eta ablation, then Burgers.
These are full research workloads; memory usage grows with the number of workers.

```bash
# Throughput run using four scenario processes.
python experiments/run_all.py --parallel 4

# Sequential run for timing comparisons.
python experiments/run_all.py --uncontended
``

`--uncontended` takes an exclusive repository-wide study lock and runs scenarios
sequentially. It cannot stop unrelated applications or other users' jobs; use
an otherwise idle machine for timing measurements. Timing values vary with
hardware, operating system, and numerical libraries.

Each study can also be run independently:

```bash
python experiments/ieee39/run_main.py --parallel 4
python experiments/ieee39/run_rate_ablation.py --parallel 4
python experiments/ieee39/run_eta_ablation.py --parallel 4
python experiments/burgers/run_main.py --parallel 4
```

Replace `--parallel 4` with `--uncontended` for any of these commands.

### Inspect, resume, or customize a run

```bash
python experiments/run_all.py --parallel 4 --dry-run
python experiments/run_all.py --parallel 4 --resume
python experiments/ieee39/run_main.py --uncontended --limit 2 --output runs/swing-check
python experiments/burgers/run_main.py --config configs/burgers/main.json --parallel 2 --output runs/burgers-custom
```

All runners support `--dry-run`, `--resume`, `--limit`, and `--output`.
Individual runners also support `--config`. The suite's `--output` sets a root
beneath which it creates the study/mode directories; an individual runner's
`--output` names the exact study directory. .

## Repository layout

- `src/newton/`: solvers, models, baselines, and study infrastructure
- `configs/`: experiment definitions
- `experiments/`: individual and full-suite experiment commands
- `data/reference/`: compressed, complete published measurements
- `runs/`: newly generated runs, excluded from version control
- `assets/`: static project assets

The four compressed manifests under `data/reference/burgers/` and
`data/reference/ieee39/` contain all measurements and their original provenance.

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

Code is released under the [MIT License](LICENSE).

# Adaptive Overlapping Temporal Decomposition

Reference implementation and reproducible experiments for *An Adaptive,
Parallel, and Inexact Newton Method for Large-scale Nonlinear Optimal Control*.

## Installation

Run these commands from the repository root on Linux with Python 3.10 or newer.
The reference results were generated with Python 3.10; numerical dependencies
are pinned in `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Reference data

The four compressed manifests in `data/reference/burgers/` and
`data/reference/ieee39/` contain all 596 measured scenarios, configurations,
solver trajectories, and original execution provenance. They can be read as
JSON using Python's standard-library `gzip` and `json` modules.

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
```

Parallelism distributes independent scenarios; it does not execute the temporal
subproblems of one solve on separate processors. Parallel-mode elapsed times
are diagnostic. The recorded `par_wall` value is an idealized critical-path
estimate, not a measured distributed runtime.

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

| Study | Scenarios | Default output under `runs/` |
| --- | ---: | --- |
| IEEE39 Swing | 285 | `ieee39/swing/<mode>/` |
| Adaptation-rate ablation | 80 | `ieee39/rate_ablation/<mode>/` |
| Initial-penalty ablation | 36 | `ieee39/eta_ablation/<mode>/` |
| Burgers | 195 | `burgers/<mode>/` |

Definitions live in `configs/`. AOTD starts at `b0=10` in every study.
FOTD uses both LU and GMRES–QLP, with fixed overlaps `b=[10,50,75]` for
Swing and `b=[10,50]` for each Burgers horizon `N=[10000,50000]`.
Both benchmark studies use five seeds. The rate ablation has 16 settings
with five seeds; the eta ablation has 36 parameter combinations with one seed.

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
`--output` names the exact study directory. All new output must be inside
`runs/`, protecting the published reference measurements.

Each run writes a manifest, progress log, per-scenario logs, and JSON records.
Resume reuses only verified records with matching configuration, source,
environment, and execution mode, including recorded non-convergence. Changed
inputs require a new output directory. A limited run is marked partial and
can be completed by resuming without `--limit`. Do not point resume at the
reference data: its original source and environment identities are preserved.

## Repository layout

- `src/newton/`: solvers, models, baselines, and study infrastructure
- `configs/`: experiment definitions
- `experiments/`: individual and full-suite experiment commands
- `data/reference/`: compressed, complete published measurements
- `runs/`: newly generated runs, excluded from version control
- `assets/`: static project assets

The four compressed manifests under `data/reference/burgers/` and
`data/reference/ieee39/` contain all measurements and their original provenance.
Generated artifacts, local run output, and historical archives are excluded from Git. Code is released under the [MIT License](LICENSE).
University marks retain separate ownership.

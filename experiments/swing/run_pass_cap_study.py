#!/usr/bin/env python3
"""Paired Swing runs at caps 6 and 50, retaining archived fixed baselines.

Each case uses a fresh Python process, hence the sketch RNG starts at zero
independently of job ordering. Outputs include all passes, not just successes.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = ("AOTD-rebuild", "AOTD-sketch")
MS = (4, 10, 20, 50)


def one(args):
    cap, method, m, seed = args.one
    cap, m, seed = int(cap), int(m), int(seed)
    work = args.output.resolve() / f"cap{cap}" / "swing"
    os.environ["AOTD_WORK"] = str(work)
    os.environ["AOTD_MAX_PASSES"] = str(cap)
    name = "run_sketch.py" if method == "AOTD-sketch" else "run_comparison.py"
    spec = importlib.util.spec_from_file_location("swing_runner", HERE / name)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if method == "AOTD-sketch":
        row = runner.run_one((m, seed))
    else:
        row = runner.run_one((method, None, m, seed))
    if "error" in row:
        raise RuntimeError(row["tb"])
    steps = row["trajectory"]
    bad = sum(t["passes"][-1]["outcome"] != "accept" for t in steps)
    print(f"cap={cap:2} {method:12} M={m:2} seed={seed} "
          f"KKT={row['kkt']:.3e} work={row['flops']:.3e} "
          f"steps={len(steps)} unaccepted={bad} "
          f"max_passes={max(len(t['passes']) for t in steps)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cap_study")
    parser.add_argument("--one", nargs=4, metavar=("CAP", "METHOD", "M", "SEED"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    # This frozen comparison intentionally studied the old cap fallback. Do
    # not replace its historical records with a different algorithm silently.
    expected = "b404120fc1af921de7e89256841d24bc92062cdf534b963b5b0e154e19434011"
    actual = hashlib.sha256((ROOT / 'src/newton/AOTD.py').read_bytes()).hexdigest()
    if actual != expected:
        parser.error('This historical cap-only study requires the pre-correction solver. '
                     'Use run_corrected_study.py for the current solver; existing cap-study data is preserved.')
    if args.one:
        one(args)
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / "data/swing.json.gz", "rt") as stream:
        archive = json.load(stream)
    provenance = {
        "caps": [6, 50], "M": list(MS), "seeds": [1, 2, 3, 4, 5],
        "N": 1000, "max_inner_iters": 100, "max_outer_iters": 25,
        "sketch_rng_seed_per_run": 0, "workers": 1,
        "adaptive_source": "fresh runs; one Python process per case",
        "baseline_source": "data/swing.json.gz (unchanged archived baselines)",
        "note": "Only max_inner_passes changes between paired adaptive runs. "
                "The original fallback after exhaustion is retained to measure it. "
                "Fresh sketch streams are controlled per case and need not match "
                "the archived worker-dependent random stream.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for cap in (6, 50):
        work = out / f"cap{cap}" / "swing"
        (work / "records").mkdir(parents=True, exist_ok=True)
        for name, value in archive.items():
            if name == "baselines_sigma06.json" or (
                name.startswith("records/") and value.get("method") not in METHODS
            ):
                (work / name).write_text(json.dumps(value))
        for method in METHODS:
            for m in MS:
                for seed in range(1, 6):
                    subprocess.run([
                        sys.executable, str(Path(__file__).resolve()),
                        "--output", str(out), "--one", str(cap), method,
                        str(m), str(seed),
                    ], check=True)
        rows = [json.loads(p.read_text()) for p in sorted((work / "records").glob("*.json"))]
        if len(rows) != 160 or any("error" in r for r in rows):
            raise RuntimeError("Incomplete study records")
        (work / "tight_rank_eps_sweep.json").write_text(json.dumps(
            [r for r in rows if r["method"] != "AOTD-sketch"]
        ))
        env = dict(os.environ, AOTD_WORK=str(work), MPLBACKEND="Agg")
        subprocess.run([sys.executable, str(HERE / "plot_results.py")],
                       env=env, check=True)
    subprocess.run([sys.executable, str(HERE / "summarize_pass_cap_study.py"),
                    "--input", str(out)], check=True)


if __name__ == "__main__":
    main()

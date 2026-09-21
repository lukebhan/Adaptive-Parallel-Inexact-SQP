#!/usr/bin/env python3
"""Run Swing, both ablations, and Burgers in sequence."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from newton.studies.common import add_execution_arguments, worker_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_execution_arguments(parser)
    args = parser.parse_args()
    workers = worker_count(parser, args)
    output = (args.output or ROOT / "runs").resolve()
    if not output.is_relative_to(ROOT / "runs"):
        parser.error("Output must be inside the repository runs/ directory")
    mode = "uncontended" if args.uncontended else "parallel"
    studies = [
        ("ieee39/run_main.py", "ieee39/swing"),
        ("ieee39/run_rate_ablation.py", "ieee39/rate_ablation"),
        ("ieee39/run_eta_ablation.py", "ieee39/eta_ablation"),
        ("burgers/run_main.py", "burgers"),
    ]
    for script, directory in studies:
        command = [sys.executable, str(ROOT / "experiments" / script)]
        command += (
            ["--uncontended"] if args.uncontended else ["--parallel", str(workers)]
        )
        command += ["--output", str(output / directory / mode)]
        for flag in ("resume", "dry_run"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        print(f"Running {directory} ({mode})", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

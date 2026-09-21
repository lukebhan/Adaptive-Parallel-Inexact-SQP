#!/usr/bin/env python3
"""Run the configured IEEE39 swing study; use --help for execution modes."""

import os
from pathlib import Path
import sys

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from newton.studies.ieee39.runner import main

if __name__ == "__main__":
    main("swing")

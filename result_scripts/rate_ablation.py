#!/usr/bin/env python3
"""Render rate_ablation from a complete experiment manifest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from newton.studies.artifacts import main

if __name__ == "__main__":
    main("rate_ablation")

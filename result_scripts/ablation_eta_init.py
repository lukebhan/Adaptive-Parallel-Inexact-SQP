#!/usr/bin/env python3
"""Render ablation_eta_init from a complete experiment manifest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from newton.studies.artifacts import main

if __name__ == "__main__":
    main("ablation_eta_init")

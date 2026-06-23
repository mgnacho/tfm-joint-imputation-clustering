#!/usr/bin/env python
"""Convenience entry point for the UCI Wholesale Customers experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from joint_imputation_clustering.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

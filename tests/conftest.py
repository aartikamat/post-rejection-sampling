"""Test fixtures shared across the prfs test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the implementation package is importable when pytest is invoked
# from the IMPLEMENTATION directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"


@pytest.fixture(scope="session")
def registry_csv() -> Path:
    p = DATA_DIR / "rejections.csv"
    if not p.exists():
        pytest.skip(f"deposit CSV not present at {p}")
    return p


@pytest.fixture(scope="session")
def outcomes_csv() -> Path:
    p = DATA_DIR / "rejection_outcomes.csv"
    if not p.exists():
        pytest.skip(f"deposit CSV not present at {p}")
    return p

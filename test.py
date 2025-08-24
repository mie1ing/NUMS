"""Visualization utilities for Rayleigh-Bénard simulations.

This script gathers multiple saved simulation cases (``Ra_XXXX.npz``) and
generates four kinds of visualisations:

1. Temperature fields for each Rayleigh number on a single figure.
2. Velocity fields (contours of speed with quiver arrows) on a single figure.
3. Convergence histories of the Nusselt number for all cases.
4. A Nusselt--Rayleigh relation plot compared against a simple theoretical
   scaling ``Nu = 1 + 0.15 Ra^{1/3}``.

The intent is to compare four simulations with different Rayleigh numbers.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from grid import Grid2D
from operators import FluidOperators


@dataclass
class CaseData:
    """Container for a single Rayleigh--Bénard simulation snapshot."""

    Ra: float
    grid: Grid2D
    u: np.ndarray
    w: np.ndarray
    T: np.ndarray
    Nu_history: np.ndarray
    dt: float


def load_case(filename: str):
    """Load a simulation case stored in ``npz`` format."""

    data = np.load(filename)

    # Attempt to obtain Ra value from file if stored, otherwise parse from name
    if "Ra" in data:
        print("Ra found in file")
    else:
        print("Ra not found in file")


if __name__ == "__main__":
    # By default look for four files named ``Ra_XXXX.npz`` in ``rb_data``.
    load_case("rb_data/Ra_1000.npz")


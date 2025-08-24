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


def load_case(filename: str) -> CaseData:
    """Load a simulation case stored in ``npz`` format."""

    data = np.load(filename)
    nx = int(data["nx"])
    nz = int(data["nz"])
    Lx = float(data["Lx"])
    Lz = float(data["Lz"])
    grid = Grid2D(nx=nx, nz=nz, Lx=Lx, Lz=Lz, staggered=True)

    u = data["u"]
    w = data["w"]
    T = data["T"]
    Nu_history = data["Nu_history"]
    dt = float(data["dt"])

    match = re.search(r"Ra_(\d+)", filename)
    Ra = float(match.group(1)) if match else float("nan")

    return CaseData(Ra=Ra, grid=grid, u=u, w=w, T=T, Nu_history=Nu_history, dt=dt)


def plot_velocity_field(case: CaseData, stride: int = 2, ax: plt.Axes | None = None) -> None:
    """Plot velocity magnitude contours with quiver arrows."""

    grid = case.grid
    ops = FluidOperators(grid)
    u_p, w_p = ops.interpolate_to_pressure_points(case.u, case.w)
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)
    magnitude = np.sqrt(u_p ** 2 + w_p ** 2)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    im = ax.contourf(X, Z, magnitude, levels=10, cmap="viridis")
    skip = max(1, stride)
    if np.max(magnitude) > 0:
        ax.quiver(
            X[::skip, ::skip],
            Z[::skip, ::skip],
            u_p[::skip, ::skip],
            w_p[::skip, ::skip],
            scale=None,
            color="white",
        )
    ax.set_aspect("equal")
    ax.set_title(f"Ra = {case.Ra:g}")
    plt.colorbar(im, ax=ax)


def plot_temperature_field(case: CaseData, ax: plt.Axes | None = None) -> None:
    """Plot temperature field for a single case."""

    grid = case.grid
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    im = ax.contourf(X, Z, case.T, levels=20, cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_title(f"Ra = {case.Ra:g}")
    plt.colorbar(im, ax=ax)


def plot_cases(file_pattern: str = "rb_data/Ra_*.npz", stride: int = 2) -> None:
    """Load multiple cases and generate comparison figures."""

    filenames = sorted(glob.glob(file_pattern), key=lambda s: float(re.search(r"Ra_(\d+)", s).group(1)))
    cases = [load_case(fn) for fn in filenames]

    if not cases:
        raise FileNotFoundError(f"No files matched pattern {file_pattern}")

    # --- Temperature fields ---
    fig_T, axes_T = plt.subplots(2, 2, figsize=(10, 8))
    for ax, case in zip(axes_T.flat, cases):
        plot_temperature_field(case, ax=ax)
    fig_T.suptitle("Temperature Fields")
    plt.tight_layout()
    plt.savefig("plots/rb_staggered_T.png")

    # --- Velocity fields ---
    fig_V, axes_V = plt.subplots(2, 2, figsize=(10, 8))
    for ax, case in zip(axes_V.flat, cases):
        plot_velocity_field(case, stride=stride, ax=ax)
    fig_V.suptitle("Velocity Fields")
    plt.tight_layout()
    plt.savefig("plots/rb_staggered_V.png")

    # --- Convergence history ---
    fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
    for case in cases:
        times = np.arange(len(case.Nu_history)) * case.dt * 1000
        ax_conv.plot(times, case.Nu_history, label=f"Ra = {case.Ra:g}")
    ax_conv.set_xlabel("Time")
    ax_conv.set_ylabel("Nusselt number")
    ax_conv.set_title("Convergence Histories")
    ax_conv.legend()
    ax_conv.grid(True)
    plt.savefig("plots/rb_staggered_conv.png")

    # --- Nu-Ra relation ---
    fig_NR, ax_NR = plt.subplots(figsize=(6, 4))
    Ra_vals = np.array([c.Ra for c in cases])
    Nu_vals = np.array([c.Nu_history[-1] for c in cases])
    ax_NR.plot(Ra_vals, Nu_vals, "o-", label="Simulation")

    # theoretical scaling: Nu = 1 + 0.15 * Ra^{1/3}
    Ra_theory = np.logspace(np.log10(Ra_vals.min()), np.log10(Ra_vals.max()), 100)
    Nu_theory = 1 + 0.15 * Ra_theory ** (1 / 3)
    ax_NR.plot(Ra_theory, Nu_theory, "--", label="Theory")

    ax_NR.set_xscale("log")
    ax_NR.set_yscale("log")
    ax_NR.set_xlabel("Ra")
    ax_NR.set_ylabel("Nu")
    ax_NR.set_title("Nu-Ra Relation")
    ax_NR.legend()
    ax_NR.grid(True, which="both")
    plt.savefig("plots/rb_staggered_NR.png")

    plt.show()


if __name__ == "__main__":
    # By default look for four files named ``Ra_XXXX.npz`` in ``rb_data``.
    plot_cases("rb_data/Ra_*.npz", stride=3)


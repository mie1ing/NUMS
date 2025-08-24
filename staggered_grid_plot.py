from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from grid import Grid2D


def plot_staggered_grid(
    nx: int = 8,
    nz: int = 4,
    Lx: float = 2.0,
    Lz: float = 1.0,
    filename: str | None = "plots/staggered_grid.png",
) -> plt.Axes:
    """Plot a schematic of the staggered grid layout.

    Parameters
    ----------
    nx, nz:
        Number of cells in the x and z directions.
    Lx, Lz:
        Domain size in the x and z directions.
    filename:
        If provided, path where the figure will be saved.
    """

    grid = Grid2D(nx=nx, nz=nz, Lx=Lx, Lz=Lz, staggered=True)
    x_p, z_p = grid.get_pressure_grid()
    x_u, z_u = grid.get_u_grid()
    x_w, z_w = grid.get_w_grid()

    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw grid lines for pressure/temperature points
    for x in x_p:
        ax.plot([x, x], [z_w[0], z_w[-1]], color="black", linewidth=0.5)
    for z in z_p:
        ax.plot([x_u[0], x_u[-1]], [z, z], color="black", linewidth=0.5)

    # Pressure/temperature points (at cell corners)
    Xp, Zp = np.meshgrid(x_p, z_p)
    ax.scatter(Xp, Zp, color="C0", label="p, T", zorder=3)

    # u-velocity points (vertical face centers)
    Xu, Zu = np.meshgrid(x_u, z_u)
    ax.scatter(Xu, Zu, color="C1", marker="^", label="u", zorder=3)

    # w-velocity points (horizontal face centers)
    Xw, Zw = np.meshgrid(x_w, z_w)
    ax.scatter(Xw, Zw, color="C2", marker=">", label="w", zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Staggered Grid")

    if filename:
        fig.savefig(filename, bbox_inches="tight")

    return ax


if __name__ == "__main__":
    plot_staggered_grid()
    plt.show()
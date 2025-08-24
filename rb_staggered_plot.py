import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators


def plot_velocity_field(u, w, grid, stride=1, ax=None, title=None):
    """Plot velocity vectors on the pressure grid."""
    ops = FluidOperators(grid)
    u_p, w_p = ops.interpolate_to_pressure_points(u, w)
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.quiver(
        X[::stride, ::stride],
        Z[::stride, ::stride],
        u_p[::stride, ::stride],
        w_p[::stride, ::stride],
        scale_units='xy',
        angles='xy',
    )
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    return ax


def plot_saved_case(filename, stride=2):
    """Load saved simulation data and visualize temperature, velocity and Nu history."""
    data = np.load(filename)
    nx = int(data['nx'])
    nz = int(data['nz'])
    Lx = float(data['Lx'])
    Lz = float(data['Lz'])
    grid = Grid2D(nx=nx, nz=nz, Lx=Lx, Lz=Lz, staggered=True)
    u = data['u']
    w = data['w']
    T = data['T']
    Nu_history = data['Nu_history']
    dt = float(data['dt'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)
    im = axes[0].contourf(X, Z, T, levels=20, cmap='RdBu_r')
    axes[0].set_title('Temperature')
    axes[0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0])

    plot_velocity_field(u, w, grid, stride=stride, ax=axes[1], title='Velocity')

    times = np.arange(len(Nu_history)) * 1000 * dt
    axes[2].plot(times, Nu_history)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Nusselt number')
    axes[2].set_title('Nu history')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot saved RB simulation data')
    parser.add_argument('file', help='npz file produced by run_study')
    parser.add_argument('--stride', type=int, default=2, help='quiver stride')
    args = parser.parse_args()
    plot_saved_case(args.file, stride=args.stride)

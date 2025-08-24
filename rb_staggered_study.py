import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators


def simulate_rb(Ra, n_steps=5000, grid=None, dt=None, Lx=2.0, Lz=1.0,
                verbose=False):
    """Run a Rayleigh-Bénard simulation for a given Rayleigh number.

    Parameters
    ----------
    Ra : float
        Target Rayleigh number.
    n_steps : int
        Number of time steps to integrate.
    grid : Grid2D, optional
        Pre-built grid.  If ``None`` a small staggered grid is created.
    dt : float, optional
        Time step.  When ``None`` a stable step is estimated automatically.
    Lx, Lz : float
        Domain size.
    verbose : bool
        When True print progress information every 1000 steps.

    Returns
    -------
    dict containing final fields, histories and diagnostics.
    """
    if grid is None:
        grid = Grid2D(nx=32, nz=16, Lx=Lx, Lz=Lz, staggered=True)
    ops = FluidOperators(grid)

    # Physical parameters
    g = 9.81
    alpha = 1e-3
    Delta_T = 1.0
    # choose nu=kappa for simplicity
    nu = kappa = np.sqrt((g * alpha * Delta_T * grid.Lz ** 3) / Ra)

    if dt is None:
        dt = 0.25 * min(grid.dx, grid.dz) ** 2 / max(nu, kappa)

    # Create staggered fields
    u = grid.create_field('u_velocity')
    w = grid.create_field('w_velocity')
    T = grid.create_field('pressure')

    # Initial temperature linear profile with perturbation
    x_p, z_p = grid.get_pressure_grid()
    T_hot, T_cold = 0.5, -0.5
    for j in range(grid.nz + 1):
        T_linear = T_cold + (T_hot - T_cold) * (1 - z_p[j] / grid.Lz)
        for i in range(grid.nx + 1):
            pert = 0.1 * np.sin(np.pi * x_p[i] / grid.Lx) * \
                   np.sin(np.pi * z_p[j] / grid.Lz)
            T[j, i] = T_linear + pert

    def apply_bc(u, w, T):
        u[0, :] = u[-1, :] = 0
        u[:, 0] = u[:, -1] = 0
        w[0, :] = w[-1, :] = 0
        w[:, 0] = w[:, -1] = 0
        T[0, :] = T_hot
        T[-1, :] = T_cold
        T[:, 0] = T[:, 1]
        T[:, -1] = T[:, -2]

    apply_bc(u, w, T)

    def compute_Nu(T, u, w):
        dT_dz = ops.d_dz(T)
        w_at_p = 0.5 * (w[:-1, :] + w[1:, :])
        heat_flux = -kappa * dT_dz + w_at_p * T
        return grid.Lz * np.mean(heat_flux) / (kappa * (T_hot - T_cold))

    def solve_poisson(rhs, max_iter=200):
        phi = np.zeros_like(rhs)
        rhs = rhs - np.mean(rhs)
        for _ in range(max_iter):
            phi_old = phi.copy()
            for j in range(1, phi.shape[0] - 1):
                for i in range(1, phi.shape[1] - 1):
                    phi[j, i] = 0.25 * (
                        phi[j + 1, i] + phi[j - 1, i] +
                        phi[j, i + 1] + phi[j, i - 1] -
                        grid.dx ** 2 * rhs[j, i]
                    )
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]
            phi -= np.mean(phi)
            if np.max(np.abs(phi - phi_old)) < 1e-6:
                break
        return phi

    Nu_history = [compute_Nu(T, u, w)]
    max_vel_hist = []

    for step in range(n_steps):
        T_ref = 0.5 * (T_hot + T_cold)
        T_w = 0.5 * (T[:-1, :] + T[1:, :])
        buoyancy = np.zeros_like(w)
        buoyancy[1:-1, :] = g * alpha * (T_w - T_ref)

        du_dx = ops.d_dx(u)
        du_dz = ops.d_dz(u)
        dw_dx = ops.d_dx(w)
        dw_dz = ops.d_dz(w)

        w_on_u = np.zeros_like(u)
        w_on_u[:, 1:-1] = 0.25 * (
            w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:]
        )
        u_on_w = np.zeros_like(w)
        u_on_w[1:-1, :] = 0.25 * (
            u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:]
        )

        conv_u = u * du_dx + w_on_u * du_dz
        conv_w = u_on_w * dw_dx + w * dw_dz

        visc_u = nu * (ops.d2_dx2(u) + ops.d2_dz2(u))
        visc_w = nu * (ops.d2_dx2(w) + ops.d2_dz2(w))

        temp_conv = ops.advection_u(u, w, T)
        temp_diff = kappa * ops.laplacian(T)

        u_star = u + dt * (-conv_u + visc_u)
        w_star = w + dt * (-conv_w + visc_w + buoyancy)

        u_star[0, :] = u_star[-1, :] = 0
        u_star[:, 0] = u_star[:, -1] = 0
        w_star[0, :] = w_star[-1, :] = 0
        w_star[:, 0] = w_star[:, -1] = 0

        div_u_star = ops.divergence(u_star, w_star)
        phi = solve_poisson(div_u_star / dt)
        dphi_dx, dphi_dz = ops.gradient(phi)

        u = u_star - dt * dphi_dx
        w = w_star - dt * dphi_dz
        T = T + dt * (-temp_conv + temp_diff)
        apply_bc(u, w, T)

        if (step + 1) % 1000 == 0:
            Nu_current = compute_Nu(T, u, w)
            u_p, w_p = ops.interpolate_to_pressure_points(u, w)
            max_vel = np.sqrt(np.max(u_p ** 2 + w_p ** 2))
            Nu_history.append(Nu_current)
            max_vel_hist.append(max_vel)
            if verbose:
                print(f"Ra={Ra:5.0f} step {step+1:5d} Nu={Nu_current:6.3f} "
                      f"max_vel={max_vel:6.3f}")
            if not np.isfinite(Nu_current) or max_vel > 50:
                break

    results = {
        'Nu': Nu_history[-1],
        'max_vel': max_vel_hist[-1] if max_vel_hist else 0.0,
        'Nu_history': Nu_history,
        'max_vel_history': max_vel_hist,
        'u': u,
        'w': w,
        'T': T,
        'grid': grid,
        'dt': dt,
    }
    return results


def plot_velocity_field(u, w, grid, stride=1, ax=None, title=None):
    """Plot velocity vectors on the pressure grid.

    Parameters
    ----------
    u, w : ndarray
        Velocity components on the staggered grid.
    grid : Grid2D
        Grid defining the domain and staggering.
    stride : int, optional
        Sub-sampling factor for the quiver plot.
    ax : matplotlib axis, optional
        Axis to plot on.  When ``None`` a new one is created.
    title : str, optional
        Title for the subplot.

    Returns
    -------
    matplotlib axis with the quiver plot.
    """
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


def run_study(Ra_values=None, n_steps=5000):
    """Perform a systematic study over a range of Rayleigh numbers."""
    if Ra_values is None:
        Ra_values = [1000, 2000, 5000, 10000]

    study_results = []
    fields = {}
    histories = {}
    grids = {}
    dts = {}
    vel_fields = {}

    for Ra in Ra_values:
        res = simulate_rb(Ra, n_steps=n_steps, verbose=True)
        study_results.append({'Ra': Ra, 'Nu': res['Nu'],
                              'max_vel': res['max_vel']})
        fields[Ra] = res['T']
        histories[Ra] = res['Nu_history']
        grids[Ra] = res['grid']
        dts[Ra] = res['dt']
        vel_fields[Ra] = (res['u'], res['w'])

    # Estimate critical Rayleigh number
    critical = None
    for item in study_results:
        if item['Nu'] > 1.1 and item['max_vel'] > 0.01:
            critical = item['Ra']
            break
    if critical is not None:
        print(f"Estimated critical Ra ≈ {critical} (theory ≈ 1708)")
    else:
        print("No convection observed; increase Ra or n_steps.")

    # Check Nu ~ Ra^0.25 scaling
    supercrit = [r for r in study_results if r['Ra'] > 1708]
    if len(supercrit) > 1:
        Ra_arr = np.array([r['Ra'] for r in supercrit])
        Nu_arr = np.array([r['Nu'] for r in supercrit])
        coeff = np.polyfit(np.log(Ra_arr), np.log(Nu_arr), 1)
        slope = coeff[0]
        print(f"Nu ~ Ra^{slope:.2f} (theory 0.25)")
    else:
        slope = None

    # Visualization: Nu-Ra relation
    plt.figure(figsize=(6, 4))
    Ra_arr = np.array([r['Ra'] for r in study_results])
    Nu_arr = np.array([r['Nu'] for r in study_results])
    plt.plot(Ra_arr, Nu_arr, 'o-', label='Simulation')
    theory = 0.23 * (Ra_arr / 1000) ** 0.25
    plt.plot(Ra_arr, theory, 'k--', label=r'Theory $\propto Ra^{0.25}$')
    plt.xlabel('Rayleigh number')
    plt.ylabel('Nusselt number')
    plt.legend()
    plt.title('Nu-Ra Relation')
    plt.tight_layout()

    # Temperature fields for selected Ra
    sample_Ra = [min(Ra_arr), 2000, max(Ra_arr)]
    sample_Ra = [Ra for Ra in sample_Ra if Ra in fields]
    n_samples = len(sample_Ra)
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    for ax, Ra in zip(axes, sample_Ra):
        T = fields[Ra]
        grid = grids[Ra]
        x_p, z_p = grid.get_pressure_grid()
        X, Z = np.meshgrid(x_p, z_p)
        im = ax.contourf(X, Z, T, levels=20, cmap='RdBu_r')
        ax.set_title(f'Ra={Ra}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    plt.suptitle('Temperature fields')
    plt.tight_layout()

    # Velocity fields for selected Ra
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    for ax, Ra in zip(axes, sample_Ra):
        u, w = vel_fields[Ra]
        plot_velocity_field(u, w, grids[Ra], ax=ax, title=f'Ra={Ra}', stride=2)
    plt.suptitle('Velocity fields')
    plt.tight_layout()

    # Convergence history for selected cases
    fig, ax = plt.subplots(figsize=(6, 4))
    for Ra in sample_Ra:
        history = histories[Ra]
        times = np.arange(len(history)) * 100 * dts[Ra]
        ax.plot(times, history, label=f'Ra={Ra}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Nusselt number')
    ax.legend()
    ax.set_title('Convergence history')
    plt.tight_layout()

    # Summary
    print("\nSummary:")
    for r in study_results:
        print(f"Ra={r['Ra']:5.0f} Nu={r['Nu']:.3f} max_vel={r['max_vel']:.3f}")

    plt.show()
    return study_results


if __name__ == '__main__':
    run_study()

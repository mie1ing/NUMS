import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_temperature_bc
from nums import ode


class HeatDiffusionSolver:
    """
    heat diffusion solver
    solve equation: dT/dt = κ∇²T
    """

    def __init__(self, grid, kappa=1.0):
        """
        initialize heat diffusion solver

        parameters:
            grid: Grid2D object
            kappa: heat diffusion coefficient
        """
        self.grid = grid
        self.kappa = kappa
        self.operators = FluidOperators(grid)

        # grid info
        self.nx, self.nz = grid.nx, grid.nz
        self.dx, self.dz = grid.dx, grid.dz

        print(f"Heat diffusion solver initialized")
        print(f"Grid: {self.nx}×{self.nz}, Heat diffusion coefficient: {self.kappa}")

    def diffusion_rhs(self, T_flat, t):
        """
        right hand side of the heat diffusion equation: dT/dt = κ∇²T

        parameters:
            T_flat: flattened temperature field (1D array)
            t: time

        return:
            dT_dt_flat: flattened time derivative
        """
        # reshape 1D array to 2D
        T = T_flat.reshape((self.nz + 1, self.nx + 1))

        # apply boundary conditions - before computing the Laplacian
        apply_temperature_bc(T, T_hot=1.0, T_cold=0.0)

        # calculation of the Laplacian
        laplacian_T = self.operators.laplacian(T)

        # dT/dt = κ∇²T
        dT_dt = self.kappa * laplacian_T

        # set time derivative at Dirichlet boundary to 0
        dT_dt[0, :] = 0  # bottom boundary
        dT_dt[-1, :] = 0  # top boundary


        return dT_dt.flatten()

    def solve(self, T_initial, dt, nt, bc_params=None):
        """
        solve heat diffusion equation

        parameters:
            T_initial: initial temperature field (2D array)
            dt: time step
            nt: number of time steps
            bc_params: boundary condition parameters dictionary

        return:
            T_history: temperature field history (nt+1, nz+1, nx+1)
            t_array: time array
        """
        if bc_params is None:
            bc_params = {'T_hot': 1.0, 'T_cold': 0.0}

        # check initial condition dimensions
        expected_shape = (self.nz + 1, self.nx + 1)
        if T_initial.shape != expected_shape:
            raise ValueError(f"Initial temperature field size error, expected {expected_shape}, got {T_initial.shape}")

        # apply initial boundary conditions
        T_init = T_initial.copy()
        apply_temperature_bc(T_init, bc_params['T_hot'], bc_params['T_cold'])

        print(f"Starting heat diffusion equation solving...")
        print(f"Time step: {dt}, Steps: {nt}")
        print(f"Boundary conditions: Bottom T={bc_params['T_hot']}, Top T={bc_params['T_cold']}")

        # use RK4 solver
        T_flat_init = T_init.flatten()
        T_solution, t_array = ode.rk4(self.diffusion_rhs, T_flat_init, dt, nt)

        # reshape to 3D array: (time, z, x)
        T_history = T_solution.T.reshape((nt + 1, self.nz + 1, self.nx + 1))

        print(f"Solving completed!")
        return T_history, t_array

    def check_stability(self, dt):
        """
        check numerical stability condition
        for explicit format: dt < dx²dz²/(2κ(dx²+dz²))
        """
        stability_limit = (self.dx ** 2 * self.dz ** 2) / (2 * self.kappa * (self.dx ** 2 + self.dz ** 2))
        stable = dt <= stability_limit

        print(f"Stability check:")
        print(f"  Time step: {dt}")
        print(f"  Stability limit: {stability_limit:.6f}")
        print(f"  Is stable: {'✅ Yes' if stable else '❌ No'}")

        if not stable:
            suggested_dt = 0.8 * stability_limit
            print(f"  Suggested time step: {suggested_dt:.6f}")

        return stable, stability_limit


def create_initial_condition(grid, condition_type='linear'):
    """
    create initial temperature distribution

    parameters:
        grid: Grid2D object
        condition_type: initial condition type

    return:
        T_init: initial temperature field - correct shape (nz+1, nx+1)
    """
    x_p, z_p = grid.get_pressure_grid()

    # create array with correct shape (nz+1, nx+1)
    T_init = np.zeros((grid.nz + 1, grid.nx + 1))

    if condition_type == 'linear':
        # linear distribution: bottom hot(1), top cold(0)
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz

    elif condition_type == 'gaussian':
        # gaussian distribution perturbation
        z_center = grid.Lz / 2
        # basic linear distribution
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz
        # add gaussian perturbation
        for j in range(grid.nz + 1):
            for i in range(grid.nx + 1):
                perturbation = 0.1 * np.exp(-((z_p[j] - z_center) ** 2) / (0.1 * grid.Lz) ** 2)
                T_init[j, i] += perturbation

    elif condition_type == 'random':
        # random perturbation
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz
        # add random perturbation
        T_init += 0.05 * np.random.random(T_init.shape) - 0.025

    # ensure boundary conditions
    apply_temperature_bc(T_init, T_hot=1.0, T_cold=0.0)

    return T_init


def plot_temperature_evolution(T_history, t_array, grid, save_path=None):
    """
    plot temperature field evolution

    parameters:
        T_history: temperature history (nt+1, nz+1, nx+1)
        t_array: time array
        grid: grid object
        save_path: save path
    """
    x_p, z_p = grid.get_pressure_grid()

    # select several time points for visualization
    nt = len(t_array) - 1
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, idx in enumerate(time_indices):
        if i < 5:  # only draw 5 subplots
            ax = axes[i]
            T = T_history[idx]  # Shape: (nz+1, nx+1)

            # create grid for plotting
            X_plot, Z_plot = np.meshgrid(x_p, z_p)

            im = ax.contourf(X_plot, Z_plot, T, levels=20, cmap='RdBu_r')
            ax.set_title(f't = {t_array[idx]:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

    # last subplot shows centerline temperature over time
    ax = axes[5]
    center_x_idx = grid.nx // 2
    center_z_profile = T_history[:, :, center_x_idx]

    for i in range(0, len(t_array), max(1, len(t_array) // 5)):
        ax.plot(z_p, center_z_profile[i], label=f't={t_array[i]:.3f}')

    ax.set_xlabel('z')
    ax.set_ylabel('T')
    ax.set_title('Centerline temperature profile')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


def analytical_solution_1d(z, t, kappa, L=1.0, n_terms=50):
    """
    1D heat diffusion analytical solution (for verification)
    Boundary conditions: T(0)=1, T(L)=0
    Initial condition: T(z,0) = 1 - z/L

    Analytical solution: T(z,t) = 1 - z/L + Σ[A_n * sin(nπz/L) * exp(-κ(nπ/L)²t)]
    """
    T = 1 - z / L  # steady state solution

    # Fourier series terms
    for n in range(1, n_terms + 1):
        A_n = 2 * (-1) ** (n + 1) / (n * np.pi)
        T += A_n * np.sin(n * np.pi * z / L) * np.exp(-kappa * (n * np.pi / L) ** 2 * t)

    return T


# Corrected verification function in your heat_diffusion.py

def verify_against_analytical_corrected():
    """
    Corrected analytical solution verification - using proper test cases
    """
    print("Corrected analytical solution verification...")

    # Test case 1: homogeneous boundary conditions + sine initial condition
    print("\nTest case 1: homogeneous boundary conditions")
    print("-" * 40)

    grid = Grid2D(nx=2, nz=50, Lx=0.1, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # sine initial condition: T(z,0) = sin(πz/L)
    z_p = grid.z_p
    T_init = np.sin(np.pi * z_p / grid.Lz)
    T_init = T_init[:, np.newaxis].repeat(grid.nx + 1, axis=1)

    # modify boundary condition function to handle homogeneous boundary conditions
    def apply_homogeneous_bc(T):
        T[0, :] = 0  # T(0) = 0
        T[-1, :] = 0  # T(L) = 0
        T[:, 0] = T[:, 1]  # adiabatic side boundary
        T[:, -1] = T[:, -2]

    apply_homogeneous_bc(T_init)

    # analytical solution: T(z,t) = sin(πz/L) * exp(-κ(π/L)²t)
    def analytical_homogeneous(z, t, kappa, L):
        return np.sin(np.pi * z / L) * np.exp(-kappa * (np.pi / L) ** 2 * t)

    # numerical solution
    dt = 0.0001
    nt = 500  # shorter time, see decay

    # temporarily modify solver's boundary conditions
    original_diffusion_rhs = solver.diffusion_rhs

    def homogeneous_diffusion_rhs(T_flat, t):
        T = T_flat.reshape((solver.nz + 1, solver.nx + 1))
        apply_homogeneous_bc(T)

        laplacian_T = solver.operators.laplacian(T)
        dT_dt = solver.kappa * laplacian_T

        # boundary points time derivative is 0
        dT_dt[0, :] = 0
        dT_dt[-1, :] = 0
        dT_dt[:, 0] = 0
        dT_dt[:, -1] = 0

        return dT_dt.flatten()

    solver.diffusion_rhs = homogeneous_diffusion_rhs

    T_history, t_array = solver.solve(T_init, dt, nt)

    # restore original function
    solver.diffusion_rhs = original_diffusion_rhs

    # verification
    t_final = t_array[-1]
    T_numerical = T_history[-1, :, 0]
    T_analytical = analytical_homogeneous(z_p, t_final, solver.kappa, grid.Lz)

    error = np.abs(T_numerical - T_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"Homogeneous boundary condition test:")
    print(f"  Final time: {t_final:.4f}")
    print(f"  Theoretical decay factor: {np.exp(-solver.kappa * (np.pi / grid.Lz) ** 2 * t_final):.6f}")
    print(f"  Numerical decay factor: {np.max(T_numerical) / np.max(T_init):.6f}")
    print(f"  Maximum error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")

    homogeneous_test = max_error < 0.01

    # Test case 2: non-homogeneous boundary conditions + perturbation initial condition
    print("\nTest case 2: non-homogeneous boundary conditions + perturbation")
    print("-" * 40)

    # initial condition: T = 1 - z/L + 0.1*sin(2πz/L)
    T_init_perturbed = np.zeros((grid.nz + 1, grid.nx + 1))
    for j in range(grid.nz + 1):
        steady_part = 1.0 - z_p[j] / grid.Lz
        perturbation = 0.1 * np.sin(2 * np.pi * z_p[j] / grid.Lz)
        T_init_perturbed[j, :] = steady_part + perturbation

    apply_temperature_bc(T_init_perturbed, T_hot=1.0, T_cold=0.0)

    # analytical solution: T = (1 - z/L) + 0.1*sin(2πz/L)*exp(-κ(2π/L)²t)
    def analytical_perturbed(z, t, kappa, L):
        steady = 1 - z / L
        transient = 0.1 * np.sin(2 * np.pi * z / L) * np.exp(-kappa * (2 * np.pi / L) ** 2 * t)
        return steady + transient

    # numerical solution
    T_history2, t_array2 = solver.solve(T_init_perturbed, dt, nt)

    # verification
    t_final2 = t_array2[-1]
    T_numerical2 = T_history2[-1, :, 0]
    T_analytical2 = analytical_perturbed(z_p, t_final2, solver.kappa, grid.Lz)

    error2 = np.abs(T_numerical2 - T_analytical2)
    max_error2 = np.max(error2)
    mean_error2 = np.mean(error2)

    print(f"Perturbation test:")
    print(f"  Final time: {t_final2:.4f}")
    print(f"  Perturbation decay factor: {np.exp(-solver.kappa * (2 * np.pi / grid.Lz) ** 2 * t_final2):.6f}")
    print(f"  Maximum error: {max_error2:.6f}")
    print(f"  Mean error: {mean_error2:.6f}")

    perturbation_test = max_error2 < 0.01

    # visualization of results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # homogeneous boundary condition results
    axes[0, 0].plot(z_p, T_init[:, 0], 'b-', label='initial condition')
    axes[0, 0].plot(z_p, T_numerical, 'ro-', markersize=3, label='numerical solution')
    axes[0, 0].plot(z_p, T_analytical, 'g--', linewidth=2, label='analytical solution')
    axes[0, 0].set_title('Homogenous Boundary Condition Test')
    axes[0, 0].set_xlabel('z')
    axes[0, 0].set_ylabel('T')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(z_p, error, 'r-')
    axes[0, 1].set_title('Homogenous boundary condition error')
    axes[0, 1].set_xlabel('z')
    axes[0, 1].set_ylabel('|error|')
    axes[0, 1].grid(True)

    # perturbation test results
    axes[1, 0].plot(z_p, T_init_perturbed[:, 0], 'b-', label='initial condition')
    axes[1, 0].plot(z_p, T_numerical2, 'ro-', markersize=3, label='numerical solution')
    axes[1, 0].plot(z_p, T_analytical2, 'g--', linewidth=2, label='analytical solution')
    axes[1, 0].plot(z_p, 1 - z_p / grid.Lz, 'k:', label='stable solution')
    axes[1, 0].set_title('Disturbance testing')
    axes[1, 0].set_xlabel('z')
    axes[1, 0].set_ylabel('T')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(z_p, error2, 'r-')
    axes[1, 1].set_title('Disturbance testing error')
    axes[1, 1].set_xlabel('z')
    axes[1, 1].set_ylabel('|error|')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    overall_success = homogeneous_test and perturbation_test

    print(f"\nVerification summary:")
    print(f"  Homogeneous boundary conditions: {'✅ Passed' if homogeneous_test else '❌ Failed'}")
    print(f"  Perturbation test: {'✅ Passed' if perturbation_test else '❌ Failed'}")
    print(f"  Overall verification: {'✅ Passed' if overall_success else '❌ Failed'}")

    return overall_success


# Also correct simple test, using reasonable time scales
def simple_test_corrected():
    """
    Corrected simple test - using reasonable time scales
    """
    print("Corrected simple test...")

    grid = Grid2D(nx=4, nz=10, Lx=0.2, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # diffusion time scale
    tau_diffusion = grid.Lz ** 2 / solver.kappa
    print(f"Diffusion time scale τ = L²/κ = {tau_diffusion}")

    # create initial condition with perturbation
    T_init = create_initial_condition(grid, 'gaussian')

    # run to 3τ time, should mostly reach steady state
    dt = 0.001
    nt = int(3 * tau_diffusion / dt)
    print(f"Running to t = 3τ = {3 * tau_diffusion}, requires {nt} steps")

    # if too many steps, reduce to reasonable range
    if nt > 30000:
        nt = 10000
        dt = 3 * tau_diffusion / nt
        print(f"Adjusted parameters: dt = {dt:.6f}, nt = {nt}")

    T_history, t_array = solver.solve(T_init, dt, nt)

    # check final state
    T_final = T_history[-1]
    z_p = grid.z_p
    T_steady = 1.0 - z_p / grid.Lz  # theoretical steady state

    # calculate difference from steady state
    diff = np.abs(T_final[:, 0] - T_steady)
    max_diff = np.max(diff)

    print(f"Corrected simple test results:")
    print(f"  Final time: {t_array[-1]:.3f}")
    print(f"  Time/Diffusion time scale: {t_array[-1] / tau_diffusion:.3f}")
    print(f"  Maximum difference from steady state: {max_diff:.6f}")
    print(f"  Converged: {'✅ Yes' if max_diff < 0.01 else '❌ No'}")

    # visualize convergence process
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt]
    for i, idx in enumerate(time_indices):
        plt.plot(z_p, T_history[idx, :, 0], label=f't={t_array[idx]:.2f}')
    plt.plot(z_p, T_steady, 'k--', linewidth=2, label='stable solution')
    plt.xlabel('z')
    plt.ylabel('T')
    plt.title('Temperature distribution evolution')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    # calculate total energy (integral) change over time
    total_energy = []
    for i in range(len(t_array)):
        energy = np.trapezoid(T_history[i, :, 0], z_p)
        total_energy.append(energy)

    steady_energy = np.trapezoid(T_steady, z_p)
    plt.plot(t_array, total_energy, 'b-', label='total energy')
    plt.axhline(steady_energy, color='k', linestyle='--', label='stable energy')
    plt.xlabel('time')
    plt.ylabel('total energy')
    plt.title('energy evolution')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # difference from steady state over time
    max_deviation = []
    for i in range(len(t_array)):
        dev = np.max(np.abs(T_history[i, :, 0] - T_steady))
        max_deviation.append(dev)

    plt.semilogy(t_array, max_deviation, 'r-')
    plt.axhline(0.01, color='k', linestyle='--', alpha=0.5, label='convergence criteria')
    plt.xlabel('time')
    plt.ylabel('largest deviation')
    plt.title('converge process')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return max_diff < 0.01


def demonstrate_correct_physics():
    """
    Demonstrate correct physical phenomena
    """
    print("\n" + "=" * 60)
    print("Demonstrating correct diffusion physics")
    print("=" * 60)

    # create grid
    grid = Grid2D(nx=32, nz=16, Lx=2.0, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # diffusion time scale
    tau = grid.Lz ** 2 / solver.kappa
    print(f"Diffusion time scale: τ = {tau}")

    # create interesting initial condition: step function
    T_init = np.zeros((grid.nz + 1, grid.nx + 1))
    z_p = grid.z_p

    # step initial condition
    for j in range(grid.nz + 1):
        if z_p[j] < grid.Lz / 2:
            T_init[j, :] = 1.0
        else:
            T_init[j, :] = 0.0

    # apply boundary conditions
    apply_temperature_bc(T_init, T_hot=1.0, T_cold=0.0)

    # run at several different time scales
    time_fractions = [0.01, 0.05, 0.1, 0.3, 1.0]  # multiples of τ

    results = {}

    for frac in time_fractions:
        target_time = frac * tau
        dt = 0.001
        nt = int(target_time / dt)

        if nt > 5000:  # limit computation
            nt = 5000
            dt = target_time / nt

        print(f"\nRunning to t = {frac}τ = {target_time:.3f}")
        print(f"  Time step: dt = {dt:.6f}, steps = {nt}")

        T_history, t_array = solver.solve(T_init, dt, nt)
        results[frac] = {
            'T_final': T_history[-1],
            't_final': t_array[-1],
            'T_history': T_history,
            't_array': t_array
        }

    # visualize results at different times
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)

    for i, frac in enumerate(time_fractions):
        if i < 5:
            ax = axes[i]
            T = results[frac]['T_final']
            t = results[frac]['t_final']

            im = ax.contourf(X, Z, T, levels=20, cmap='RdBu_r', vmin=0, vmax=1)
            ax.set_title(f't = {frac}τ = {t:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

    # last subplot: centerline temperature profile
    ax = axes[5]
    center_x = grid.nx // 2

    # initial condition
    ax.plot(z_p, T_init[:, center_x], 'k-', linewidth=3, label='initial')

    # profiles at different times
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, frac in enumerate(time_fractions):
        T = results[frac]['T_final']
        t = results[frac]['t_final']
        ax.plot(z_p, T[:, center_x], color=colors[i],
                label=f't={frac}τ={t:.3f}')

    # steady state solution
    T_steady = 1 - z_p / grid.Lz
    ax.plot(z_p, T_steady, 'k--', linewidth=2, label='stable')

    ax.set_xlabel('z')
    ax.set_ylabel('T')
    ax.set_title('Centerline temperature evolution')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# Main test function
def main_corrected_test():
    """
    Run all corrected tests
    """
    print("=" * 60)
    print("Corrected heat diffusion solver verification")
    print("=" * 60)

    tests = []

    # 1. Corrected simple test
    print("\n1. Corrected simple steady state test")
    print("-" * 40)
    simple_ok = simple_test_corrected()
    tests.append(("Simple steady state test", simple_ok))

    # 2. Corrected analytical solution verification
    print("\n2. Corrected analytical solution verification")
    print("-" * 40)
    analytical_ok = verify_against_analytical_corrected()
    tests.append(("Analytical solution verification", analytical_ok))

    # 3. Physics demonstration
    print("\n3. Diffusion physics demonstration")
    print("-" * 40)
    demonstrate_correct_physics()

    # Summary
    print("\n" + "=" * 60)
    print("Final verification results")
    print("=" * 60)

    for name, result in tests:
        print(f"  {name}: {'✅ Passed' if result else '❌ Failed'}")

    all_passed = all(result for _, result in tests)

    if all_passed:
        print("\n🎉 Congratulations! Your heat diffusion solver is correct!")
    else:
        print("\nThere are still some issues to resolve...")

    return all_passed


if __name__ == "__main__":
    main_corrected_test()
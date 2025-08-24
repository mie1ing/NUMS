import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators


def test_basic_staggered_operations():
    """
    Test basic staggered grid operations to ensure they work correctly
    """
    print("=" * 60)
    print("BASIC STAGGERED GRID TEST")
    print("=" * 60)

    # Create small staggered grid
    grid = Grid2D(nx=8, nz=4, Lx=2.0, Lz=1.0, staggered=True)
    operators = FluidOperators(grid)

    print(f"Grid info:")
    print(f"  Domain: {grid.Lx} × {grid.Lz}")
    print(f"  Grid size: {grid.nx} × {grid.nz}")
    print(f"  Staggered: {grid.staggered}")

    # Get grid coordinates
    x_p, z_p = grid.get_pressure_grid()
    x_u, z_u = grid.get_u_grid()
    x_w, z_w = grid.get_w_grid()

    print(f"\nGrid dimensions:")
    print(f"  Pressure grid: {len(z_p)} × {len(x_p)} = {x_p.shape} × {z_p.shape}")
    print(f"  U-velocity grid: {len(z_u)} × {len(x_u)} = {z_u.shape} × {x_u.shape}")
    print(f"  W-velocity grid: {len(z_w)} × {len(x_w)} = {z_w.shape} × {x_w.shape}")

    # Create fields with correct dimensions
    print(f"\nCreating fields:")

    # Pressure/temperature field
    T = grid.create_field('pressure')
    print(f"  Temperature field shape: {T.shape}")

    # Velocity fields
    u = grid.create_field('u_velocity')
    w = grid.create_field('w_velocity')
    print(f"  U-velocity field shape: {u.shape}")
    print(f"  W-velocity field shape: {w.shape}")

    # Initialize with simple test patterns
    print(f"\nInitializing test fields...")

    # Temperature: linear profile
    for j in range(T.shape[0]):
        for i in range(T.shape[1]):
            T[j, i] = 1.0 - z_p[j] / grid.Lz  # Hot bottom, cold top

    # U-velocity: simple profile
    u.fill(0.0)  # Start with zero

    # W-velocity: simple profile
    w.fill(0.0)  # Start with zero

    print(f"  Temperature range: {np.min(T):.3f} to {np.max(T):.3f}")

    # Test divergence calculation
    print(f"\nTesting divergence operator...")

    try:
        div_u = operators.divergence(u, w)
        print(f"  ✅ Divergence computed successfully")
        print(f"  Divergence field shape: {div_u.shape}")
        print(f"  Max divergence: {np.max(np.abs(div_u)):.2e}")

    except Exception as e:
        print(f"  ❌ Divergence failed: {e}")
        return False

    # Test gradient calculation
    print(f"\nTesting gradient operator...")

    try:
        grad_T_x, grad_T_z = operators.gradient(T)
        print(f"  ✅ Gradient computed successfully")
        print(f"  dT/dx shape: {grad_T_x.shape}")
        print(f"  dT/dz shape: {grad_T_z.shape}")

    except Exception as e:
        print(f"  ❌ Gradient failed: {e}")
        return False

    # Test Laplacian
    print(f"\nTesting Laplacian operator...")

    try:
        lap_T = operators.laplacian(T)
        print(f"  ✅ Laplacian computed successfully")
        print(f"  Laplacian shape: {lap_T.shape}")
        print(f"  Max Laplacian: {np.max(np.abs(lap_T)):.2e}")

    except Exception as e:
        print(f"  ❌ Laplacian failed: {e}")
        return False

    # Test field interpolation
    print(f"\nTesting field interpolation...")

    try:
        u_p, w_p = operators.interpolate_to_pressure_points(u, w)
        print(f"  ✅ Interpolation computed successfully")
        print(f"  u at pressure points shape: {u_p.shape}")
        print(f"  w at pressure points shape: {w_p.shape}")

    except Exception as e:
        print(f"  ❌ Interpolation failed: {e}")
        return False

    # Visualize the grid layout
    print(f"\nCreating visualization...")

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Temperature field
        ax = axes[0, 0]
        X_p, Z_p = np.meshgrid(x_p, z_p)
        im1 = ax.contourf(X_p, Z_p, T, levels=10, cmap='RdBu_r')
        ax.set_title('Temperature Field')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.colorbar(im1, ax=ax)

        # Grid points layout
        ax = axes[0, 1]

        # Plot different grid point types
        X_p_mesh, Z_p_mesh = np.meshgrid(x_p, z_p)
        X_u_mesh, Z_u_mesh = np.meshgrid(x_u, z_u)
        X_w_mesh, Z_w_mesh = np.meshgrid(x_w, z_w)

        ax.plot(X_p_mesh.flatten(), Z_p_mesh.flatten(), 'ro', markersize=6, label='P/T points')
        ax.plot(X_u_mesh.flatten(), Z_u_mesh.flatten(), 'b>', markersize=4, label='U points')
        ax.plot(X_w_mesh.flatten(), Z_w_mesh.flatten(), 'g^', markersize=4, label='W points')

        ax.set_title('Staggered Grid Layout')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Divergence field
        ax = axes[1, 0]
        im2 = ax.contourf(X_p, Z_p, div_u, levels=10, cmap='RdBu_r')
        ax.set_title('Divergence Field')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.colorbar(im2, ax=ax)

        # Summary
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
        STAGGERED GRID TEST RESULTS

        Grid Size: {grid.nx} × {grid.nz}
        Domain: {grid.Lx} × {grid.Lz}

        Field Dimensions:
        • Temperature: {T.shape}
        • U-velocity: {u.shape}  
        • W-velocity: {w.shape}

        Operator Tests:
        • Divergence: ✅
        • Gradient: ✅
        • Laplacian: ✅
        • Interpolation: ✅

        Max Divergence: {np.max(np.abs(div_u)):.2e}
        (Should be ~0 for zero velocity)
        """

        ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace')

        plt.tight_layout()
        plt.show()

        print(f"✅ Visualization created successfully")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"STAGGERED GRID TEST COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 60}")

    return True


def test_simple_rb_staggered_fixed():
    """Rayleigh-Bénard test using a staggered grid with pressure projection."""
    print("\n" + "=" * 60)
    print("FIXED R-B TEST (STAGGERED GRID)")
    print("=" * 60)

    # Small staggered grid
    grid = Grid2D(nx=16, nz=8, Lx=2.0, Lz=1.0, staggered=True)
    operators = FluidOperators(grid)

    # Physical parameters
    Ra_target = 5000
    L = grid.Lz
    g = 9.81
    alpha = 1e-3
    Delta_T = 1.0
    nu = kappa = np.sqrt((g * alpha * Delta_T * L ** 3) / Ra_target)

    print(f"Target Ra = {Ra_target}")
    print(f"ν = κ = {nu:.2e}")
    print(f"Expected: Strong convection (Nu >> 1)")

    # Create staggered fields
    u = grid.create_field('u_velocity')
    w = grid.create_field('w_velocity')
    T = grid.create_field('pressure')

    # Initialize temperature with linear profile + perturbation
    x_p, z_p = grid.get_pressure_grid()
    T_hot, T_cold = 0.5, -0.5

    for j in range(grid.nz + 1):
        T_linear = T_cold + (T_hot - T_cold) * (1 - z_p[j] / grid.Lz)
        for i in range(grid.nx + 1):
            pert = 0.1 * np.sin(np.pi * x_p[i] / grid.Lx) * np.sin(np.pi * z_p[j] / grid.Lz)
            T[j, i] = T_linear + pert

    # Apply boundary conditions
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

    def compute_Nu(T):
        dT_dz_bottom = (T[1, :] - T[0, :]) / grid.dz
        avg_gradient = np.mean(dT_dz_bottom)
        return -L * avg_gradient / (T_hot - T_cold)

    def solve_poisson_simple(rhs, max_iter=200):
        phi = np.zeros_like(rhs)
        rhs_work = rhs - np.mean(rhs)
        for it in range(max_iter):
            phi_old = phi.copy()
            for j in range(1, phi.shape[0] - 1):
                for i in range(1, phi.shape[1] - 1):
                    phi[j, i] = 0.25 * (
                        phi[j + 1, i] + phi[j - 1, i] +
                        phi[j, i + 1] + phi[j, i - 1] -
                        grid.dx ** 2 * rhs_work[j, i]
                    )
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]
            phi -= np.mean(phi)
            if np.max(np.abs(phi - phi_old)) < 1e-6:
                break
        return phi

    Nu_initial = compute_Nu(T)
    print(f"Initial Nu = {Nu_initial:.4f}")

    dt = 5e-5
    n_steps = 1000
    print(f"Running {n_steps} time steps with dt = {dt:.1e}...")

    Nu_history = [Nu_initial]
    max_vel_history = []

    for step in range(n_steps):
        T_ref = (T_hot + T_cold) / 2
        T_w = 0.5 * (T[:-1, :] + T[1:, :])
        buoyancy = g * alpha * (T_w - T_ref)

        du_dx = operators.d_dx(u)
        du_dz = operators.d_dz(u)
        dw_dx = operators.d_dx(w)
        dw_dz = operators.d_dz(w)

        w_on_u = np.zeros_like(u)
        w_on_u[:, 1:-1] = 0.5 * (w[:-1, 1:-1] + w[1:, 1:-1])
        u_on_w = np.zeros_like(w)
        u_on_w[1:-1, :] = 0.5 * (u[1:-1, :-1] + u[1:-1, 1:])

        conv_u = u * du_dx + w_on_u * du_dz
        conv_w = u_on_w * dw_dx + w * dw_dz

        visc_u = nu * (operators.d2_dx2(u) + operators.d2_dz2(u))
        visc_w = nu * (operators.d2_dx2(w) + operators.d2_dz2(w))

        temp_conv = operators.advection_u(u, w, T)
        temp_diff = kappa * operators.laplacian(T)

        du_dt_star = -conv_u + visc_u
        dw_dt_star = -conv_w + visc_w + buoyancy

        u_star = u + dt * du_dt_star
        w_star = w + dt * dw_dt_star

        u_star[0, :] = u_star[-1, :] = 0
        u_star[:, 0] = u_star[:, -1] = 0
        w_star[0, :] = w_star[-1, :] = 0
        w_star[:, 0] = w_star[:, -1] = 0

        div_u_star = operators.divergence(u_star, w_star)
        phi = solve_poisson_simple(div_u_star / dt)
        dphi_dx, dphi_dz = operators.gradient(phi)

        u_new = u_star - dt * dphi_dx
        w_new = w_star - dt * dphi_dz

        T_new = T + dt * (-temp_conv + temp_diff)

        u, w, T = u_new.copy(), w_new.copy(), T_new.copy()
        apply_bc(u, w, T)

        if (step + 1) % 100 == 0:
            Nu_current = compute_Nu(T)
            u_p, w_p = operators.interpolate_to_pressure_points(u, w)
            max_vel = np.sqrt(np.max(u_p ** 2 + w_p ** 2))
            max_div = np.max(np.abs(operators.divergence(u, w)))
            Nu_history.append(Nu_current)
            max_vel_history.append(max_vel)
            print(f"  Step {step + 1:4d}: Nu = {Nu_current:.4f}, max_vel = {max_vel:.4f}, max_div = {max_div:.1e}")
            if not np.isfinite(Nu_current) or max_vel > 50:
                print("  ❌ Instability detected!")
                break

    Nu_final = Nu_history[-1]
    max_vel_final = max_vel_history[-1] if max_vel_history else 0

    print(f"\nFinal Results:")
    print(f"  Nu = {Nu_final:.4f}")
    print(f"  Max velocity = {max_vel_final:.4f}")

    convection_detected = max_vel_final > 0.01 and Nu_final > 1.1
    if Ra_target > 1708:
        expected = f"Nu > 1 and velocity > 0 (Ra = {Ra_target} > 1708)"
        success = convection_detected
    else:
        expected = "Nu ≈ 1 and low velocity"
        success = 0.9 < Nu_final < 1.1 and max_vel_final < 0.01

    print(f"Expected: {expected}")
    print(f"Convection detected: {'✅' if convection_detected else '❌'}")
    print(f"Overall result: {'✅ SUCCESS' if success else '❌ FAILED'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    X, Z = np.meshgrid(x_p, z_p)

    ax = axes[0, 0]
    im = ax.contourf(X, Z, T, levels=20, cmap='RdBu_r')
    ax.set_title(f'Temperature (Nu = {Nu_final:.3f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    u_p, w_p = operators.interpolate_to_pressure_points(u, w)
    magnitude = np.sqrt(u_p ** 2 + w_p ** 2)
    skip = max(1, grid.nx // 8)
    im2 = ax.contourf(X, Z, magnitude, levels=10, cmap='viridis')
    if np.max(magnitude) > 0:
        ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                  u_p[::skip, ::skip], w_p[::skip, ::skip],
                  scale=None, alpha=0.8, color='white')
    ax.set_title('Velocity Field')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax)

    ax = axes[1, 0]
    time_points = np.arange(len(Nu_history)) * 100 * dt
    ax.plot(time_points, Nu_history, 'b-', linewidth=2, label='Simulated')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Nu = 1 (conduction)')
    if Ra_target > 1708:
        Nu_theory = 0.23 * (Ra_target / 1000) ** 0.25
        ax.axhline(y=Nu_theory, color='g', linestyle=':', alpha=0.7,
                   label=f'Theory ≈ {Nu_theory:.2f}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Nusselt Number')
    ax.set_title('Nu Evolution')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    if max_vel_history:
        ax.semilogy(time_points[1:], max_vel_history, 'r-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Max Velocity')
        ax.set_title('Convection Growth')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Velocity Data')

    plt.tight_layout()
    plt.show()

    return success


if __name__ == "__main__":
    # Test 1: Basic staggered grid operations
    staggered_success = test_basic_staggered_operations()

    # Test 2: Fixed physics test with proper pressure projection
    if staggered_success:
        physics_success = test_simple_rb_staggered_fixed()

        if physics_success:
            print(f"\n🎉 ALL TESTS PASSED! Ready for final project.")
        else:
            print(f"\n⚠️  Still debugging physics. Check pressure projection.")
    else:
        print(f"\n❌ Staggered grid test failed.")
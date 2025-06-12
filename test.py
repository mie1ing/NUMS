import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from navier_stokes import NavierStokesSolver, create_initial_conditions, create_boundary_conditions


def test_stokes_flow():
    """
    测试Stokes流（Re → 0）
    验证粘性主导情况下的解
    """
    print("=" * 60)
    print("Testing Stokes Flow (Low Reynolds Number)")
    print("=" * 60)

    # 创建小网格快速测试
    grid = Grid2D(nx=16, nz=16, Lx=1.0, Lz=1.0, staggered=False)

    # 高粘性（低雷诺数）
    nu = 0.1  # 较大的粘性系数
    solver = NavierStokesSolver(grid, nu=nu, rho=1.0)

    # 创建初始条件：小扰动
    u, w, p = create_initial_conditions(grid, case='vortex')
    u *= 0.1  # 小初始速度
    w *= 0.1

    # 边界条件：无滑移
    bc_params = create_boundary_conditions(case='cavity')
    bc_params['u_top'] = 0.0  # 无移动壁面，纯扩散

    # 短时间积分
    dt = 0.001
    nt = 100

    Re = 0.1 * 1.0 / nu
    print(f"Simulation parameters:")
    print(f"  Reynolds number ≈ UL/ν ≈ {Re:.1f}")
    print(f"  Time step: {dt}")
    print(f"  Steps: {nt}")

    # 求解
    u_history, w_history, p_history, t_array = solver.solve(
        u, w, p, dt, nt, bc_params=bc_params
    )

    # 分析结果
    initial_ke = 0.5 * np.mean(u_history[0] ** 2 + w_history[0] ** 2)
    final_ke = 0.5 * np.mean(u_history[-1] ** 2 + w_history[-1] ** 2)

    print(f"\nResults:")
    print(f"  Initial kinetic energy: {initial_ke:.6f}")
    print(f"  Final kinetic energy: {final_ke:.6f}")
    print(f"  Energy decay ratio: {final_ke / initial_ke:.6f}")
    print(f"  Expected decay (Stokes): exponential")

    # 可视化
    solver.plot_results(u_history[0], w_history[0], p_history[0],
                        "Initial Condition")
    solver.plot_results(u_history[-1], w_history[-1], p_history[-1],
                        f"Stokes Flow at t={t_array[-1]:.3f}")

    # 检查能量衰减
    energy_decayed = final_ke < initial_ke * 0.9
    return energy_decayed


def test_cavity_flow():
    """
    测试经典方腔流
    """
    print("\n" + "=" * 60)
    print("Testing Driven Cavity Flow")
    print("=" * 60)

    # 创建网格
    grid = Grid2D(nx=32, nz=32, Lx=1.0, Lz=1.0, staggered=False)

    # 中等雷诺数
    nu = 0.01
    solver = NavierStokesSolver(grid, nu=nu, rho=1.0)

    # 初始条件：静止流体
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))
    p = np.zeros((grid.nz + 1, grid.nx + 1))

    # 边界条件：顶部移动壁面
    bc_params = create_boundary_conditions(case='cavity')

    # 时间积分
    dt = 0.001
    nt = 500

    Re = bc_params['u_top'] * grid.Lx / nu
    print(f"Simulation parameters:")
    print(f"  Reynolds number: {Re:.0f}")
    print(f"  Time step: {dt}")
    print(f"  Steps: {nt}")

    # 求解
    u_history, w_history, p_history, t_array = solver.solve(
        u, w, p, dt, nt, bc_params=bc_params
    )

    # 分析中心线速度剖面
    center_x = grid.nx // 2
    center_z = grid.nz // 2

    u_centerline = u_history[-1, :, center_x]
    w_centerline = w_history[-1, center_z, :]

    print(f"\nResults:")
    print(f"  Max u-velocity: {np.max(np.abs(u_history[-1])):.4f}")
    print(f"  Max w-velocity: {np.max(np.abs(w_history[-1])):.4f}")
    print(f"  Max pressure: {np.max(np.abs(p_history[-1])):.4f}")

    # 可视化
    solver.plot_results(u_history[-1], w_history[-1], p_history[-1],
                        f"Cavity Flow at t={t_array[-1]:.3f}")

    # 绘制中心线速度剖面
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    z_p = grid.z_p
    plt.plot(u_centerline, z_p / grid.Lz, 'b-', linewidth=2)
    plt.xlabel('u-velocity')
    plt.ylabel('z/L')
    plt.title('U-velocity at centerline')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x_p = grid.x_p
    plt.plot(x_p / grid.Lx, w_centerline, 'r-', linewidth=2)
    plt.xlabel('x/L')
    plt.ylabel('w-velocity')
    plt.title('W-velocity at centerline')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
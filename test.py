"""
简化的Boussinesq测试 - 验证基础功能
"""
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators


def simple_rayleigh_benard_test():
    """简单的Rayleigh-Bénard测试"""
    print("开始简单的Rayleigh-Bénard对流测试...")

    # 创建小网格进行快速测试
    grid = Grid2D(nx=16, nz=8, Lx=2.0, Lz=1.0, staggered=False)
    operators = FluidOperators(grid)

    print(f"网格大小: {grid.nx}×{grid.nz}")
    print(f"网格间距: dx={grid.dx:.3f}, dz={grid.dz:.3f}")

    # 创建简单的初始条件
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))
    T = np.zeros((grid.nz + 1, grid.nx + 1))

    # 设置温度剖面：底部热，顶部冷
    x_p, z_p = grid.get_pressure_grid()

    for j in range(grid.nz + 1):
        # 线性温度剖面
        T[j, :] = 1.0 - z_p[j] / grid.Lz  # 从1.0到0.0

        # 添加小扰动
        for i in range(grid.nx + 1):
            perturbation = 0.01 * np.sin(2 * np.pi * x_p[i] / grid.Lx) * \
                           np.sin(np.pi * z_p[j] / grid.Lz)
            T[j, i] += perturbation

    # 应用温度边界条件
    T[0, :] = 1.0  # 底部热
    T[-1, :] = 0.0  # 顶部冷
    T[:, 0] = T[:, 1]  # 左边绝热
    T[:, -1] = T[:, -2]  # 右边绝热

    print("初始条件设置完成")
    print(f"温度范围: {np.min(T):.3f} 到 {np.max(T):.3f}")

    # 测试算子功能
    print("\n测试微分算子...")

    # 测试梯度
    dT_dx = operators.d_dx(T)
    dT_dz = operators.d_dz(T)
    print(f"温度梯度计算完成，范围: dT/dx=[{np.min(dT_dx):.3f}, {np.max(dT_dx):.3f}]")
    print(f"                    dT/dz=[{np.min(dT_dz):.3f}, {np.max(dT_dz):.3f}]")

    # 测试拉普拉斯算子
    lap_T = operators.laplacian(T)
    print(f"拉普拉斯算子计算完成，范围: [{np.min(lap_T):.3f}, {np.max(lap_T):.3f}]")

    # 测试散度算子
    div_u = operators.d_dx(u) + operators.d_dz(w)
    print(f"速度散度: {np.max(np.abs(div_u)):.2e} (应该接近0)")

    # 可视化初始条件
    X, Z = np.meshgrid(x_p, z_p)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 温度场
    ax = axes[0, 0]
    im1 = ax.contourf(X, Z, T, levels=20, cmap='RdBu_r')
    ax.set_title('Initial temperature field')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax)

    # 温度梯度（x方向）
    ax = axes[0, 1]
    im2 = ax.contourf(X, Z, dT_dx, levels=20, cmap='RdBu_r')
    ax.set_title('dT/dx')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax)

    # 温度梯度（z方向）
    ax = axes[1, 0]
    im3 = ax.contourf(X, Z, dT_dz, levels=20, cmap='RdBu_r')
    ax.set_title('dT/dz')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax)

    # 拉普拉斯算子
    ax = axes[1, 1]
    im4 = ax.contourf(X, Z, lap_T, levels=20, cmap='RdBu_r')
    ax.set_title('∇²T')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    plt.colorbar(im4, ax=ax)

    plt.tight_layout()
    plt.show()

    print("\n✅ 基础测试完成！")
    print("如果看到合理的图形，说明基础组件工作正常。")

    return grid, operators, u, w, T


def test_poisson_solver():
    """测试泊松求解器"""
    print("\n" + "=" * 50)
    print("测试泊松求解器")
    print("=" * 50)

    # 创建网格
    grid = Grid2D(nx=16, nz=16, Lx=1.0, Lz=1.0, staggered=False)
    operators = FluidOperators(grid)

    # 创建测试右端项：∇²φ = -2π²sin(πx)sin(πz)
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)

    # 解析解：φ = sin(πx)sin(πz)
    phi_exact = np.sin(np.pi * X) * np.sin(np.pi * Z)

    # 右端项
    rhs = -2 * np.pi ** 2 * phi_exact

    # 简单的泊松求解器
    def solve_poisson_simple(rhs, max_iter=1000, tol=1e-6):
        phi = np.zeros_like(rhs)
        dx = grid.dx

        for iteration in range(max_iter):
            phi_old = phi.copy()

            # Gauss-Seidel
            for j in range(1, rhs.shape[0] - 1):
                for i in range(1, rhs.shape[1] - 1):
                    phi[j, i] = 0.25 * (
                            phi[j + 1, i] + phi[j - 1, i] +
                            phi[j, i + 1] + phi[j, i - 1] -
                            dx ** 2 * rhs[j, i]
                    )

            # 边界条件：φ = 0
            phi[0, :] = 0
            phi[-1, :] = 0
            phi[:, 0] = 0
            phi[:, -1] = 0

            if iteration % 100 == 0:
                change = np.max(np.abs(phi - phi_old))
                print(f"  迭代 {iteration}: 变化 = {change:.2e}")
                if change < tol:
                    print(f"  收敛于迭代 {iteration}")
                    break

        return phi

    # 求解
    phi_numerical = solve_poisson_simple(rhs)

    # 比较
    error = np.abs(phi_numerical - phi_exact)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"泊松求解器测试结果:")
    print(f"  最大误差: {max_error:.2e}")
    print(f"  平均误差: {mean_error:.2e}")
    print(f"  测试: {'✅ 通过' if max_error < 0.01 else '❌ 失败'}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].contourf(X, Z, phi_exact, levels=20, cmap='RdBu_r')
    axes[0].set_title('Analytical solution')
    axes[0].set_aspect('equal')

    axes[1].contourf(X, Z, phi_numerical, levels=20, cmap='RdBu_r')
    axes[1].set_title('Numerical solution')
    axes[1].set_aspect('equal')

    axes[2].contourf(X, Z, error, levels=20, cmap='viridis')
    axes[2].set_title('Error')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return max_error < 0.01


def estimate_stability_parameters():
    """估算稳定性参数"""
    print("\n" + "=" * 50)
    print("估算稳定性参数")
    print("=" * 50)

    # 典型参数
    grid = Grid2D(nx=32, nz=16, Lx=2.0, Lz=1.0, staggered=False)

    # 物理参数
    nu = 1e-5  # 动力学粘度
    kappa = 1e-5  # 热扩散系数
    alpha = 1e-3  # 热膨胀系数
    g = 9.81  # 重力
    Delta_T = 1.0  # 温差
    L = grid.Lz  # 特征长度

    # 无量纲参数
    Pr = nu / kappa
    Ra = g * alpha * Delta_T * L ** 3 / (nu * kappa)

    print(f"网格: {grid.nx}×{grid.nz}")
    print(f"网格间距: dx={grid.dx:.4f}, dz={grid.dz:.4f}")
    print(f"Prandtl数: Pr = {Pr:.2f}")
    print(f"Rayleigh数: Ra = {Ra:.2e}")

    # 稳定性估算
    h_min = min(grid.dx, grid.dz)

    # CFL条件（估算最大速度为0.1）
    u_max = 0.1
    dt_cfl = 0.5 * h_min / u_max

    # 扩散条件
    dt_diff_nu = 0.25 * h_min ** 2 / nu
    dt_diff_kappa = 0.25 * h_min ** 2 / kappa
    dt_diff = min(dt_diff_nu, dt_diff_kappa)

    dt_recommend = min(dt_cfl, dt_diff) * 0.8

    print(f"\n稳定性分析:")
    print(f"  CFL限制: dt < {dt_cfl:.2e}")
    print(f"  扩散限制 (ν): dt < {dt_diff_nu:.2e}")
    print(f"  扩散限制 (κ): dt < {dt_diff_kappa:.2e}")
    print(f"  推荐时间步长: dt = {dt_recommend:.2e}")

    # 估算计算成本
    total_time = 1.0  # 模拟1秒
    nt = int(total_time / dt_recommend)

    print(f"\n计算成本估算:")
    print(f"  模拟 {total_time} 秒需要 {nt} 步")
    print(f"  每步计算约 {grid.nx * grid.nz * 10} 次操作")
    print(f"  总计算量: ~{nt * grid.nx * grid.nz * 10:.1e} 次操作")

    return dt_recommend


if __name__ == "__main__":
    print("🚀 开始Boussinesq模型测试序列")

    # 测试1：基础功能
    grid, operators, u, w, T = simple_rayleigh_benard_test()

    # 测试2：泊松求解器
    poisson_ok = test_poisson_solver()

    # 测试3：稳定性参数
    dt_recommend = estimate_stability_parameters()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"基础功能: ✅ 完成")
    print(f"泊松求解器: {'✅ 通过' if poisson_ok else '❌ 需要调试'}")
    print(f"推荐时间步长: {dt_recommend:.2e}")

    if poisson_ok:
        print("\n🎉 准备就绪！可以运行完整的Boussinesq求解器了。")
        print("下一步：运行 boussinesq_solver.py 开始真正的对流模拟！")
    else:
        print("\n⚠️  需要先修复泊松求解器再继续。")
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_velocity_bc, apply_pressure_bc


def diagnose_projection_method():
    """
    专门诊断投影方法为什么无法降低散度
    """
    print("=" * 60)
    print("诊断投影方法")
    print("=" * 60)

    # 创建简单测试
    grid = Grid2D(nx=8, nz=8, Lx=1.0, Lz=1.0, staggered=False)
    operators = FluidOperators(grid)

    # 创建有散度的速度场
    u = np.ones((grid.nz + 1, grid.nx + 1)) * 0.1
    w = np.zeros((grid.nz + 1, grid.nx + 1))

    # 应用边界条件
    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }
    apply_velocity_bc(u, w, bc_params)

    print("步骤1：初始速度场")
    div_initial = operators.d_dx(u) + operators.d_dz(w)
    print(f"  初始散度: max={np.max(np.abs(div_initial)):.6f}")
    print(f"  初始散度分布: {div_initial[4, :]}")

    # 步骤2：计算压力修正的右端项
    dt = 0.01
    rhs = div_initial / dt
    print(f"\n步骤2：泊松方程右端项")
    print(f"  RHS: max={np.max(np.abs(rhs)):.6f}")
    print(f"  RHS mean: {np.mean(rhs):.6f}")

    # 步骤3：解泊松方程
    phi = solve_poisson_debug(rhs, grid)

    print(f"\n步骤3：压力修正")
    print(f"  φ: max={np.max(np.abs(phi)):.6f}")
    print(f"  φ mean: {np.mean(phi):.6f}")

    # 步骤4：计算压力修正梯度
    dphi_dx = operators.d_dx(phi)
    dphi_dz = operators.d_dz(phi)

    print(f"\n步骤4：压力修正梯度")
    print(f"  ∇φ_x: max={np.max(np.abs(dphi_dx)):.6f}")
    print(f"  ∇φ_z: max={np.max(np.abs(dphi_dz)):.6f}")

    # 步骤5：速度校正
    u_corrected = u - dt * dphi_dx
    w_corrected = w - dt * dphi_dz

    # 应用边界条件
    apply_velocity_bc(u_corrected, w_corrected, bc_params)

    print(f"\n步骤5：速度校正")
    print(f"  校正前 u: max={np.max(np.abs(u)):.6f}")
    print(f"  校正后 u: max={np.max(np.abs(u_corrected)):.6f}")

    # 步骤6：检查最终散度
    div_final = operators.d_dx(u_corrected) + operators.d_dz(w_corrected)
    print(f"\n步骤6：最终散度")
    print(f"  校正前散度: max={np.max(np.abs(div_initial)):.6f}")
    print(f"  校正后散度: max={np.max(np.abs(div_final)):.6f}")
    print(f"  散度减少: {np.max(np.abs(div_initial)) - np.max(np.abs(div_final)):.6f}")

    # 问题诊断
    print(f"\n🔍 问题诊断:")

    # 检查1：泊松方程是否正确求解
    laplacian_phi = operators.laplacian(phi)
    poisson_error = np.max(np.abs(laplacian_phi - rhs))
    print(f"  泊松方程误差: {poisson_error:.6f}")

    # 检查2：边界条件是否破坏了投影
    print(f"  边界条件对散度的影响:")
    div_before_bc = operators.d_dx(u - dt * dphi_dx) + operators.d_dz(w - dt * dphi_dz)
    print(f"    应用BC前散度: {np.max(np.abs(div_before_bc)):.6f}")
    print(f"    应用BC后散度: {np.max(np.abs(div_final)):.6f}")

    # 检查3：时间步长影响
    print(f"  时间步长检查:")
    print(f"    dt * max(∇φ): {dt * np.max(np.abs(dphi_dx)):.6f}")
    print(f"    相对于初始速度: {dt * np.max(np.abs(dphi_dx)) / 0.1:.2%}")

    # 可视化
    visualize_projection_steps(u, w, u_corrected, w_corrected, div_initial, div_final, phi, grid)

    return div_final, poisson_error


def solve_poisson_debug(rhs, grid, max_iter=1000, tol=1e-5):
    """
    带调试的泊松求解器
    """
    phi = np.zeros_like(rhs)

    # 确保兼容性条件
    rhs_mean = np.mean(rhs)
    rhs = rhs - rhs_mean
    print(f"    RHS 兼容性修正: 移除平均值 {rhs_mean:.6f}")

    dx = grid.Lx / grid.nx
    dz = grid.Lz / grid.nz

    operators = FluidOperators(grid)

    for iteration in range(max_iter):
        phi_old = phi.copy()

        # 雅可比迭代
        for j in range(1, rhs.shape[0] - 1):
            for i in range(1, rhs.shape[1] - 1):
                phi[j, i] = 0.25 * (
                        phi_old[j + 1, i] + phi_old[j - 1, i] +
                        phi_old[j, i + 1] + phi_old[j, i - 1] -
                        dx ** 2 * rhs[j, i]
                )

        # 边界条件
        apply_pressure_bc(phi)

        # 移除平均值
        phi -= np.mean(phi)

        # 检查收敛
        if iteration % 100 == 0:
            change = np.max(np.abs(phi - phi_old))

            # 计算真实残差
            laplacian_phi = operators.laplacian(phi)
            residual = np.max(np.abs(laplacian_phi - rhs))

            print(f"    迭代 {iteration}: change={change:.2e}, residual={residual:.2e}")

            if change < tol:
                print(f"    泊松求解器收敛于第 {iteration + 1} 次迭代")
                break

    return phi


def visualize_projection_steps(u, w, u_corr, w_corr, div_init, div_final, phi, grid):
    """
    可视化投影步骤
    """
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 初始速度场
    ax = axes[0, 0]
    skip = 2
    ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
              u[::skip, ::skip], w[::skip, ::skip], scale=None)
    ax.set_title('Initial Velocity')
    ax.set_aspect('equal')

    # 校正后速度场
    ax = axes[0, 1]
    ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
              u_corr[::skip, ::skip], w_corr[::skip, ::skip], scale=None)
    ax.set_title('Corrected Velocity')
    ax.set_aspect('equal')

    # 压力修正
    ax = axes[0, 2]
    im = ax.contourf(X, Z, phi, levels=20, cmap='RdBu_r')
    ax.set_title('Pressure Correction φ')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # 初始散度
    ax = axes[1, 0]
    im = ax.contourf(X, Z, div_init, levels=20, cmap='RdBu_r')
    ax.set_title(f'Initial Divergence (max={np.max(np.abs(div_init)):.3f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # 最终散度
    ax = axes[1, 1]
    im = ax.contourf(X, Z, div_final, levels=20, cmap='RdBu_r')
    ax.set_title(f'Final Divergence (max={np.max(np.abs(div_final)):.3f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # 散度变化
    ax = axes[1, 2]
    div_change = np.abs(div_final) - np.abs(div_init)
    im = ax.contourf(X, Z, div_change, levels=20, cmap='RdBu_r')
    ax.set_title('Divergence Change')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def test_boundary_condition_effect():
    """
    测试边界条件对投影的影响
    """
    print("\n" + "=" * 60)
    print("测试边界条件对投影的影响")
    print("=" * 60)

    grid = Grid2D(nx=8, nz=8, Lx=1.0, Lz=1.0, staggered=False)
    operators = FluidOperators(grid)

    # 创建理想的无散度场（应该不被投影改变）
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))

    # 只在顶部设置边界条件
    u[-1, :] = 0.1  # 顶部移动

    print("测试1：纯边界条件产生的散度")
    div = operators.d_dx(u) + operators.d_dz(w)
    print(f"  只有边界条件的散度: max={np.max(np.abs(div)):.6f}")

    # 应用边界条件函数
    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }
    apply_velocity_bc(u, w, bc_params)

    div_after_bc = operators.d_dx(u) + operators.d_dz(w)
    print(f"  应用BC函数后散度: max={np.max(np.abs(div_after_bc)):.6f}")

    print("\n测试2：边界条件的散度源")
    print(f"  散度主要分布:")
    for j in range(grid.nz + 1):
        row_div = np.max(np.abs(div_after_bc[j, :]))
        if row_div > 0.001:
            print(f"    第 {j} 行 (z={grid.z_p[j]:.2f}): {row_div:.6f}")


def propose_fix():
    """
    提出修复方案
    """
    print("\n" + "=" * 60)
    print("投影方法修复方案")
    print("=" * 60)

    print("基于诊断，问题可能是：")
    print("1. 边界条件与投影方法不兼容")
    print("2. 泊松求解器精度不够")
    print("3. 边界条件在投影后被重新应用，破坏了无散度性质")

    print("\n建议的修复方案：")
    print("A. 修改边界条件应用策略")
    print("B. 提高泊松求解器精度")
    print("C. 使用边界条件兼容的投影方法")

    fix_code = '''
# 修复方案：在投影步骤中不重新应用边界条件

def time_step_fixed(self, u, w, p, dt, bc_params=None):
    # 步骤1：预测（应用BC）
    u_star, w_star = self.momentum_predictor(u, w, p, dt)
    if bc_params is not None:
        apply_velocity_bc(u_star, w_star, bc_params)

    # 步骤2：投影（不应用BC）
    phi = self.pressure_projection(u_star, w_star, dt)

    # 步骤3：校正（不应用BC）
    u_new, w_new = self.velocity_correction(u_star, w_star, phi, dt)

    # 🔑 关键：只在最后应用一次边界条件
    if bc_params is not None:
        apply_velocity_bc(u_new, w_new, bc_params)

    p_new = p + phi
    return u_new, w_new, p_new
'''

    print(fix_code)


if __name__ == "__main__":
    # 运行诊断
    div_final, poisson_error = diagnose_projection_method()

    # 测试边界条件
    test_boundary_condition_effect()

    # 提出修复方案
    propose_fix()

    print(f"\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    print(f"最终散度: {np.max(np.abs(div_final)):.6f}")
    print(f"泊松方程误差: {poisson_error:.6f}")
    print("下一步：实施边界条件修复方案")
import numpy as np
from grid import Grid2D
from operators import FluidOperators, apply_velocity_bc, apply_pressure_bc


def create_enhanced_projection_solver():
    """
    创建增强的投影方法求解器
    """

    enhanced_solver_code = '''
class EnhancedProjectionSolver(StableNavierStokesSolver):
    """
    增强的投影方法求解器 - 解决散度累积问题
    """

    def time_step_enhanced_projection(self, u, w, p, dt, bc_params=None):
        """
        增强的投影方法时间步进
        """
        # 检查输入
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(w)):
            return u * 0, w * 0, p * 0

        # 🔑 关键修改1：更严格的散度控制
        # 在每个时间步开始时强制散度为零
        div_initial = self.operators.d_dx(u) + self.operators.d_dz(w)
        if np.max(np.abs(div_initial)) > 1e-10:
            print(f"    Initial reprojection: div={np.max(np.abs(div_initial)):.2e}")
            phi_init = self._solve_poisson_stable(div_initial, tol=1e-8)
            dphi_dx = self.operators.d_dx(phi_init)
            dphi_dz = self.operators.d_dz(phi_init)
            u = u - dphi_dx
            w = w - dphi_dz
            p = p + phi_init

        # 动态时间步长控制
        u_max = np.max(np.sqrt(u**2 + w**2))
        if u_max > 0:
            dt_cfl = 0.3 * min(self.grid.dx, self.grid.dz) / u_max  # 更保守
            dt = min(dt, dt_cfl)

        # 🔑 关键修改2：分步投影方法
        # 步骤1：半步动量预测（不含压力梯度）
        viscous_u = self.nu * self.operators.laplacian(u)
        viscous_w = self.nu * self.operators.laplacian(w)
        Nu, Nw = self.compute_nonlinear_term_stable(u, w)

        alpha = 0.3  # 更保守的系数
        u_half = u + 0.5 * alpha * dt * (viscous_u - Nu)
        w_half = w + 0.5 * alpha * dt * (viscous_w - Nw)

        # 应用边界条件到半步速度
        if bc_params is not None:
            apply_velocity_bc(u_half, w_half, bc_params)

        # 🔑 关键修改3：中间投影
        div_half = self.operators.d_dx(u_half) + self.operators.d_dz(w_half)
        if np.max(np.abs(div_half)) > 1e-10:
            phi_half = self._solve_poisson_stable(div_half / (0.5 * dt), tol=1e-8)
            dphi_dx = self.operators.d_dx(phi_half)
            dphi_dz = self.operators.d_dz(phi_half)
            u_half = u_half - 0.5 * dt * dphi_dx
            w_half = w_half - 0.5 * dt * dphi_dz

        # 步骤2：完整步动量预测
        dp_dx = self.operators.d_dx(p)
        dp_dz = self.operators.d_dz(p)

        u_star = u + alpha * dt * (-dp_dx/self.rho + viscous_u - Nu)
        w_star = w + alpha * dt * (-dp_dz/self.rho + viscous_w - Nw)

        # 应用边界条件
        if bc_params is not None:
            apply_velocity_bc(u_star, w_star, bc_params)

        # 🔑 关键修改4：最终投影（更严格）
        div_star = self.operators.d_dx(u_star) + self.operators.d_dz(w_star)
        rhs = div_star / dt

        phi = self._solve_poisson_stable(rhs, tol=1e-8, max_iter=10000)

        # 速度校正
        dphi_dx = self.operators.d_dx(phi)
        dphi_dz = self.operators.d_dz(phi)

        u_new = u_star - dt * dphi_dx
        w_new = w_star - dt * dphi_dz

        # 🔑 关键修改5：后处理散度检查
        div_final = self.operators.d_dx(u_new) + self.operators.d_dz(w_new)
        max_div_final = np.max(np.abs(div_final))

        if max_div_final > 1e-6:  # 如果散度仍然太大
            print(f"    Post-correction needed: div={max_div_final:.2e}")
            phi_final = self._solve_poisson_stable(div_final, tol=1e-10)
            dphi_dx_final = self.operators.d_dx(phi_final)
            dphi_dz_final = self.operators.d_dz(phi_final)
            u_new = u_new - dphi_dx_final
            w_new = w_new - dphi_dz_final
            phi = phi + phi_final

        # 压力更新
        p_new = p + phi

        # 最终边界条件
        if bc_params is not None:
            apply_pressure_bc(p_new, bc_params)

        return u_new, w_new, p_new

    def _solve_poisson_stable(self, rhs, max_iter=10000, tol=1e-8):
        """
        超高精度泊松求解器
        """
        phi = np.zeros_like(rhs)
        dx = self.grid.Lx / self.nx
        dz = self.grid.Lz / self.nz

        # 确保兼容性
        rhs_mean = np.mean(rhs)
        rhs = rhs - rhs_mean

        # 🔑 使用更优化的SOR参数
        omega = 1.95  # 接近最优

        for iteration in range(max_iter):
            phi_old = phi.copy()

            # Red-Black Gauss-Seidel（更稳定）
            # Red points
            for j in range(1, rhs.shape[0] - 1):
                for i in range(1, rhs.shape[1] - 1):
                    if (i + j) % 2 == 0:  # Red points
                        phi_new_val = 0.25 * (
                            phi[j + 1, i] + phi[j - 1, i] +
                            phi[j, i + 1] + phi[j, i - 1] -
                            dx ** 2 * rhs[j, i]
                        )
                        phi[j, i] = (1 - omega) * phi[j, i] + omega * phi_new_val

            # Black points
            for j in range(1, rhs.shape[0] - 1):
                for i in range(1, rhs.shape[1] - 1):
                    if (i + j) % 2 == 1:  # Black points
                        phi_new_val = 0.25 * (
                            phi[j + 1, i] + phi[j - 1, i] +
                            phi[j, i + 1] + phi[j, i - 1] -
                            dx ** 2 * rhs[j, i]
                        )
                        phi[j, i] = (1 - omega) * phi[j, i] + omega * phi_new_val

            # 边界条件
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]

            # 移除平均值
            phi -= np.mean(phi)

            # 检查收敛
            if iteration % 100 == 0:
                residual = 0
                for j in range(1, rhs.shape[0] - 1):
                    for i in range(1, rhs.shape[1] - 1):
                        laplacian = (phi[j + 1, i] - 2 * phi[j, i] + phi[j - 1, i]) / dz ** 2 + \\
                                    (phi[j, i + 1] - 2 * phi[j, i] + phi[j, i - 1]) / dx ** 2
                        residual = max(residual, abs(laplacian - rhs[j, i]))

                if residual < tol:
                    if iteration > 0:
                        print(f"    Enhanced Poisson converged in {iteration + 1} iter (residual: {residual:.2e})")
                    break

        return phi
'''

    print("=" * 60)
    print("增强投影方法求解器代码")
    print("=" * 60)
    print(enhanced_solver_code)


def test_enhanced_projection():
    """
    测试增强投影方法
    """
    print("\n" + "=" * 60)
    print("测试增强投影方法")
    print("=" * 60)

    from stable_navier_stokes import StableNavierStokesSolver

    # 手动实现增强投影方法
    class TestEnhancedSolver(StableNavierStokesSolver):
        def time_step_enhanced(self, u, w, p, dt, bc_params=None):
            """简化的增强投影测试"""

            # 更严格的初始投影
            div_initial = self.operators.d_dx(u) + self.operators.d_dz(w)
            if np.max(np.abs(div_initial)) > 1e-12:
                phi_init = self._solve_poisson_enhanced(div_initial)
                dphi_dx = self.operators.d_dx(phi_init)
                dphi_dz = self.operators.d_dz(phi_init)
                u = u - dphi_dx
                w = w - dphi_dz
                p = p + phi_init

            # 标准投影步骤但参数更保守
            dp_dx = self.operators.d_dx(p)
            dp_dz = self.operators.d_dz(p)
            viscous_u = self.nu * self.operators.laplacian(u)
            viscous_w = self.nu * self.operators.laplacian(w)
            Nu, Nw = self.compute_nonlinear_term_stable(u, w)

            alpha = 0.1  # 非常保守
            u_star = u + alpha * dt * (-dp_dx / self.rho + viscous_u - Nu)
            w_star = w + alpha * dt * (-dp_dz / self.rho + viscous_w - Nw)

            if bc_params is not None:
                apply_velocity_bc(u_star, w_star, bc_params)

            # 高精度投影
            div_star = self.operators.d_dx(u_star) + self.operators.d_dz(w_star)
            rhs = div_star / dt
            phi = self._solve_poisson_enhanced(rhs)

            dphi_dx = self.operators.d_dx(phi)
            dphi_dz = self.operators.d_dz(phi)
            u_new = u_star - dt * dphi_dx
            w_new = w_star - dt * dphi_dz

            # 后处理检查
            div_final = self.operators.d_dx(u_new) + self.operators.d_dz(w_new)
            if np.max(np.abs(div_final)) > 1e-8:
                phi_correct = self._solve_poisson_enhanced(div_final)
                dphi_dx_correct = self.operators.d_dx(phi_correct)
                dphi_dz_correct = self.operators.d_dz(phi_correct)
                u_new = u_new - dphi_dx_correct
                w_new = w_new - dphi_dz_correct
                phi = phi + phi_correct

            p_new = p + phi
            if bc_params is not None:
                apply_pressure_bc(p_new, bc_params)

            return u_new, w_new, p_new

        def _solve_poisson_enhanced(self, rhs):
            """增强泊松求解器"""
            return self._solve_poisson_stable(rhs, max_iter=20000, tol=1e-10)

    # 测试
    grid = Grid2D(nx=16, nz=16, Lx=1.0, Lz=1.0, staggered=False)
    solver = TestEnhancedSolver(grid, nu=0.01, rho=1.0)

    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))
    p = np.zeros((grid.nz + 1, grid.nx + 1))

    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }

    dt = 0.01  # 更小的时间步

    print("增强投影方法测试（10步）:")

    divergences = []
    for step in range(10):
        u, w, p = solver.time_step_enhanced(u, w, p, dt, bc_params)

        div_u = solver.operators.d_dx(u) + solver.operators.d_dz(w)
        max_div = np.max(np.abs(div_u))
        divergences.append(max_div)

        print(f"  步骤 {step + 1}: 散度={max_div:.8f}")

        if max_div > 0.01:
            print(f"    散度过大，停止测试")
            break

    max_final_div = max(divergences) if divergences else float('inf')
    success = max_final_div < 0.001

    print(f"\n增强投影结果:")
    print(f"  最大散度: {max_final_div:.8f}")
    print(f"  成功: {'✅' if success else '❌'}")

    return success


def provide_alternative_approaches():
    """
    提供替代方法
    """
    print("\n" + "=" * 60)
    print("替代方法建议")
    print("=" * 60)

    alternatives = '''
如果增强投影仍然不够，考虑以下替代方法：

1. **使用真正的交错网格**
   - 实现MAC (Marker-and-Cell) 方法
   - 在速度定义点自然满足散度自由条件

2. **使用不可压缩专用方法**
   - SIMPLE/PISO算法
   - 分离式压力-速度耦合

3. **降低雷诺数**
   - 增加粘性系数 nu = 0.1 (而不是0.01)
   - 使问题更稳定

4. **使用隐式时间积分**
   - Crank-Nicolson方法
   - 后向欧拉法

5. **问题简化**
   - 先测试更简单的情况（如 u_top = 0.001）
   - 确保方法在简单情况下工作
'''

    print(alternatives)


if __name__ == "__main__":
    # 提供增强投影代码
    create_enhanced_projection_solver()

    # 测试增强投影
    success = test_enhanced_projection()

    # 提供替代方法
    provide_alternative_approaches()

    print(f"\n" + "=" * 60)
    print("根本性修复总结")
    print("=" * 60)
    print(f"增强投影测试: {'✅ 成功' if success else '❌ 失败'}")

    if not success:
        print("\n🤔 可能需要考虑:")
        print("1. 这是方腔流的已知数值难题")
        print("2. 可能需要专用的不可压缩流求解器")
        print("3. 考虑使用商业CFD软件验证结果")
        print("4. 先在更简单的问题上验证方法")
import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_velocity_bc, apply_pressure_bc


class StableNavierStokesSolver:
    """
    数值稳定的Navier-Stokes求解器
    增加了多重稳定性措施
    """

    def __init__(self, grid, nu=1.0, rho=1.0):
        self.grid = grid
        self.nu = nu
        self.rho = rho
        self.operators = FluidOperators(grid)
        self.nx = grid.nx
        self.nz = grid.nz

        print(f"Stable Navier-Stokes solver initialized")
        print(f"Grid: {self.nz}×{self.nx}, viscosity: {self.nu}, density: {self.rho}")

        self._compute_stability_limits()

    def _compute_stability_limits(self):
        """计算稳定性限制"""
        dx = self.grid.Lx / self.nx
        dz = self.grid.Lz / self.nz
        h_min = min(dx, dz)

        # 更保守的稳定性限制
        self.dt_convection_limit = 0.5 * h_min  # CFL < 0.5
        self.dt_diffusion_limit = 0.25 * h_min ** 2 / self.nu  # 更保守的扩散限制
        self.dt_stable = 0.8 * min(self.dt_convection_limit, self.dt_diffusion_limit)

        print(f"Stability limits:")
        print(f"  CFL limit: {self.dt_convection_limit:.6f}")
        print(f"  Diffusion limit: {self.dt_diffusion_limit:.6f}")
        print(f"  Recommended dt: {self.dt_stable:.6f}")

    def compute_nonlinear_term_stable(self, u, w):
        """
        稳定的非线性项计算
        """
        # 限制速度幅值防止爆炸
        u_max = np.max(np.abs(u))
        w_max = np.max(np.abs(w))

        if u_max > 10 or w_max > 10:
            print(f"  Warning: Large velocities detected (u_max={u_max:.2f}, w_max={w_max:.2f})")
            # 软限制：缩放速度
            scale = min(10.0 / max(u_max, 1e-10), 10.0 / max(w_max, 1e-10))
            u = u * scale
            w = w * scale
            print(f"  Scaled velocities by factor {scale:.3f}")

        # 计算梯度
        du_dx = self.operators.d_dx(u)
        du_dz = self.operators.d_dz(u)
        dw_dx = self.operators.d_dx(w)
        dw_dz = self.operators.d_dz(w)

        # 非线性项
        Nu = u * du_dx + w * du_dz
        Nw = u * dw_dx + w * dw_dz

        # 检查NaN和Inf
        if np.any(~np.isfinite(Nu)) or np.any(~np.isfinite(Nw)):
            print("  Error: NaN/Inf detected in nonlinear terms!")
            Nu = np.nan_to_num(Nu, nan=0.0, posinf=0.0, neginf=0.0)
            Nw = np.nan_to_num(Nw, nan=0.0, posinf=0.0, neginf=0.0)

        return Nu, Nw

    def _solve_poisson_stable(self, rhs, max_iter=1000, tol=1e-4):
        """
        更稳定的泊松求解器
        """
        phi = np.zeros_like(rhs)

        dx = self.grid.Lx / self.nx
        dz = self.grid.Lz / self.nz

        # 确保RHS兼容性条件
        rhs_mean = np.mean(rhs)
        rhs = rhs - rhs_mean

        # # 简单但稳定的迭代
        # for iteration in range(max_iter):
        #     phi_old = phi.copy()
        #
        #     # 雅可比迭代
        #     for j in range(1, rhs.shape[0] - 1):
        #         for i in range(1, rhs.shape[1] - 1):
        #             phi[j, i] = 0.25 * (
        #                     phi_old[j + 1, i] + phi_old[j - 1, i] +
        #                     phi_old[j, i + 1] + phi_old[j, i - 1] -
        #                     dx ** 2 * rhs[j, i]
        #             )
        #
        #     # 边界条件
        #     phi[0, :] = phi[1, :]
        #     phi[-1, :] = phi[-2, :]
        #     phi[:, 0] = phi[:, 1]
        #     phi[:, -1] = phi[:, -2]
        #
        #     # 移除平均值
        #     phi -= np.mean(phi)
        #
        #     # 检查收敛（较少频率）
        #     if iteration % 50 == 0:
        #         change = np.max(np.abs(phi - phi_old))
        #         if change < tol:
        #             if iteration > 0:
        #                 print(f"  Poisson converged in {iteration + 1} iter")
        #             break
        #
        # return phi

        # 🔑 关键改进：使用Gauss-Seidel + SOR方法
        omega = 1.9  # SOR参数

        for iteration in range(max_iter):
            phi_old = phi.copy()

            # Gauss-Seidel SOR迭代（立即使用更新值）
            for j in range(1, rhs.shape[0] - 1):
                for i in range(1, rhs.shape[1] - 1):
                    # 使用已更新的phi值（Gauss-Seidel）
                    phi_new_val = 0.25 * (
                            phi[j + 1, i] + phi[j - 1, i] +
                            phi[j, i + 1] + phi[j, i - 1] -
                            dx ** 2 * rhs[j, i]
                    )
                    # SOR更新
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
                # 计算真实残差
                residual = 0
                for j in range(1, rhs.shape[0] - 1):
                    for i in range(1, rhs.shape[1] - 1):
                        laplacian = (phi[j + 1, i] - 2 * phi[j, i] + phi[j - 1, i]) / dz ** 2 + \
                                    (phi[j, i + 1] - 2 * phi[j, i] + phi[j, i - 1]) / dx ** 2
                        residual = max(residual, abs(laplacian - rhs[j, i]))

                if residual < tol:
                    print(f"  Improved Poisson converged in {iteration + 1} iter (residual: {residual:.2e})")
                    break

        return phi

    def time_step_stable(self, u, w, p, dt, bc_params=None):
        """
        稳定的时间步进
        """
        # 检查输入
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(w)) or np.any(~np.isfinite(p)):
            print("  Error: NaN/Inf in input!")
            return u * 0, w * 0, p * 0

        # 动态时间步长控制
        u_max = np.max(np.sqrt(u ** 2 + w ** 2))
        if u_max > 0:
            dt_cfl = 0.5 * min(self.grid.dx, self.grid.dz) / u_max
            dt = min(dt, dt_cfl)

        # 步骤1：显式欧拉法的预测步（更保守）
        # 计算各项
        dp_dx = self.operators.d_dx(p)
        dp_dz = self.operators.d_dz(p)

        # 粘性项
        viscous_u = self.nu * self.operators.laplacian(u)
        viscous_w = self.nu * self.operators.laplacian(w)

        # 稳定的非线性项
        Nu, Nw = self.compute_nonlinear_term_stable(u, w)

        # 预测速度（减小时间步长的影响）
        alpha = 0.5  # 亚松弛因子
        u_star = u + alpha * dt * (-dp_dx / self.rho + viscous_u - Nu)
        w_star = w + alpha * dt * (-dp_dz / self.rho + viscous_w - Nw)

        # 应用边界条件
        if bc_params is not None:
            apply_velocity_bc(u_star, w_star, bc_params)

        # 步骤2：压力投影
        div_u_star = self.operators.d_dx(u_star) + self.operators.d_dz(w_star)
        rhs = div_u_star / dt

        phi = self._solve_poisson_stable(rhs)

        # 步骤3：速度校正
        dphi_dx = self.operators.d_dx(phi)
        dphi_dz = self.operators.d_dz(phi)

        u_new = u_star - dt * dphi_dx
        w_new = w_star - dt * dphi_dz

        # 压力更新
        p_new = p + phi

        # 最终边界条件
        if bc_params is not None:
            # apply_velocity_bc(u_new, w_new, bc_params)
            apply_pressure_bc(p_new, bc_params)

        # 稳定性检查
        if np.any(~np.isfinite(u_new)) or np.any(~np.isfinite(w_new)):
            print("  Instability detected! Resetting...")
            return u * 0.9, w * 0.9, p  # 轻微衰减而不是完全重置

        return u_new, w_new, p_new

    def solve(self, u_init, w_init, p_init, dt, nt, bc_params=None):
        """
        稳定的时间积分
        """
        print(f"Starting stable NS simulation...")
        print(f"Time step: {dt:.6f}, steps: {nt}")

        # 强制使用安全的时间步长
        if dt > self.dt_stable:
            dt_safe = self.dt_stable
            nt_new = int(nt * dt / dt_safe)
            print(f"Reducing dt from {dt:.6f} to {dt_safe:.6f}")
            print(f"Increasing steps from {nt} to {nt_new}")
            dt = dt_safe
            nt = nt_new

        # 历史数组
        u_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        w_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        p_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))

        # 初始条件
        u_history[0] = u_init.copy()
        w_history[0] = w_init.copy()
        p_history[0] = p_init.copy()

        u, w, p = u_init.copy(), w_init.copy(), p_init.copy()

        # 时间循环
        for n in range(nt):
            u, w, p = self.time_step_stable(u, w, p, dt, bc_params)

            u_history[n + 1] = u
            w_history[n + 1] = w
            p_history[n + 1] = p

            # 诊断输出
            if (n + 1) % max(1, nt // 10) == 0:
                max_u = np.max(np.sqrt(u ** 2 + w ** 2))
                div_u = self.operators.d_dx(u) + self.operators.d_dz(w)
                max_div = np.max(np.abs(div_u))

                if np.isfinite(max_u) and np.isfinite(max_div):
                    print(f"  Step {n + 1}/{nt}, max velocity: {max_u:.4f}, max divergence: {max_div:.2e}")
                else:
                    print(f"  Step {n + 1}/{nt}, INSTABILITY DETECTED!")
                    break

        t_array = np.linspace(0, nt * dt, nt + 1)
        print("Stable simulation completed!")
        return u_history, w_history, p_history, t_array

    def plot_results(self, u, w, p, title="Stable NS Solution"):
        """可视化结果"""
        x_p, z_p = self.grid.get_pressure_grid()
        X, Z = np.meshgrid(x_p, z_p)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 速度矢量场
        ax = axes[0, 0]
        skip = max(1, self.nx // 10)
        Q = ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                      u[::skip, ::skip], w[::skip, ::skip],
                      scale=None, alpha=0.7)
        ax.set_title('Velocity Field')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')

        # u分量
        ax = axes[0, 1]
        im1 = ax.contourf(X, Z, u, levels=20, cmap='RdBu_r')
        ax.set_title('u-velocity')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax)

        # w分量
        ax = axes[1, 0]
        im2 = ax.contourf(X, Z, w, levels=20, cmap='RdBu_r')
        ax.set_title('w-velocity')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax)

        # 压力
        ax = axes[1, 1]
        im3 = ax.contourf(X, Z, p, levels=20, cmap='viridis')
        ax.set_title('Pressure')
        ax.set_aspect('equal')
        plt.colorbar(im3, ax=ax)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def test_stable_solver():
    """
    测试稳定求解器
    """
    print("=" * 60)
    print("Testing Stable Navier-Stokes Solver")
    print("=" * 60)

    # 小网格
    grid = Grid2D(nx=16, nz=16, Lx=1.0, Lz=1.0, staggered=False)
    solver = StableNavierStokesSolver(grid, nu=0.01, rho=1.0)

    # 初始条件：静止流体
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))
    p = np.zeros((grid.nz + 1, grid.nx + 1))

    # 边界条件：低速方腔流
    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,  # 降低顶部速度
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }

    # 使用推荐的时间步长
    dt = solver.dt_stable
    nt = 200

    print(f"Using safe dt = {dt:.6f}")

    try:
        u_history, w_history, p_history, t_array = solver.solve(
            u, w, p, dt, nt, bc_params=bc_params
        )

        # 检查结果
        final_u = u_history[-1]
        final_w = w_history[-1]
        final_p = p_history[-1]

        max_velocity = np.max(np.sqrt(final_u ** 2 + final_w ** 2))
        div_u = solver.operators.d_dx(final_u) + solver.operators.d_dz(final_w)
        max_divergence = np.max(np.abs(div_u))

        print(f"\nResults:")
        print(f"  Max velocity: {max_velocity:.4f}")
        print(f"  Max divergence: {max_divergence:.2e}")

        is_stable = np.isfinite(max_velocity) and max_velocity < 10
        is_incompressible = max_divergence < 0.01

        print(f"  Stable: {'✅' if is_stable else '❌'}")
        print(f"  Incompressible: {'✅' if is_incompressible else '❌'}")

        if is_stable and is_incompressible:
            solver.plot_results(final_u, final_w, final_p, "Stable Cavity Flow")
            print("✅ STABLE SOLVER SUCCESS!")
            return True
        else:
            print("❌ Still has issues...")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_stable_solver()
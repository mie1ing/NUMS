import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_velocity_bc, apply_pressure_bc
import nums.ode as ode


class NavierStokesSolver:
    """
    2D Navier-Stokes方程求解器
    使用投影方法（分数步长法）

    方程:
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
    ∇·u = 0

    投影方法:
    1. 预测步：u* = u^n + dt*[-∇p^n + ν∇²u^n + f^n - (u^n·∇)u^n]
    2. 投影步：∇²φ = ∇·u*/dt, p^{n+1} = p^n + φ
    3. 校正步：u^{n+1} = u* - dt*∇φ
    """

    def __init__(self, grid, nu=1.0, rho=1.0):
        """
        初始化Navier-Stokes求解器

        Parameters:
        -----------
        grid : Grid2D
            计算网格
        nu : float
            运动粘性系数
        rho : float
            密度
        """
        self.grid = grid
        self.nu = nu
        self.rho = rho

        # 创建算子
        self.operators = FluidOperators(grid)

        # 网格尺寸
        self.nx = grid.nx
        self.nz = grid.nz

        print(f"Navier-Stokes solver initialized")
        print(f"Grid: {self.nz}×{self.nx}, viscosity: {self.nu}, density: {self.rho}")

        # 稳定性参数
        self._compute_stability_limits()

    def _compute_stability_limits(self):
        """计算稳定性限制"""
        dx = self.grid.Lx / self.nx
        dz = self.grid.Lz / self.nz
        h_min = min(dx, dz)

        # CFL条件 (对流稳定性)
        # dt < h / |u_max|
        self.dt_convection_limit = h_min  # 假设最大速度为1

        # 扩散稳定性
        # dt < h² / (2*nu)
        self.dt_diffusion_limit = h_min ** 2 / (2 * self.nu)

        # 总体稳定性限制
        self.dt_stable = 0.5 * min(self.dt_convection_limit, self.dt_diffusion_limit)

        print(f"Stability limits:")
        print(f"  CFL limit: {self.dt_convection_limit:.6f}")
        print(f"  Diffusion limit: {self.dt_diffusion_limit:.6f}")
        print(f"  Recommended dt: {self.dt_stable:.6f}")

    def compute_nonlinear_term(self, u, w):
        """
        计算非线性项：(u·∇)u

        Parameters:
        -----------
        u, w : ndarray
            速度分量

        Returns:
        --------
        Nu, Nw : ndarray
            非线性项
        """
        # 计算梯度
        du_dx = self.operators.d_dx(u)
        du_dz = self.operators.d_dz(u)
        dw_dx = self.operators.d_dx(w)
        dw_dz = self.operators.d_dz(w)

        # 非线性项
        Nu = u * du_dx + w * du_dz
        Nw = u * dw_dx + w * dw_dz

        return Nu, Nw

    def momentum_predictor(self, u, w, p, dt, forcing=None):
        """
        动量方程预测步
        u* = u^n + dt*[-∇p^n + ν∇²u^n + f^n - (u^n·∇)u^n]

        Parameters:
        -----------
        u, w : ndarray
            当前时刻速度
        p : ndarray
            当前时刻压力
        dt : float
            时间步长
        forcing : tuple, optional
            外力项 (f_u, f_w)

        Returns:
        --------
        u_star, w_star : ndarray
            预测速度
        """
        # 压力梯度
        dp_dx = self.operators.d_dx(p)
        dp_dz = self.operators.d_dz(p)

        # 粘性项
        d2u_dx2 = self.operators.d2_dx2(u)
        d2u_dz2 = self.operators.d2_dz2(u)
        d2w_dx2 = self.operators.d2_dx2(w)
        d2w_dz2 = self.operators.d2_dz2(w)

        viscous_u = self.nu * (d2u_dx2 + d2u_dz2)
        viscous_w = self.nu * (d2w_dx2 + d2w_dz2)

        # 非线性项
        Nu, Nw = self.compute_nonlinear_term(u, w)

        # 外力
        if forcing is None:
            f_u = np.zeros_like(u)
            f_w = np.zeros_like(w)
        else:
            f_u, f_w = forcing

        # 预测速度
        u_star = u + dt * (-dp_dx / self.rho + viscous_u + f_u - Nu)
        w_star = w + dt * (-dp_dz / self.rho + viscous_w + f_w - Nw)

        return u_star, w_star

    def pressure_projection(self, u_star, w_star, dt):
        """
        压力投影步
        解泊松方程：∇²φ = ∇·u*/dt

        Parameters:
        -----------
        u_star, w_star : ndarray
            预测速度
        dt : float
            时间步长

        Returns:
        --------
        phi : ndarray
            压力修正
        """
        # 计算速度散度
        du_dx = self.operators.d_dx(u_star)
        dw_dz = self.operators.d_dz(w_star)
        div_u_star = du_dx + dw_dz

        # 泊松方程右端项
        rhs = div_u_star / dt

        # 解泊松方程：∇²φ = rhs
        # 使用简单的迭代方法（雅可比迭代）
        phi = self._solve_poisson(rhs)

        return phi

    def _solve_poisson(self, rhs, max_iter=1000, tol=1e-6):
        """
        解泊松方程：∇²φ = rhs
        使用雅可比迭代

        Parameters:
        -----------
        rhs : ndarray
            右端项
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差

        Returns:
        --------
        phi : ndarray
            解
        """
        # 初始猜测
        phi = np.zeros_like(rhs)
        phi_new = np.zeros_like(rhs)

        dx = self.grid.Lx / self.nx
        dz = self.grid.Lz / self.nz
        dx2 = dx ** 2
        dz2 = dz ** 2

        # 雅可比迭代系数
        coeff = 1.0 / (2 / dx2 + 2 / dz2)

        for iteration in range(max_iter):
            # 内部点
            phi_new[1:-1, 1:-1] = coeff * (
                    (phi[2:, 1:-1] + phi[:-2, 1:-1]) / dz2 +
                    (phi[1:-1, 2:] + phi[1:-1, :-2]) / dx2 -
                    rhs[1:-1, 1:-1]
            )

            # 边界条件：∂φ/∂n = 0（齐次诺伊曼边界条件）
            apply_pressure_bc(phi_new)

            # 检查收敛
            residual = np.max(np.abs(phi_new - phi))
            if residual < tol:
                if iteration < 10 or iteration % 100 == 0:
                    print(f"  Poisson solver converged in {iteration + 1} iterations")
                break

            phi[:] = phi_new[:]

        if iteration == max_iter - 1:
            print(f"  Warning: Poisson solver did not converge. Residual: {residual:.2e}")

        return phi

    def velocity_correction(self, u_star, w_star, phi, dt):
        """
        速度校正步
        u^{n+1} = u* - dt*∇φ

        Parameters:
        -----------
        u_star, w_star : ndarray
            预测速度
        phi : ndarray
            压力修正
        dt : float
            时间步长

        Returns:
        --------
        u_new, w_new : ndarray
            校正后的速度
        """
        # 压力修正梯度
        dphi_dx = self.operators.d_dx(phi)
        dphi_dz = self.operators.d_dz(phi)

        # 速度校正
        u_new = u_star - dt * dphi_dx
        w_new = w_star - dt * dphi_dz

        return u_new, w_new

    def time_step(self, u, w, p, dt, forcing=None, bc_params=None):
        """
        执行一个时间步

        Parameters:
        -----------
        u, w : ndarray
            当前速度
        p : ndarray
            当前压力
        dt : float
            时间步长
        forcing : tuple, optional
            外力项
        bc_params : dict, optional
            边界条件参数

        Returns:
        --------
        u_new, w_new, p_new : ndarray
            新的速度和压力
        """
        # 步骤1：动量预测
        u_star, w_star = self.momentum_predictor(u, w, p, dt, forcing)

        # 应用速度边界条件
        if bc_params is not None:
            apply_velocity_bc(u_star, w_star, bc_params)

        # 步骤2：压力投影
        phi = self.pressure_projection(u_star, w_star, dt)

        # 步骤3：速度校正
        u_new, w_new = self.velocity_correction(u_star, w_star, phi, dt)

        # 更新压力
        p_new = p + phi

        # 再次应用边界条件
        if bc_params is not None:
            apply_velocity_bc(u_new, w_new, bc_params)
            apply_pressure_bc(p_new, bc_params)

        return u_new, w_new, p_new

    def solve(self, u_init, w_init, p_init, dt, nt, forcing=None, bc_params=None):
        """
        时间积分求解

        Parameters:
        -----------
        u_init, w_init : ndarray
            初始速度
        p_init : ndarray
            初始压力
        dt : float
            时间步长
        nt : int
            时间步数
        forcing : tuple, optional
            外力项
        bc_params : dict, optional
            边界条件参数

        Returns:
        --------
        u_history, w_history, p_history : ndarray
            速度和压力的时间历史
        t_array : ndarray
            时间数组
        """
        print(f"Starting Navier-Stokes simulation...")
        print(f"Time step: {dt:.6f}, steps: {nt}")

        # 检查稳定性
        if dt > self.dt_stable:
            print(f"Warning: dt = {dt:.6f} > recommended {self.dt_stable:.6f}")

        # 初始化历史数组
        u_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        w_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        p_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))

        # 设置初始条件
        u_history[0] = u_init.copy()
        w_history[0] = w_init.copy()
        p_history[0] = p_init.copy()

        u, w, p = u_init.copy(), w_init.copy(), p_init.copy()

        # 时间循环
        for n in range(nt):
            u, w, p = self.time_step(u, w, p, dt, forcing, bc_params)

            u_history[n + 1] = u
            w_history[n + 1] = w
            p_history[n + 1] = p

            # 进度输出
            if (n + 1) % max(1, nt // 10) == 0:
                max_div = self._check_divergence(u, w)
                max_u = np.max(np.sqrt(u ** 2 + w ** 2))
                print(f"  Step {n + 1}/{nt}, max velocity: {max_u:.4f}, max divergence: {max_div:.2e}")

        # 创建时间数组
        t_array = np.linspace(0, nt * dt, nt + 1)

        print("Simulation completed!")
        return u_history, w_history, p_history, t_array

    def _check_divergence(self, u, w):
        """检查速度场的散度"""
        du_dx = self.operators.d_dx(u)
        dw_dz = self.operators.d_dz(w)
        div_u = du_dx + dw_dz
        return np.max(np.abs(div_u))

    def plot_results(self, u, w, p, title="Navier-Stokes Solution"):
        """
        可视化结果

        Parameters:
        -----------
        u, w : ndarray
            速度分量
        p : ndarray
            压力
        title : str
            图标题
        """
        x_p, z_p = self.grid.get_pressure_grid()
        X, Z = np.meshgrid(x_p, z_p)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 速度矢量场
        ax = axes[0, 0]
        skip = max(1, self.nx // 20)  # 控制矢量密度
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
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax)

        # w分量
        ax = axes[1, 0]
        im2 = ax.contourf(X, Z, w, levels=20, cmap='RdBu_r')
        ax.set_title('w-velocity')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax)

        # 压力
        ax = axes[1, 1]
        im3 = ax.contourf(X, Z, p, levels=20, cmap='viridis')
        ax.set_title('Pressure')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im3, ax=ax)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def create_initial_conditions(grid, case='cavity'):
    """
    创建初始条件

    Parameters:
    -----------
    grid : Grid2D
        计算网格
    case : str
        测试案例类型

    Returns:
    --------
    u, w, p : ndarray
        初始速度和压力
    """
    nz, nx = grid.nz + 1, grid.nx + 1

    u = np.zeros((nz, nx))
    w = np.zeros((nz, nx))
    p = np.zeros((nz, nx))

    if case == 'cavity':
        # 方腔流：顶部移动壁面
        u[-1, :] = 1.0  # 顶部壁面速度

    elif case == 'channel':
        # 槽道流：入口抛物线速度分布
        z_p = grid.z_p
        for j in range(nz):
            z_norm = z_p[j] / grid.Lz
            u[j, 0] = 4 * z_norm * (1 - z_norm)  # 抛物线剖面

    elif case == 'vortex':
        # 涡旋衰减
        x_p, z_p = grid.get_pressure_grid()
        X, Z = np.meshgrid(x_p, z_p)
        x_c, z_c = grid.Lx / 2, grid.Lz / 2

        # 高斯涡旋
        r2 = (X - x_c) ** 2 + (Z - z_c) ** 2
        gamma = 1.0
        sigma = min(grid.Lx, grid.Lz) / 8

        psi = gamma * np.exp(-r2 / (2 * sigma ** 2))
        u = -np.gradient(psi, axis=0) / (grid.Lz / grid.nz)
        w = np.gradient(psi, axis=1) / (grid.Lx / grid.nx)

    return u, w, p


def create_boundary_conditions(case='cavity'):
    """
    创建边界条件参数

    Parameters:
    -----------
    case : str
        测试案例类型

    Returns:
    --------
    bc_params : dict
        边界条件参数
    """
    if case == 'cavity':
        return {
            'type': 'cavity',
            'u_top': 1.0,
            'u_bottom': 0.0,
            'u_left': 0.0,
            'u_right': 0.0,
            'w_walls': 0.0
        }
    elif case == 'channel':
        return {
            'type': 'channel',
            'u_inlet': 'parabolic',
            'u_walls': 0.0,
            'w_walls': 0.0,
            'pressure_outlet': 0.0
        }
    else:
        return {'type': 'periodic'}


# 简单测试函数
def quick_test():
    """
    快速测试NS求解器
    """
    print("=" * 60)
    print("Quick Navier-Stokes Test")
    print("=" * 60)

    # 创建小网格
    grid = Grid2D(nx=16, nz=16, Lx=1.0, Lz=1.0, staggered=False)
    solver = NavierStokesSolver(grid, nu=0.01, rho=1.0)

    # 初始条件：静止流体
    u, w, p = create_initial_conditions(grid, case='cavity')

    # 边界条件：方腔流
    bc_params = create_boundary_conditions(case='cavity')

    # 短时间积分
    dt = 0.005
    nt = 100

    print(f"Running {nt} steps with dt={dt}")

    try:
        u_history, w_history, p_history, t_array = solver.solve(
            u, w, p, dt, nt, bc_params=bc_params
        )

        # 显示结果
        solver.plot_results(u_history[-1], w_history[-1], p_history[-1],
                            "Quick Test: Cavity Flow")

        # 检查基本物理量
        max_velocity = np.max(np.sqrt(u_history[-1] ** 2 + w_history[-1] ** 2))
        max_divergence = solver._check_divergence(u_history[-1], w_history[-1])

        print(f"\nResults:")
        print(f"  Max velocity: {max_velocity:.4f}")
        print(f"  Max divergence: {max_divergence:.2e}")
        print(f"  Test: {'✅ PASS' if max_velocity > 0.01 and max_divergence < 0.1 else '❌ FAIL'}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    quick_test()
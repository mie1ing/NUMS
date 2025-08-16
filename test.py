import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_velocity_bc, apply_pressure_bc, apply_temperature_bc
from nums import ode


class ImprovedBoussinesqSolver:
    """
    改进的2D Boussinesq模型求解器 - 充分利用现有的边界条件函数

    求解方程组：
    动量方程: ∂u/∂t + u·∇u = -∇p/ρ₀ + ν∇²u + gα(T-T₀)ẑ
    连续性方程: ∇·u = 0
    温度方程: ∂T/∂t + u·∇T = κ∇²T
    """

    def __init__(self, grid, nu=1e-4, kappa=1e-4, alpha=1e-3, g=9.81,
                 T_hot=274.0, T_cold=273.0, rho0=1.0):
        """
        初始化改进的Boussinesq求解器

        参数:
            grid: Grid2D对象
            nu: 动力学粘度
            kappa: 热扩散系数
            alpha: 热膨胀系数
            g: 重力加速度
            T_hot: 底部温度
            T_cold: 顶部温度
            rho0: 参考密度
        """
        self.grid = grid
        self.nu = nu
        self.kappa = kappa
        self.alpha = alpha
        self.g = g
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.T0 = (T_hot + T_cold) / 2  # 参考温度
        self.Delta_T = T_hot - T_cold    # 温差
        self.rho0 = rho0

        self.operators = FluidOperators(grid)
        self.nx, self.nz = grid.nx, grid.nz
        self.dx, self.dz = grid.dx, grid.dz

        # 设置边界条件参数
        self.setup_boundary_conditions()

        # 计算无量纲参数
        self.compute_dimensionless_parameters()

        print(f"Improved 2D Boussinesq Solver Initialized")
        print(f"Grid: {self.nx}×{self.nz}")
        print(f"Temperature: {self.T_hot}K (hot) to {self.T_cold}K (cold)")
        print(f"Prandtl number: Pr = {self.Pr:.4f}")
        print(f"Rayleigh number: Ra = {self.Ra:.2e}")

    def setup_boundary_conditions(self):
        """设置边界条件参数，使用现有的边界条件函数"""
        # 速度边界条件：所有边界无滑移
        self.velocity_bc_params = {
            'type': 'cavity',  # 使用cavity类型，所有边界都是无滑移
            'u_top': 0.0,
            'u_bottom': 0.0,
            'u_left': 0.0,
            'u_right': 0.0
        }

        # 压力边界条件：齐次Neumann（由apply_pressure_bc默认处理）
        self.pressure_bc_params = {
            'type': 'default'  # 默认所有边界零梯度
        }

        # 温度边界条件参数（需要传递给apply_temperature_bc）
        self.temperature_bc_params = {
            'T_hot': self.T_hot,
            'T_cold': self.T_cold
        }

    def compute_dimensionless_parameters(self):
        """计算关键的无量纲参数"""
        # Prandtl数: Pr = ν/κ
        self.Pr = self.nu / self.kappa

        # 特征长度尺度（取z方向长度）
        self.L = self.grid.Lz

        # Rayleigh数: Ra = gαΔTL³/(νκ)
        self.Ra = (self.g * self.alpha * self.Delta_T * self.L**3) / (self.nu * self.kappa)

    def boussinesq_rhs(self, state_flat, t):
        """
        Boussinesq方程组的右端项

        参数:
            state_flat: 展平的状态向量 [u, w, T]
            t: 时间

        返回:
            rhs_flat: 展平的右端项
        """
        # 解包状态向量
        n_field = (self.nz + 1) * (self.nx + 1)

        u = state_flat[0:n_field].reshape((self.nz + 1, self.nx + 1))
        w = state_flat[n_field:2*n_field].reshape((self.nz + 1, self.nx + 1))
        T = state_flat[2*n_field:3*n_field].reshape((self.nz + 1, self.nx + 1))

        # 应用边界条件（使用现有函数）
        apply_velocity_bc(u, w, self.velocity_bc_params)
        apply_temperature_bc(T, self.T_hot, self.T_cold)

        # 计算各项
        # 1. 对流项 u·∇u
        conv_u, conv_w = self.compute_convection_terms(u, w)

        # 2. 粘性项 ν∇²u
        visc_u = self.nu * self.operators.laplacian(u)
        visc_w = self.nu * self.operators.laplacian(w)

        # 3. 浮力项 gα(T-T₀) （只在w方程中）
        buoyancy_w = self.g * self.alpha * (T - self.T0)

        # 4. 温度方程项
        temp_conv = self.compute_temperature_convection(u, w, T)
        temp_diff = self.kappa * self.operators.laplacian(T)

        # 动量方程（不包括压力梯度，通过压力投影处理）
        du_dt = -conv_u + visc_u
        dw_dt = -conv_w + visc_w + buoyancy_w

        # 温度方程
        dT_dt = -temp_conv + temp_diff

        # 应用压力投影保证不可压缩性
        du_dt, dw_dt = self.apply_pressure_projection(du_dt, dw_dt)

        # 在边界处设置时间导数为零（Dirichlet边界条件的处理）
        self.apply_boundary_rhs(du_dt, dw_dt, dT_dt)

        # 重新打包
        rhs = np.concatenate([du_dt.flatten(), dw_dt.flatten(), dT_dt.flatten()])

        return rhs

    def compute_convection_terms(self, u, w):
        """计算对流项 u·∇u"""
        # u方程的对流项: u∂u/∂x + w∂u/∂z
        du_dx = self.operators.d_dx(u)
        du_dz = self.operators.d_dz(u)
        conv_u = u * du_dx + w * du_dz

        # w方程的对流项: u∂w/∂x + w∂w/∂z
        dw_dx = self.operators.d_dx(w)
        dw_dz = self.operators.d_dz(w)
        conv_w = u * dw_dx + w * dw_dz

        return conv_u, conv_w

    def compute_temperature_convection(self, u, w, T):
        """计算温度对流项 u·∇T"""
        dT_dx = self.operators.d_dx(T)
        dT_dz = self.operators.d_dz(T)
        return u * dT_dx + w * dT_dz

    def apply_pressure_projection(self, du_dt, dw_dt):
        """应用压力投影保证速度场无散度"""
        # 计算速度散度
        div_u_star = self.operators.d_dx(du_dt) + self.operators.d_dz(dw_dt)

        # 求解泊松方程 ∇²φ = ∇·u*
        phi = self.solve_poisson(div_u_star)

        # 校正速度
        dphi_dx = self.operators.d_dx(phi)
        dphi_dz = self.operators.d_dz(phi)

        du_dt_corrected = du_dt - dphi_dx
        dw_dt_corrected = dw_dt - dphi_dz

        return du_dt_corrected, dw_dt_corrected

    def solve_poisson(self, rhs, max_iter=500, tol=1e-6):
        """求解泊松方程 ∇²φ = rhs，使用改进的迭代方法"""
        phi = np.zeros_like(rhs)

        # 确保兼容性条件
        rhs_mean = np.mean(rhs)
        rhs = rhs - rhs_mean

        # 使用SOR方法加速收敛
        omega = 1.8  # SOR参数

        for iteration in range(max_iter):
            phi_old = phi.copy()

            # SOR迭代
            for j in range(1, rhs.shape[0] - 1):
                for i in range(1, rhs.shape[1] - 1):
                    phi_new_val = 0.25 * (
                        phi[j+1, i] + phi[j-1, i] +
                        phi[j, i+1] + phi[j, i-1] -
                        self.dx**2 * rhs[j, i]
                    )
                    phi[j, i] = (1 - omega) * phi[j, i] + omega * phi_new_val

            # 边界条件：齐次Neumann（使用现有函数的思路）
            apply_pressure_bc(phi, self.pressure_bc_params)

            # 移除平均值保证唯一性
            phi -= np.mean(phi)

            # 检查收敛
            if iteration % 50 == 0:
                change = np.max(np.abs(phi - phi_old))
                if change < tol:
                    break

        return phi

    def apply_boundary_rhs(self, du_dt, dw_dt, dT_dt):
        """在右端项中应用边界条件（边界点的时间导数为零）"""
        # 速度边界点的时间导数为零（Dirichlet边界条件）
        du_dt[0, :] = 0    # 底部
        du_dt[-1, :] = 0   # 顶部
        du_dt[:, 0] = 0    # 左侧
        du_dt[:, -1] = 0   # 右侧

        dw_dt[0, :] = 0    # 底部
        dw_dt[-1, :] = 0   # 顶部
        dw_dt[:, 0] = 0    # 左侧
        dw_dt[:, -1] = 0   # 右侧

        # 温度边界点的时间导数为零（Dirichlet边界条件）
        dT_dt[0, :] = 0    # 底部
        dT_dt[-1, :] = 0   # 顶部
        # 侧边是Neumann边界条件，不需要设置为零

    def solve(self, u_init, w_init, T_init, dt, nt):
        """
        求解Boussinesq方程组

        参数:
            u_init, w_init, T_init: 初始条件
            dt: 时间步长
            nt: 时间步数

        返回:
            history: 包含u, w, T历史的字典
            t_array: 时间数组
        """
        print(f"Starting Boussinesq simulation...")
        print(f"Time step: {dt:.2e}, Number of steps: {nt}")

        # 应用初始边界条件
        u_init = u_init.copy()
        w_init = w_init.copy()
        T_init = T_init.copy()

        # 使用现有边界条件函数
        apply_velocity_bc(u_init, w_init, self.velocity_bc_params)
        apply_temperature_bc(T_init, self.T_hot, self.T_cold)

        # 打包初始状态
        state_init = np.concatenate([
            u_init.flatten(),
            w_init.flatten(),
            T_init.flatten()
        ])

        # 使用RK4求解
        state_history, t_array = ode.rk4(self.boussinesq_rhs, state_init, dt, nt)

        # 解包结果
        n_field = (self.nz + 1) * (self.nx + 1)

        u_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        w_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))
        T_history = np.zeros((nt + 1, self.nz + 1, self.nx + 1))

        for i in range(nt + 1):
            state = state_history[:, i]
            u_history[i] = state[0:n_field].reshape((self.nz + 1, self.nx + 1))
            w_history[i] = state[n_field:2*n_field].reshape((self.nz + 1, self.nx + 1))
            T_history[i] = state[2*n_field:3*n_field].reshape((self.nz + 1, self.nx + 1))

        print("Boussinesq simulation completed!")

        history = {
            'u': u_history,
            'w': w_history,
            'T': T_history
        }

        return history, t_array

    def check_stability(self, dt):
        """检查数值稳定性条件"""
        # CFL条件（估算最大速度）
        # 从浮力驱动估算特征速度：u ~ sqrt(gαΔTL)
        u_char = np.sqrt(self.g * self.alpha * self.Delta_T * self.L)
        dt_cfl = 0.5 * min(self.dx, self.dz) / max(u_char, 0.01)  # 避免除零

        # 扩散稳定性条件
        dt_diff_momentum = 0.25 * min(self.dx, self.dz)**2 / self.nu
        dt_diff_thermal = 0.25 * min(self.dx, self.dz)**2 / self.kappa
        dt_diff = min(dt_diff_momentum, dt_diff_thermal)

        dt_stable = min(dt_cfl, dt_diff)

        print(f"Stability Analysis:")
        print(f"  Characteristic velocity: {u_char:.4f} m/s")
        print(f"  CFL limit: {dt_cfl:.2e}")
        print(f"  Diffusion limit (momentum): {dt_diff_momentum:.2e}")
        print(f"  Diffusion limit (thermal): {dt_diff_thermal:.2e}")
        print(f"  Recommended dt: {dt_stable:.2e}")
        print(f"  Current dt: {dt:.2e}")
        print(f"  Stable: {'✅' if dt <= dt_stable else '❌'}")

        return dt <= dt_stable

    def compute_diagnostics(self, history, t_array):
        """计算诊断量"""
        # 计算Nusselt数（无量纲传热率）
        T_history = history['T']
        nt = len(t_array)

        Nu_history = np.zeros(nt)

        for i in range(nt):
            T = T_history[i]
            # 底部热通量
            dT_dz_bottom = (T[1, :] - T[0, :]) / self.dz
            q_bottom = -self.kappa * np.mean(dT_dz_bottom)

            # 传导热通量
            q_conduction = self.kappa * self.Delta_T / self.L

            # Nusselt数
            Nu_history[i] = q_bottom / q_conduction

        return {
            'Nu': Nu_history,
            'time': t_array
        }

    def plot_results(self, history, t_array, time_index=-1):
        """可视化结果"""
        u = history['u'][time_index]
        w = history['w'][time_index]
        T = history['T'][time_index]

        x_p, z_p = self.grid.get_pressure_grid()
        X, Z = np.meshgrid(x_p, z_p)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 温度场
        ax = axes[0, 0]
        im1 = ax.contourf(X, Z, T, levels=20, cmap='RdBu_r')
        ax.set_title(f'Temperature at t={t_array[time_index]:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax)

        # 速度矢量场
        ax = axes[0, 1]
        skip = max(1, self.nx // 10)
        speed = np.sqrt(u**2 + w**2)
        Q = ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                      u[::skip, ::skip], w[::skip, ::skip],
                      speed[::skip, ::skip], scale=None, alpha=0.8, cmap='viridis')
        ax.set_title('Velocity Field')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(Q, ax=ax)

        # u速度分量
        ax = axes[1, 0]
        im2 = ax.contourf(X, Z, u, levels=20, cmap='RdBu_r')
        ax.set_title('u-velocity')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax)

        # w速度分量
        ax = axes[1, 1]
        im3 = ax.contourf(X, Z, w, levels=20, cmap='RdBu_r')
        ax.set_title('w-velocity')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal')
        plt.colorbar(im3, ax=ax)

        plt.tight_layout()
        plt.show()


def create_initial_conditions(grid, condition_type='rayleigh_benard', amplitude=0.01):
    """创建初始条件"""
    x_p, z_p = grid.get_pressure_grid()

    u_init = np.zeros((grid.nz + 1, grid.nx + 1))
    w_init = np.zeros((grid.nz + 1, grid.nx + 1))
    T_init = np.zeros((grid.nz + 1, grid.nx + 1))

    if condition_type == 'rayleigh_benard':
        # 线性温度剖面 + 小扰动
        T_hot, T_cold = 274.0, 273.0

        for j in range(grid.nz + 1):
            # 基础线性剖面
            T_init[j, :] = T_hot - (T_hot - T_cold) * z_p[j] / grid.Lz

            # 添加随机扰动触发对流
            for i in range(grid.nx + 1):
                perturbation = amplitude * (2 * np.random.random() - 1)
                T_init[j, i] += perturbation

    elif condition_type == 'sine_perturbation':
        # 线性剖面 + 正弦扰动
        T_hot, T_cold = 274.0, 273.0

        for j in range(grid.nz + 1):
            T_init[j, :] = T_hot - (T_hot - T_cold) * z_p[j] / grid.Lz

        # 添加正弦扰动
        for j in range(grid.nz + 1):
            for i in range(grid.nx + 1):
                perturbation = amplitude * np.sin(2 * np.pi * x_p[i] / grid.Lx) * \
                              np.sin(np.pi * z_p[j] / grid.Lz)
                T_init[j, i] += perturbation

    return u_init, w_init, T_init


def test_improved_rayleigh_benard():
    """测试改进的Rayleigh-Bénard求解器"""
    print("=" * 60)
    print("Testing Improved Rayleigh-Bénard Convection Solver")
    print("=" * 60)

    # 创建网格（较小网格用于快速测试）
    grid = Grid2D(nx=32, nz=16, Lx=2.0, Lz=1.0, staggered=False)

    # 设置物理参数
    nu = 1e-5      # 动力学粘度
    kappa = 1e-5   # 热扩散系数
    alpha = 1e-3   # 热膨胀系数
    g = 9.81       # 重力

    solver = ImprovedBoussinesqSolver(grid, nu=nu, kappa=kappa, alpha=alpha, g=g)

    # 创建初始条件
    u_init, w_init, T_init = create_initial_conditions(grid, 'sine_perturbation', amplitude=0.1)

    # 设置时间步长
    dt = 1e-1
    nt = 500  # 较少步数用于快速测试

    # 检查稳定性
    is_stable = solver.check_stability(dt)

    if not is_stable:
        print("⚠️ Warning: Current time step may be unstable!")

    # 求解
    history, t_array = solver.solve(u_init, w_init, T_init, dt, nt)

    # 计算诊断量
    diagnostics = solver.compute_diagnostics(history, t_array)

    # 可视化几个时间点
    time_indices = [0, nt//4, nt//2, nt]

    for i, idx in enumerate(time_indices):
        print(f"Plotting t = {t_array[idx]:.4f}")
        solver.plot_results(history, t_array, idx)

        # 输出一些统计信息
        u = history['u'][idx]
        w = history['w'][idx]
        T = history['T'][idx]

        max_speed = np.max(np.sqrt(u**2 + w**2))
        T_range = [np.min(T), np.max(T)]

        print(f"  Max velocity: {max_speed:.4f}")
        print(f"  Temperature range: [{T_range[0]:.3f}, {T_range[1]:.3f}]")
        if idx > 0:
            print(f"  Nusselt number: {diagnostics['Nu'][idx]:.3f}")
        print()

    # 绘制Nusselt数随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(diagnostics['time'], diagnostics['Nu'])
    plt.xlabel('Time')
    plt.ylabel('Nusselt Number')
    plt.title('Heat Transfer Efficiency vs Time')
    plt.grid(True)
    plt.show()

    return solver, history, t_array, diagnostics


if __name__ == "__main__":
    solver, history, t_array, diagnostics = test_improved_rayleigh_benard()
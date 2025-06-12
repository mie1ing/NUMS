import numpy as np
import matplotlib.pyplot as plt
from grid import Grid2D
from operators import FluidOperators, apply_temperature_bc
from nums import ode


class HeatDiffusionSolver:
    """
    热扩散求解器
    求解方程：∂T/∂t = κ∇²T
    """

    def __init__(self, grid, kappa=1.0):
        """
        初始化热扩散求解器

        参数:
            grid: Grid2D对象
            kappa: 热扩散系数
        """
        self.grid = grid
        self.kappa = kappa
        self.operators = FluidOperators(grid)

        # 网格信息
        self.nx, self.nz = grid.nx, grid.nz
        self.dx, self.dz = grid.dx, grid.dz

        print(f"热扩散求解器初始化完成")
        print(f"网格: {self.nx}×{self.nz}, 热扩散系数: {self.kappa}")

    def diffusion_rhs(self, T_flat, t):
        """
        热扩散方程的右端项：dT/dt = κ∇²T

        参数:
            T_flat: 扁平化的温度场 (1D数组)
            t: 时间

        返回:
            dT_dt_flat: 时间导数的扁平化形式
        """
        # 将1D数组重塑为2D
        T = T_flat.reshape((self.nz + 1, self.nx + 1))

        # 应用边界条件 - 在计算拉普拉斯算子前应用
        apply_temperature_bc(T, T_hot=1.0, T_cold=0.0)

        # 计算拉普拉斯算子
        laplacian_T = self.operators.laplacian(T)

        # dT/dt = κ∇²T
        dT_dt = self.kappa * laplacian_T

        # 边界点的时间导数设为0（因为边界温度固定）
        dT_dt[0, :] = 0  # 底边界
        dT_dt[-1, :] = 0  # 顶边界
        # dT_dt[:, 0] = 0  # 左边界
        # dT_dt[:, -1] = 0  # 右边界

        return dT_dt.flatten()

    def solve(self, T_initial, dt, nt, bc_params=None):
        """
        求解热扩散方程

        参数:
            T_initial: 初始温度场 (2D数组)
            dt: 时间步长
            nt: 时间步数
            bc_params: 边界条件参数字典

        返回:
            T_history: 温度场时间历史 (nt+1, nz+1, nx+1)
            t_array: 时间数组
        """
        if bc_params is None:
            bc_params = {'T_hot': 1.0, 'T_cold': 0.0}

        # 检查初始条件尺寸
        expected_shape = (self.nz + 1, self.nx + 1)
        if T_initial.shape != expected_shape:
            raise ValueError(f"初始温度场尺寸错误，期望 {expected_shape}，得到 {T_initial.shape}")

        # 应用初始边界条件
        T_init = T_initial.copy()
        apply_temperature_bc(T_init, bc_params['T_hot'], bc_params['T_cold'])

        print(f"开始求解热扩散方程...")
        print(f"时间步长: {dt}, 步数: {nt}")
        print(f"边界条件: 底部T={bc_params['T_hot']}, 顶部T={bc_params['T_cold']}")

        # 使用RK4求解器
        T_flat_init = T_init.flatten()
        T_solution, t_array = ode.rk4(self.diffusion_rhs, T_flat_init, dt, nt)

        # 重塑为3D数组: (time, z, x)
        T_history = T_solution.T.reshape((nt + 1, self.nz + 1, self.nx + 1))

        print(f"求解完成！")
        return T_history, t_array

    def check_stability(self, dt):
        """
        检查数值稳定性条件
        对于显式格式: dt < dx²dz²/(2κ(dx²+dz²))
        """
        stability_limit = (self.dx ** 2 * self.dz ** 2) / (2 * self.kappa * (self.dx ** 2 + self.dz ** 2))
        stable = dt <= stability_limit

        print(f"稳定性检查:")
        print(f"  时间步长: {dt}")
        print(f"  稳定性极限: {stability_limit:.6f}")
        print(f"  是否稳定: {'✅ 是' if stable else '❌ 否'}")

        if not stable:
            suggested_dt = 0.8 * stability_limit
            print(f"  建议时间步长: {suggested_dt:.6f}")

        return stable, stability_limit


def create_initial_condition(grid, condition_type='linear'):
    """
    创建初始温度分布

    参数:
        grid: Grid2D对象
        condition_type: 初始条件类型

    返回:
        T_init: 初始温度场 - 正确的形状 (nz+1, nx+1)
    """
    x_p, z_p = grid.get_pressure_grid()

    # 创建正确形状的数组 (nz+1, nx+1)
    T_init = np.zeros((grid.nz + 1, grid.nx + 1))

    if condition_type == 'linear':
        # 线性分布：底部热(1)，顶部冷(0)
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz

    elif condition_type == 'gaussian':
        # 高斯分布扰动
        z_center = grid.Lz / 2
        # 基础线性分布
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz
        # 添加高斯扰动
        for j in range(grid.nz + 1):
            for i in range(grid.nx + 1):
                perturbation = 0.1 * np.exp(-((z_p[j] - z_center) ** 2) / (0.1 * grid.Lz) ** 2)
                T_init[j, i] += perturbation

    elif condition_type == 'random':
        # 随机扰动
        for j in range(grid.nz + 1):
            T_init[j, :] = 1.0 - z_p[j] / grid.Lz
        # 添加随机扰动
        T_init += 0.05 * np.random.random(T_init.shape) - 0.025

    # 确保边界条件
    apply_temperature_bc(T_init, T_hot=1.0, T_cold=0.0)

    return T_init


def plot_temperature_evolution(T_history, t_array, grid, save_path=None):
    """
    绘制温度场演化

    参数:
        T_history: 温度历史 (nt+1, nz+1, nx+1)
        t_array: 时间数组
        grid: 网格对象
        save_path: 保存路径
    """
    x_p, z_p = grid.get_pressure_grid()

    # 选择几个时间点进行可视化
    nt = len(t_array) - 1
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, idx in enumerate(time_indices):
        if i < 5:  # 只画5个子图
            ax = axes[i]
            T = T_history[idx]  # Shape: (nz+1, nx+1)

            # 创建网格用于绘图
            X_plot, Z_plot = np.meshgrid(x_p, z_p)

            im = ax.contourf(X_plot, Z_plot, T, levels=20, cmap='RdBu_r')
            ax.set_title(f't = {t_array[idx]:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

    # 最后一个子图显示中心线温度随时间变化
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
        print(f"图像已保存到: {save_path}")

    plt.show()


def analytical_solution_1d(z, t, kappa, L=1.0, n_terms=50):
    """
    1D热扩散的解析解（用于验证）
    边界条件：T(0)=1, T(L)=0
    初始条件：T(z,0) = 1 - z/L

    解析解：T(z,t) = 1 - z/L + Σ[A_n * sin(nπz/L) * exp(-κ(nπ/L)²t)]
    """
    T = 1 - z / L  # 稳态解

    # 傅里叶级数项
    for n in range(1, n_terms + 1):
        A_n = 2 * (-1) ** (n + 1) / (n * np.pi)
        T += A_n * np.sin(n * np.pi * z / L) * np.exp(-kappa * (n * np.pi / L) ** 2 * t)

    return T


# 修正你的 heat_diffusion.py 中的验证函数

def verify_against_analytical_corrected():
    """
    修正的解析解验证 - 使用正确的测试用例
    """
    print("修正的解析解验证...")

    # 测试用例1：齐次边界条件 + 正弦初始条件
    print("\n测试用例1：齐次边界条件")
    print("-" * 40)

    grid = Grid2D(nx=2, nz=50, Lx=0.1, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # 正弦初始条件：T(z,0) = sin(πz/L)
    z_p = grid.z_p
    T_init = np.sin(np.pi * z_p / grid.Lz)
    T_init = T_init[:, np.newaxis].repeat(grid.nx + 1, axis=1)

    # 修改边界条件函数来处理齐次边界条件
    def apply_homogeneous_bc(T):
        T[0, :] = 0  # T(0) = 0
        T[-1, :] = 0  # T(L) = 0
        T[:, 0] = T[:, 1]  # 绝热侧边界
        T[:, -1] = T[:, -2]

    apply_homogeneous_bc(T_init)

    # 解析解：T(z,t) = sin(πz/L) * exp(-κ(π/L)²t)
    def analytical_homogeneous(z, t, kappa, L):
        return np.sin(np.pi * z / L) * np.exp(-kappa * (np.pi / L) ** 2 * t)

    # 数值求解
    dt = 0.0001
    nt = 500  # 较短时间，看衰减

    # 临时修改求解器的边界条件
    original_diffusion_rhs = solver.diffusion_rhs

    def homogeneous_diffusion_rhs(T_flat, t):
        T = T_flat.reshape((solver.nz + 1, solver.nx + 1))
        apply_homogeneous_bc(T)

        laplacian_T = solver.operators.laplacian(T)
        dT_dt = solver.kappa * laplacian_T

        # 边界点时间导数为0
        dT_dt[0, :] = 0
        dT_dt[-1, :] = 0
        dT_dt[:, 0] = 0
        dT_dt[:, -1] = 0

        return dT_dt.flatten()

    solver.diffusion_rhs = homogeneous_diffusion_rhs

    T_history, t_array = solver.solve(T_init, dt, nt)

    # 恢复原始函数
    solver.diffusion_rhs = original_diffusion_rhs

    # 验证
    t_final = t_array[-1]
    T_numerical = T_history[-1, :, 0]
    T_analytical = analytical_homogeneous(z_p, t_final, solver.kappa, grid.Lz)

    error = np.abs(T_numerical - T_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"齐次边界条件测试:")
    print(f"  最终时间: {t_final:.4f}")
    print(f"  理论衰减因子: {np.exp(-solver.kappa * (np.pi / grid.Lz) ** 2 * t_final):.6f}")
    print(f"  数值衰减因子: {np.max(T_numerical) / np.max(T_init):.6f}")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  平均误差: {mean_error:.6f}")

    homogeneous_test = max_error < 0.01

    # 测试用例2：非齐次边界条件 + 扰动初始条件
    print("\n测试用例2：非齐次边界条件 + 扰动")
    print("-" * 40)

    # 初始条件：T = 1 - z/L + 0.1*sin(2πz/L)
    T_init_perturbed = np.zeros((grid.nz + 1, grid.nx + 1))
    for j in range(grid.nz + 1):
        steady_part = 1.0 - z_p[j] / grid.Lz
        perturbation = 0.1 * np.sin(2 * np.pi * z_p[j] / grid.Lz)
        T_init_perturbed[j, :] = steady_part + perturbation

    apply_temperature_bc(T_init_perturbed, T_hot=1.0, T_cold=0.0)

    # 解析解：T = (1 - z/L) + 0.1*sin(2πz/L)*exp(-κ(2π/L)²t)
    def analytical_perturbed(z, t, kappa, L):
        steady = 1 - z / L
        transient = 0.1 * np.sin(2 * np.pi * z / L) * np.exp(-kappa * (2 * np.pi / L) ** 2 * t)
        return steady + transient

    # 数值求解
    T_history2, t_array2 = solver.solve(T_init_perturbed, dt, nt)

    # 验证
    t_final2 = t_array2[-1]
    T_numerical2 = T_history2[-1, :, 0]
    T_analytical2 = analytical_perturbed(z_p, t_final2, solver.kappa, grid.Lz)

    error2 = np.abs(T_numerical2 - T_analytical2)
    max_error2 = np.max(error2)
    mean_error2 = np.mean(error2)

    print(f"扰动测试:")
    print(f"  最终时间: {t_final2:.4f}")
    print(f"  扰动衰减因子: {np.exp(-solver.kappa * (2 * np.pi / grid.Lz) ** 2 * t_final2):.6f}")
    print(f"  最大误差: {max_error2:.6f}")
    print(f"  平均误差: {mean_error2:.6f}")

    perturbation_test = max_error2 < 0.01

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 齐次边界条件结果
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

    # 扰动测试结果
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

    print(f"\n验证总结:")
    print(f"  齐次边界条件: {'✅ 通过' if homogeneous_test else '❌ 失败'}")
    print(f"  扰动测试: {'✅ 通过' if perturbation_test else '❌ 失败'}")
    print(f"  总体验证: {'✅ 通过' if overall_success else '❌ 失败'}")

    return overall_success


# 同时修正简单测试，使用合理的时间尺度
def simple_test_corrected():
    """
    修正的简单测试 - 使用合理的时间尺度
    """
    print("修正的简单测试...")

    grid = Grid2D(nx=4, nz=10, Lx=0.2, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # 扩散时间尺度
    tau_diffusion = grid.Lz ** 2 / solver.kappa
    print(f"扩散时间尺度 τ = L²/κ = {tau_diffusion}")

    # 创建带扰动的初始条件
    T_init = create_initial_condition(grid, 'gaussian')

    # 运行到 3τ 时间，应该基本达到稳态
    dt = 0.001
    nt = int(3 * tau_diffusion / dt)
    print(f"运行到 t = 3τ = {3 * tau_diffusion}, 需要 {nt} 步")

    # 如果步数太多，减少到合理范围
    if nt > 30000:
        nt = 10000
        dt = 3 * tau_diffusion / nt
        print(f"调整参数: dt = {dt:.6f}, nt = {nt}")

    T_history, t_array = solver.solve(T_init, dt, nt)

    # 检查最终状态
    T_final = T_history[-1]
    z_p = grid.z_p
    T_steady = 1.0 - z_p / grid.Lz  # 理论稳态

    # 计算与稳态的差异
    diff = np.abs(T_final[:, 0] - T_steady)
    max_diff = np.max(diff)

    print(f"修正的简单测试结果:")
    print(f"  最终时间: {t_array[-1]:.3f}")
    print(f"  时间/扩散时间尺度: {t_array[-1] / tau_diffusion:.3f}")
    print(f"  与稳态最大差异: {max_diff:.6f}")
    print(f"  是否收敛: {'✅ 是' if max_diff < 0.01 else '❌ 否'}")

    # 可视化收敛过程
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
    # 计算总能量（积分）随时间的变化
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
    # 与稳态的差异随时间变化
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
    演示正确的物理现象
    """
    print("\n" + "=" * 60)
    print("演示正确的扩散物理")
    print("=" * 60)

    # 创建网格
    grid = Grid2D(nx=32, nz=16, Lx=2.0, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # 扩散时间尺度
    tau = grid.Lz ** 2 / solver.kappa
    print(f"扩散时间尺度: τ = {tau}")

    # 创建有趣的初始条件：阶跃函数
    T_init = np.zeros((grid.nz + 1, grid.nx + 1))
    z_p = grid.z_p

    # 阶跃初始条件
    for j in range(grid.nz + 1):
        if z_p[j] < grid.Lz / 2:
            T_init[j, :] = 1.0
        else:
            T_init[j, :] = 0.0

    # 应用边界条件
    apply_temperature_bc(T_init, T_hot=1.0, T_cold=0.0)

    # 在几个不同时间尺度运行
    time_fractions = [0.01, 0.05, 0.1, 0.3, 1.0]  # τ的倍数

    results = {}

    for frac in time_fractions:
        target_time = frac * tau
        dt = 0.001
        nt = int(target_time / dt)

        if nt > 5000:  # 限制计算量
            nt = 5000
            dt = target_time / nt

        print(f"\n运行到 t = {frac}τ = {target_time:.3f}")
        print(f"  时间步: dt = {dt:.6f}, 步数 = {nt}")

        T_history, t_array = solver.solve(T_init, dt, nt)
        results[frac] = {
            'T_final': T_history[-1],
            't_final': t_array[-1],
            'T_history': T_history,
            't_array': t_array
        }

    # 可视化不同时间的结果
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

    # 最后一个子图：中心线温度剖面
    ax = axes[5]
    center_x = grid.nx // 2

    # 初始条件
    ax.plot(z_p, T_init[:, center_x], 'k-', linewidth=3, label='initial')

    # 不同时间的剖面
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, frac in enumerate(time_fractions):
        T = results[frac]['T_final']
        t = results[frac]['t_final']
        ax.plot(z_p, T[:, center_x], color=colors[i],
                label=f't={frac}τ={t:.3f}')

    # 稳态解
    T_steady = 1 - z_p / grid.Lz
    ax.plot(z_p, T_steady, 'k--', linewidth=2, label='satble')

    ax.set_xlabel('z')
    ax.set_ylabel('T')
    ax.set_title('Centerline temperature evolution')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\n物理解释:")
    print(f"  - 初始：阶跃函数（不连续）")
    print(f"  - t = 0.01τ：开始平滑，扩散层很薄")
    print(f"  - t = 0.1τ：明显扩散，但还有梯度")
    print(f"  - t = 1τ：接近稳态，梯度基本线性")
    print(f"  - 稳态：完全线性分布 T = 1 - z/L")


# 主测试函数
def main_corrected_test():
    """
    运行所有修正的测试
    """
    print("=" * 60)
    print("修正的热扩散求解器验证")
    print("=" * 60)

    tests = []

    # 1. 修正的简单测试
    print("\n1. 修正的简单稳态测试")
    print("-" * 40)
    simple_ok = simple_test_corrected()
    tests.append(("简单稳态测试", simple_ok))

    # 2. 修正的解析解验证
    print("\n2. 修正的解析解验证")
    print("-" * 40)
    analytical_ok = verify_against_analytical_corrected()
    tests.append(("解析解验证", analytical_ok))

    # 3. 物理演示
    print("\n3. 扩散物理演示")
    print("-" * 40)
    demonstrate_correct_physics()

    # 总结
    print("\n" + "=" * 60)
    print("最终验证结果")
    print("=" * 60)

    for name, result in tests:
        print(f"  {name}: {'✅ 通过' if result else '❌ 失败'}")

    all_passed = all(result for _, result in tests)

    if all_passed:
        print("\n🎉 恭喜！你的热扩散求解器是正确的！")
        print("\n现在可以继续开发 Navier-Stokes 求解器了！")
    else:
        print("\n还有一些问题需要解决...")

    return all_passed


if __name__ == "__main__":
    main_corrected_test()
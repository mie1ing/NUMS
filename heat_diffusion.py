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

        # 应用边界条件
        apply_temperature_bc(T, T_hot=1.0, T_cold=0.0)

        # 计算拉普拉斯算子
        laplacian_T = self.operators.laplacian(T)

        # dT/dt = κ∇²T
        dT_dt = self.kappa * laplacian_T

        # 边界点的时间导数设为0（因为边界温度固定）
        dT_dt[0, :] = 0  # 底边界
        dT_dt[-1, :] = 0  # 顶边界
        dT_dt[:, 0] = 0  # 左边界
        dT_dt[:, -1] = 0  # 右边界

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
        T_init: 初始温度场
    """
    x_p, z_p = grid.get_pressure_grid()
    X, Z = np.meshgrid(x_p, z_p, indexing='ij')
    T_init = np.zeros_like(X)

    if condition_type == 'linear':
        # 线性分布：底部热(1)，顶部冷(0)
        # 使用广播，避免手动循环
        T_init = 1.0 - Z / grid.Lz

    elif condition_type == 'gaussian':
        # 高斯分布扰动
        z_center = grid.Lz / 2
        # 基础线性分布
        T_init = 1.0 - Z / grid.Lz
        # 添加高斯扰动
        perturbation = 0.1 * np.exp(-((Z - z_center) ** 2) / (0.1 * grid.Lz) ** 2)
        T_init += perturbation

    elif condition_type == 'random':
        # 随机扰动
        # 基础线性分布
        T_init = 1.0 - Z / grid.Lz
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
            T = T_history[idx]

            im = ax.contourf(x_p, z_p, T, levels=20, cmap='RdBu_r')
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


def analytical_solution_1d(z, t, kappa, L=1.0, n_terms=100):
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


def verify_against_analytical():
    """
    与解析解对比验证
    """
    print("与1D解析解对比验证...")

    # 创建1D风格的网格（nx=1）
    grid = Grid2D(nx=1, nz=50, Lx=0.1, Lz=1.0, staggered=False)
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # 创建初始条件
    T_init = create_initial_condition(grid, 'linear')

    # 修复：如果形状不匹配，进行转置
    if T_init.shape != (grid.nz + 1, grid.nx + 1):
        T_init = T_init.T

    # 数值求解
    dt = 0.001
    nt = 1000
    T_history, t_array = solver.solve(T_init, dt, nt)

    # 与解析解比较
    z_p = grid.z_p
    t_final = t_array[-1]

    T_numerical = T_history[-1, :, 0]  # 最终时刻的数值解
    T_analytical = analytical_solution_1d(z_p, t_final, solver.kappa, grid.Lz)

    # 计算误差
    error = np.abs(T_numerical - T_analytical)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"验证结果:")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  平均误差: {mean_error:.6f}")

    # 绘制比较图
    plt.figure(figsize=(10, 6))
    plt.plot(z_p, T_numerical, 'bo-', label='numerical solution', markersize=4)
    plt.plot(z_p, T_analytical, 'r-', label='analytical solutions', linewidth=2)
    plt.plot(z_p, error, 'g--', label='error', linewidth=1)
    plt.xlabel('z')
    plt.ylabel('T')
    plt.title(f'Comparison of numerical and analytical solutions (t = {t_final:.3f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return max_error < 0.01  # 误差应该很小


# 测试和演示函数
def demo_heat_diffusion():
    """
    热扩散求解器演示
    """
    print("=" * 50)
    print("热扩散求解器演示")
    print("=" * 50)

    # 创建网格
    grid = Grid2D(nx=32, nz=16, Lx=2.0, Lz=1.0, staggered=False)
    grid.info()

    # 创建求解器
    solver = HeatDiffusionSolver(grid, kappa=0.1)

    # 设置时间参数
    dt = 0.001
    nt = 2000

    # 检查稳定性
    stable, _ = solver.check_stability(dt)
    if not stable:
        print("警告：时间步长可能过大，结果可能不稳定")

    # 创建初始条件
    T_init = create_initial_condition(grid, 'gaussian')

    # 修复：转置温度场以匹配期望的形状 (nz+1, nx+1)
    T_init = T_init.T

    print(f"\n初始温度场统计:")
    print(f"  最小值: {np.min(T_init):.3f}")
    print(f"  最大值: {np.max(T_init):.3f}")
    print(f"  平均值: {np.mean(T_init):.3f}")

    # 求解
    T_history, t_array = solver.solve(T_init, dt, nt)

    print(f"\n最终温度场统计:")
    T_final = T_history[-1]
    print(f"  最小值: {np.min(T_final):.3f}")
    print(f"  最大值: {np.max(T_final):.3f}")
    print(f"  平均值: {np.mean(T_final):.3f}")

    # 可视化
    plot_temperature_evolution(T_history, t_array, grid)

    print("\n演示完成！")


if __name__ == "__main__":
    # 运行验证
    print("验证热扩散求解器...")
    if verify_against_analytical():
        print("✅ 验证通过")
    else:
        print("❌ 验证失败")

    print("\n" + "=" * 50)

    # 运行演示
    demo_heat_diffusion()
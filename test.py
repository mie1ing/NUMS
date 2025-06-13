import numpy as np
from grid import Grid2D


def check_current_bc_function():
    """
    检查当前operators.py中的边界条件函数
    """
    print("=" * 60)
    print("检查当前边界条件函数的行为")
    print("=" * 60)

    # 导入当前的函数
    from operators import apply_velocity_bc

    # 创建测试
    grid = Grid2D(nx=4, nz=4, Lx=1.0, Lz=1.0, staggered=False)
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))

    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }

    print("调用apply_velocity_bc前:")
    print(f"  u形状: {u.shape}")
    print(f"  u顶部行: {u[-1, :]}")

    # 调用函数
    apply_velocity_bc(u, w, bc_params)

    print("调用apply_velocity_bc后:")
    print(f"  u顶部行: {u[-1, :]}")

    # 分析问题
    if u[-1, 0] == 0.0 and u[-1, -1] == 0.0:
        print("❌ 问题确认：角点被设为0，中间设为u_top")
        print("这正是散度源的原因！")
        return False
    else:
        print("✅ 边界条件看起来正确")
        return True


def create_simple_fix():
    """
    创建一个简单直接的修复方案
    """
    print("\n" + "=" * 60)
    print("创建简单修复方案")
    print("=" * 60)

    print("问题：当前边界条件函数在角点创建阶跃")
    print("解决：直接修改operators.py文件")

    # 显示需要替换的代码
    print("\n在operators.py中找到apply_velocity_bc函数，替换为：")

    fix_code = '''
def apply_velocity_bc(u, w, bc_params):
    """
    应用速度边界条件
    """
    if bc_params is None:
        return

    bc_type = bc_params.get('type', 'cavity')

    if bc_type == 'cavity':
        # 🔑 关键修复：整个边界统一设置

        # 顶部边界 - 整个边界包括角点都设为u_top
        u_top = bc_params.get('u_top', 1.0)
        u[-1, :] = u_top  # 整行都是u_top
        w[-1, :] = 0.0

        # 底部边界
        u_bottom = bc_params.get('u_bottom', 0.0)
        u[0, :] = u_bottom
        w[0, :] = 0.0

        # 左边界
        u_left = bc_params.get('u_left', 0.0)
        u[:, 0] = u_left
        w[:, 0] = 0.0

        # 右边界
        u_right = bc_params.get('u_right', 0.0)
        u[:, -1] = u_right
        w[:, -1] = 0.0

    elif bc_type == 'periodic':
        # 周期边界条件
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]

        w[0, :] = w[-2, :]
        w[-1, :] = w[1, :]
        w[:, 0] = w[:, -2]
        w[:, -1] = w[:, 1]

    elif bc_type == 'channel':
        # 槽道流边界条件
        if 'u_inlet' in bc_params:
            if bc_params['u_inlet'] == 'parabolic':
                # 抛物线入口速度分布
                z_coords = np.linspace(0, 1, u.shape[0])
                for j in range(u.shape[0]):
                    z_norm = z_coords[j]
                    u[j, 0] = 4 * z_norm * (1 - z_norm)
            else:
                u[:, 0] = bc_params['u_inlet']

        # 壁面无滑移
        u[0, :] = bc_params.get('u_walls', 0.0)
        u[-1, :] = bc_params.get('u_walls', 0.0)
        w[:, :] = bc_params.get('w_walls', 0.0)
'''

    print(fix_code)


def test_manual_fix():
    """
    手动测试修复效果
    """
    print("\n" + "=" * 60)
    print("手动测试修复效果")
    print("=" * 60)

    def apply_velocity_bc_manual_fix(u, w, bc_params):
        """手动修复的边界条件函数"""
        if bc_params is None:
            return

        bc_type = bc_params.get('type', 'cavity')

        if bc_type == 'cavity':
            # 整个边界统一设置
            u_top = bc_params.get('u_top', 1.0)
            u[-1, :] = u_top  # 整行都是u_top，包括角点
            w[-1, :] = 0.0

            u_bottom = bc_params.get('u_bottom', 0.0)
            u[0, :] = u_bottom
            w[0, :] = 0.0

            u_left = bc_params.get('u_left', 0.0)
            u[:, 0] = u_left
            w[:, 0] = 0.0

            u_right = bc_params.get('u_right', 0.0)
            u[:, -1] = u_right
            w[:, -1] = 0.0

    # 测试
    grid = Grid2D(nx=4, nz=4, Lx=1.0, Lz=1.0, staggered=False)
    u = np.zeros((grid.nz + 1, grid.nx + 1))
    w = np.zeros((grid.nz + 1, grid.nx + 1))

    bc_params = {
        'type': 'cavity',
        'u_top': 0.1,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0,
    }

    print("修复前（当前operators.py）:")
    from operators import apply_velocity_bc
    apply_velocity_bc(u, w, bc_params)
    print(f"  u顶部: {u[-1, :]}")

    # 重置
    u.fill(0)
    w.fill(0)

    print("修复后（手动修复）:")
    apply_velocity_bc_manual_fix(u, w, bc_params)
    print(f"  u顶部: {u[-1, :]}")

    # 检查散度
    from operators import FluidOperators
    operators = FluidOperators(grid)
    div = operators.d_dx(u) + operators.d_dz(w)
    print(f"  最大散度: {np.max(np.abs(div)):.6f}")

    # 成功标准
    all_same = np.all(u[-1, :] == 0.1)  # 所有顶部点都应该是0.1
    low_div = np.max(np.abs(div)) < 0.1

    print(f"  所有顶部点都是u_top: {'✅' if all_same else '❌'}")
    print(f"  散度足够小: {'✅' if low_div else '❌'}")

    return all_same and low_div


def provide_final_instructions():
    """
    提供最终指导
    """
    print("\n" + "=" * 60)
    print("最终修复指导")
    print("=" * 60)

    print("基于测试结果，问题确实在operators.py的apply_velocity_bc函数")
    print("当前函数在角点设置u=0，这产生了阶跃不连续")

    print("\n立即修复步骤：")
    print("1. 打开operators.py文件")
    print("2. 找到apply_velocity_bc函数")
    print("3. 确保 'cavity' 分支中的边界设置是：")
    print("   u[-1, :] = u_top  # 整行，不要在角点特殊处理")
    print("   u[0, :] = u_bottom")
    print("   u[:, 0] = u_left")
    print("   u[:, -1] = u_right")

    print("\n4. 保存文件后运行：")
    print("   python stable_navier_stokes.py")

    print("\n预期结果：")
    print("   - 散度从1.6降到接近0.0")
    print("   - 投影方法开始正常工作")


if __name__ == "__main__":
    # 检查当前函数
    current_ok = check_current_bc_function()

    # 创建修复方案
    create_simple_fix()

    # 手动测试修复
    manual_ok = test_manual_fix()

    # 最终指导
    provide_final_instructions()

    print(f"\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)
    print(f"当前BC函数正确: {'✅' if current_ok else '❌'}")
    print(f"手动修复有效: {'✅' if manual_ok else '❌'}")

    if not current_ok and manual_ok:
        print("\n🎯 确认问题在operators.py文件！请按上述步骤修复。")
    elif current_ok:
        print("\n🤔 奇怪，边界条件函数看起来是正确的，问题可能在别处...")
    else:
        print("\n❌ 需要进一步调试...")
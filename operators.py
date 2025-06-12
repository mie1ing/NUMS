import numpy as np
from nums import fdm


class FluidOperators:
    """
    fluid dynamics defferential operators
    deal with gradient, divergence, and Laplacian operators on staggered grids
    """

    def __init__(self, grid):
        """
        initialize operators, need a Grid2D object
        """
        self.grid = grid
        self.dx = grid.dx
        self.dz = grid.dz
        self.staggered = grid.staggered

    def divergence(self, u, w):
        """
        calculation of divergence of velocity field: ∇·u = ∂u/∂x + ∂w/∂z

        parameters:
            u: u velocity component (nz+1, nx+2) for staggered grid
            w: w velocity component (nz+2, nx+1) for staggered grid

        return:
            div: divergence field (nz+1, nx+1) define on pressure point
        """
        if self.staggered:
            # staggered grid: divergence calculated on pressure point
            nz, nx = self.grid.nz, self.grid.nx
            div = np.zeros((nz + 1, nx + 1))

            # ∂u/∂x: u staggered in x direction
            # u[j, i] is the value at point (x[i-1/2], z[j])
            # calculate ∂u/∂x at pressure points (x[i], z[j])
            div += (u[:, 1:] - u[:, :-1]) / self.dx

            # ∂w/∂z: w staggered in z direction
            # w[j, i] is the value at point (x[i], z[j-1/2])
            # calculate ∂w/∂z at pressure points (x[i], z[j])
            div += (w[1:, :] - w[:-1, :]) / self.dz

            return div
        else:
            # non-staggered grid: use standard centered difference
            du_dx = fdm.pd1_2d(u, axis=1, h=self.dx, method='centered', bc='non-periodic')
            dw_dz = fdm.pd1_2d(w, axis=0, h=self.dz, method='centered', bc='non-periodic')
            return du_dx + dw_dz

    def gradient(self, p):
        """
        calculation of gradient of scalar field: ∇p = (∂p/∂x, ∂p/∂z)

        parameter:
            p: pressure field (nz+1, nx+1)

        return:
            grad_x: gradient in x direction, defined on u velocity point (nz+1, nx+2)
            grad_z: gradient in z direction, defined on w velocity point (nz+2, nx+1)
        """
        if self.staggered:
            nz, nx = self.grid.nz, self.grid.nx

            # gradient in x direction (apply to u velocity point)
            grad_x = np.zeros((nz + 1, nx + 2))
            # inner points: standard centered difference
            grad_x[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / self.dx
            # 边界点：外推或边界条件
            # boundary points: extrapolation or boundary condition
            grad_x[:, 0] = (p[:, 0] - p[:, 0]) / self.dx  # left boundary, normally 0
            grad_x[:, -1] = (p[:, -1] - p[:, -1]) / self.dx  # rights boundary, normally 0

            # gradient in z direction (apply to w velocity point)
            grad_z = np.zeros((nz + 2, nx + 1))
            # internal points: standard centered difference
            grad_z[1:-1, :] = (p[1:, :] - p[:-1, :]) / self.dz
            # boundary points
            grad_z[0, :] = (p[0, :] - p[0, :]) / self.dz  # bottom boundary
            grad_z[-1, :] = (p[-1, :] - p[-1, :]) / self.dz  # top boundary

            return grad_x, grad_z
        else:
            # non-staggered grid
            grad_x = fdm.pd1_2d(p, axis=1, h=self.dx, method='centered', bc='non-periodic')
            grad_z = fdm.pd1_2d(p, axis=0, h=self.dz, method='centered', bc='non-periodic')
            return grad_x, grad_z

    def laplacian(self, f):
        """
        calculation of Laplacian of scalar field: ∇²f = (∂²f/∂x² + ∂²f/∂z²)

        parameter:
            f: scalar field (nz+1, nx+1)

        return:
            lap: result of laplacian operator (nz+1, nx+1)
        """
        d2f_dx2 = fdm.pd2_2d(f, axis=1, h=self.dx, method='centered', bc='non-periodic')
        d2f_dz2 = fdm.pd2_2d(f, axis=0, h=self.dz, method='centered', bc='non-periodic')
        return d2f_dx2 + d2f_dz2

    def advection_u(self, u, w, phi):
        """
        calculation of u·∇φ, (the advection term)

        parameters:
            u, w: velocity field
            phi: advected scalar field

        return:
            adv: advective term
        """
        dphi_dx = fdm.pd1_2d(phi, axis=1, h=self.dx, method='centered', bc='non-periodic')
        dphi_dz = fdm.pd1_2d(phi, axis=0, h=self.dz, method='centered', bc='non-periodic')

        if self.staggered:
            # need to interpolate velocity to scalar points
            # here we use simple average, it can be more accurate
            u_at_p = 0.5 * (u[:, :-1] + u[:, 1:])
            w_at_p = 0.5 * (w[:-1, :] + w[1:, :])
            return u_at_p * dphi_dx + w_at_p * dphi_dz
        else:
            return u * dphi_dx + w * dphi_dz

    def interpolate_to_pressure_points(self, u, w):
        """
        interpolate velocity on staggered grids to pressure points
        to calculate energy and visualization

        return:
            u_p, w_p: velocity components interpolated to pressure points
        """
        if self.staggered:
            # u速度插值到压力点
            u_p = 0.5 * (u[:, :-1] + u[:, 1:])
            # w速度插值到压力点
            w_p = 0.5 * (w[:-1, :] + w[1:, :])
            return u_p, w_p
        else:
            return u, w

    def d_dx(self, f):
        """计算x方向一阶导数"""
        return fdm.pd1_2d(f, axis=1, h=self.dx, method='centered', bc='non-periodic')

    def d_dz(self, f):
        """计算z方向一阶导数"""
        return fdm.pd1_2d(f, axis=0, h=self.dz, method='centered', bc='non-periodic')

    def d2_dx2(self, f):
        """计算x方向二阶导数"""
        return fdm.pd2_2d(f, axis=1, h=self.dx, method='centered', bc='non-periodic')

    def d2_dz2(self, f):
        """计算z方向二阶导数"""
        return fdm.pd2_2d(f, axis=0, h=self.dz, method='centered', bc='non-periodic')


# boundary condition handling functions
# def apply_velocity_bc(u, w, grid, bc_type='no_slip'):
#     """
#     apply velocity boundary conditions
#
#     parameters:
#         u, w: velocity field
#         grid: Grid2D object
#         bc_type: boundary condition type ('no_slip', 'free_slip', 'periodic')
#     """
#     if bc_type == 'no_slip':
#         if grid.staggered:
#             # no slip boundary condition for staggered grids
#             # bottom and top boundary (z=0, z=Lz)
#             u[0, :] = 0  # bottom boundary u=0
#             u[-1, :] = 0  # top boundary u=0
#             w[0, :] = 0  # bottom boundary w=0
#             w[-1, :] = 0  # top boundary w=0
#
#             # left and right boundaty can be periodic or no slip
#             # here assume left and right are periodic, so no setting
#         else:
#             # boundary condition for non-staggered grids
#             u[0, :] = 0  # bottom boundary
#             u[-1, :] = 0  # top boundary
#             u[:, 0] = 0  # left boundary
#             u[:, -1] = 0  # right boundary
#
#             w[0, :] = 0  # bottom boundary
#             w[-1, :] = 0  # top boundary
#             w[:, 0] = 0  # left boundary
#             w[:, -1] = 0  # right boundary


def apply_temperature_bc(T, T_hot=1.0, T_cold=0.0):
    """
    apply temperature boundary conditions

    parameters:
        T: temperature field
        T_hot: bottom boundary temperature
        T_cold: top boundary temperature
    """
    # bottom boundary：hot
    T[0, :] = T_hot
    # top boundary：cold
    T[-1, :] = T_cold
    # left and right boundary: adiabatic (∂T/∂x = 0)
    # achieve by setting boundary points equal to adjacent internal points
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]


def apply_velocity_bc(u, w, bc_params):
    """
    apply velocity boundary conditions

    Parameters:
        u, w : ndarray
            velocity components
        bc_params : dict
            boundary condition parameters
    """
    if bc_params['type'] == 'cavity':
        # cavity boundary condition
        # bottom boundary: no slip
        u[0, :] = bc_params.get('u_bottom', 0.0)
        w[0, :] = 0.0

        # top boundary: move walls
        u[-1, :] = bc_params.get('u_top', 1.0)
        w[-1, :] = 0.0

        # left and right boundary: no slip
        u[:, 0] = bc_params.get('u_left', 0.0)
        u[:, -1] = bc_params.get('u_right', 0.0)
        w[:, 0] = 0.0
        w[:, -1] = 0.0

    elif bc_params['type'] == 'channel':
        # channel boundary condition
        # bottom and top boundary: no slip
        u[0, :] = 0.0
        u[-1, :] = 0.0
        w[0, :] = 0.0
        w[-1, :] = 0.0

        # inlet: given velocity distribution
        if bc_params.get('u_inlet') == 'parabolic':
            nz = u.shape[0]
            for j in range(nz):
                z_norm = j / (nz - 1)
                u[j, 0] = 4 * z_norm * (1 - z_norm)  # parabolic profile
        else:
            u[:, 0] = bc_params.get('u_inlet', 0.0)

        w[:, 0] = 0.0

        # outslet: zero gradient (simplified processing)
        u[:, -1] = u[:, -2]
        w[:, -1] = w[:, -2]

    elif bc_params['type'] == 'periodic':
        # periodic boundary condition
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]

        w[0, :] = w[-2, :]
        w[-1, :] = w[1, :]
        w[:, 0] = w[:, -2]
        w[:, -1] = w[:, 1]


def apply_pressure_bc(p, bc_params=None):
    """
    apply pressure boundary conditions
    normally, homogeneous Neumann boundary condition: ∂p/∂n = 0

    Parameters:
        p : ndarray
            pressure field
        bc_params : dict, optional
            boundary condition parameters
    """
    # default: all boundaries are zero gradient
    p[0, :] = p[1, :]  # bottom boundary
    p[-1, :] = p[-2, :]  # top boundary
    p[:, 0] = p[:, 1]  # left boundary
    p[:, -1] = p[:, -2]  # right boundary

    # if there is special pressure boundary condition
    if bc_params is not None:
        if bc_params['type'] == 'channel' and 'pressure_outlet' in bc_params:
            # outslet specified pressure
            p[:, -1] = bc_params['pressure_outlet']


def create_test_operators():
    """
    测试边界条件函数
    """
    print("Testing boundary condition functions...")

    from grid import Grid2D
    import numpy as np

    # 创建测试网格
    grid = Grid2D(nx=5, nz=5, Lx=1.0, Lz=1.0, staggered=False)

    # 测试速度边界条件
    u = np.ones((6, 6))
    w = np.ones((6, 6))

    # 方腔流测试
    bc_params = {
        'type': 'cavity',
        'u_top': 2.0,
        'u_bottom': 0.0,
        'u_left': 0.0,
        'u_right': 0.0
    }

    print("Before applying velocity BC:")
    print(f"u top: {u[-1, :]}")
    print(f"u bottom: {u[0, :]}")

    apply_velocity_bc(u, w, bc_params)

    print("After applying velocity BC:")
    print(f"u top: {u[-1, :]}")
    print(f"u bottom: {u[0, :]}")
    print(f"w walls: {w[:, 0]}")

    # 测试压力边界条件
    p = np.random.random((6, 6))
    print(f"\nBefore pressure BC:")
    print(f"p[0, :] = {p[0, :]}")
    print(f"p[1, :] = {p[1, :]}")

    apply_pressure_bc(p)

    print(f"After pressure BC:")
    print(f"p[0, :] = {p[0, :]}")
    print(f"p[1, :] = {p[1, :]}")
    print(f"Equal? {np.allclose(p[0, :], p[1, :])}")

    print("Boundary condition tests completed!")


if __name__ == "__main__":
    create_test_operators()

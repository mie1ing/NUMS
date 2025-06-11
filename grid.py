import numpy as np

class Grid2D:
    def __init__(self, nx, nz, Lx, Lz, staggered=True):
        """
        Creation of a 2D grid
        nx, nz: numbers of grid points in x and z directions
        Lx, Lz: length of the domain in x and z directions
        """
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz

        self.dx = Lx / nx
        self.dz = Lz / nz

        self.staggered = staggered

        self._create_coordinates()

    def _create_coordinates(self):
        """creation of the coordinates of the grid"""
        # basic grid point（for pressure and temperature）
        self.x_p = np.linspace(0, self.Lx, self.nx + 1)
        self.z_p = np.linspace(0, self.Lz, self.nz + 1)

        if self.staggered:
            # staggered grid point
            # u defined on x boundary center
            self.x_u = np.linspace(-self.dx / 2, self.Lx + self.dx / 2, self.nx + 2)
            self.z_u = self.z_p.copy()

            # w defined on z boundary center
            self.x_w = self.x_p.copy()
            self.z_w = np.linspace(-self.dz / 2, self.Lz + self.dz / 2, self.nz + 2)
        else:
            # non-staggered grid, all variables on the same point
            self.x_u = self.x_p.copy()
            self.z_u = self.z_p.copy()
            self.x_w = self.x_p.copy()
            self.z_w = self.z_p.copy()

    def get_pressure_grid(self):
        """return the coordinates of the pressure/temperature grid"""
        return self.x_p, self.z_p

    def get_u_grid(self):
        """return the coordinates of the u grid"""
        return self.x_u, self.z_u

    def get_w_grid(self):
        """return the coordinates of the w grid"""
        return self.x_w, self.z_w

    def create_field(self, field_type='pressure'):
        """
        creation of a zero array corresponding to the field type
        field_type: 'pressure', 'u_velocity', 'w_velocity'
        """
        if field_type == 'pressure':
            return np.zeros((self.nz + 1, self.nx + 1))
        elif field_type == 'u_velocity':
            if self.staggered:
                return np.zeros((self.nz + 1, self.nx + 2))
            else:
                return np.zeros((self.nz + 1, self.nx + 1))
        elif field_type == 'w_velocity':
            if self.staggered:
                return np.zeros((self.nz + 2, self.nx + 1))
            else:
                return np.zeros((self.nz + 1, self.nx + 1))
        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def info(self):
        """show information about the grid"""
        print(f"Grid2D Information:")
        print(f"  Size: {self.nx}×{self.nz}")
        print(f"  Domain: {self.Lx}×{self.Lz}")
        print(f"  Spacing: dx={self.dx:.4f}, dz={self.dz:.4f}")
        print(f"  Staggered: {self.staggered}")

        if self.staggered:
            print(f"  Pressure grid: ({len(self.z_p)}, {len(self.x_p)})")
            print(f"  U-velocity grid: ({len(self.z_u)}, {len(self.x_u)})")
            print(f"  W-velocity grid: ({len(self.z_w)}, {len(self.x_w)})")
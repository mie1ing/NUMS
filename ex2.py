import numpy as np
import nums.fdm as fdm

x = np.linspace(0, 1, 10, endpoint=True)
y = np.linspace(0, 1, 10, endpoint=True)

X, Y = np.meshgrid(x, y)
# stream function
psi = X * (1 - X) * Y * (1 - Y)

u = fdm.pd2d(psi, axis=0, h=x[1] - x[0], method='forward', bc='periodic')
v = fdm.pd2d(psi, axis=1, h=y[1] - y[0], method='forward', bc='periodic')

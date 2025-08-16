import matplotlib.pyplot as plt
import numpy as np
import nums.ode as ode

# Example: harmonic oscillator d²x/dt² = -x -> y = [dx/dt, x]
# f = lambda y, t: np.array([-y[1], y[0]])
# y0 = [0, 1]
# t = np.linspace(0, 10, 1000)

# y = ode.rk4(f, y0, t)

# plt.plot(t, y[1, :])
# plt.xlabel('Time')
# plt.ylabel('Displacement')
# plt.title('Harmonic Oscillator')
# plt.grid(True)
# plt.show()

# Example: lorenz attractor
a = 10
b = 8/3
r = 28
f = lambda y, t: np.array([
    a * (y[1] - y[0]),
    r * y[0] - y[1] - y[0] * y[2],
    y[0] * y[1] - b * y[2]
])
y0 = [4, 5, 6]
dt = 0.05
n = 1000

y, t = ode.rk4(f, y0, dt, n)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(t, y[0, :])
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')
ax1.set_title('Lorenz Attractor')
ax1.grid(True)

ax2 = fig.add_subplot(2,1,2)
ax2.plot(y[1, :], y[2, :])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.show()



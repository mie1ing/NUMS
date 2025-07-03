import numpy as np
from tqdm import tqdm

def rk4(f, y0, dt, n):
    """
    Solve ODE problem, i.e. dy/dt = f(y,t), using Runge−Kutta algorithm.

    Parameters:
    - f: function that receives current state y and time t, returns dy/dt
    - y0: initial state of the system, numpy array of shape (M,)
    - dt: time steps length
    - n: number of time steps

    Returns:
    - y: (M x N) numpy array, each column is the state at corresponding time

    Example: harmonic oscillator d²x/dt² = -x -> y = [dx/dt, x]
    f = lambda y, t: np.array([-y[1], y[0]])
    y0 = [0, 1]
    dt = 0.05
    n = 1000

    y = ode.rk4(f, y0, t)
    """
    y0 = np.array(y0)
    M = y0.shape[0]
    t = np.linspace(0, n*dt, n + 1)
    Y = np.zeros((M, n + 1))
    Y[:, 0] = y0

    for n in tqdm(range(n), desc='Running RK4: '):
        y1 = f(Y[:, n], t[n])
        y2 = f(Y[:, n] + dt * y1 / 2, t[n] + dt / 2)
        y3 = f(Y[:, n] + dt * y2 / 2, t[n] + dt / 2)
        y4 = f(Y[:, n] + dt * y3, t[n] + dt)
        Y[:, n + 1] = Y[:, n] + dt * (y1 + 2*y2 + 2*y3 + y4) / 6

    return Y, t


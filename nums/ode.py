import numpy as np

def rk4(f, y0, t):
    """
    Solve ODE problem, i.e. dy/dt = f(y,t), using Runge−Kutta algorithm.

    Parameters:
    - f: function that receives current state y and time t, returns dy/dt
    - y0: initial state of the system, numpy array of shape (M,)
    - t: array of time steps (length N)

    Returns:
    - y: (M x N) numpy array, each column is the state at corresponding time

    Example: harmonic oscillator d²x/dt² = -x -> y = [dx/dt, x]
    f = lambda y, t: np.array([-y[1], y[0]])
    y0 = [0, 1]
    t = np.linspace(0, 10, 1000)

    y = ode.rk4(f, y0, t)
    """
    y0 = np.array(y0)
    M = y0.shape[0]
    N = len(t)
    Y = np.zeros((M, N))
    Y[:, 0] = y0
    delta = t[1] - t[0]

    for n in range(N - 1):
        y1 = f(Y[:, n], t[n])
        y2 = f(Y[:, n] + delta * y1 / 2, t[n] + delta / 2)
        y3 = f(Y[:, n] + delta * y2 / 2, t[n] + delta / 2)
        y4 = f(Y[:, n] + delta * y3, t[n] + delta)
        Y[:, n + 1] = Y[:, n] + delta * (y1 + 2*y2 + 2*y3 + y4) / 6

    return Y


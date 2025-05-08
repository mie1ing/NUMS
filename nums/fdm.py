import numpy as np

def fdm1_f2p_pbc(f, h):
    # For first order derivative.
    # Forward, two-point stencil, first order approximation, for periodic BCs.
    df = np.zeros_like(f)
    df[:-1] = (f[1:] - f[:-1]) / h
    df[-1] = (f[0] - f[-1]) / h  # periodic BCs
    return df


def fdm1_f2p_npbc(f, h):
    # For first order derivative
    # Forward, two-point stencil, first order approximation, for non-periodic BCs.
    df = np.zeros_like(f)
    df[:-1] = (f[1:] - f[:-1]) / h
    df[-1] = (f[-1] - f[-2]) / h # backward
    return df


def fdm1_b2p_pbc(f, h):
    # For first order derivative.
    # Backward, two-point stencil, first order approximation, for periodic BCs.
    df = np.zeros_like(f)
    df[1:] = (f[1:] - f[:-1]) / h
    df[0] = (f[0] - f[-1]) / h # periodic BCs
    return df


def fdm1_b2p_npbc(f, h):
    # For first order derivative.
    # Backward, two-point stencil, first order approximation, for non-periodic BCs.
    df = np.zeros_like(f)
    df[1:] = (f[1:] - f[:-1]) / h
    df[0] = (f[1] - f[0]) / h # forward
    return df


def fdm1_c3p_pbc(f, h):
    # For first order derivative.
    # Centered, three-point stencil, second order approximation, for periodic BCs.
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    df[0] = (f[1] - f[-1]) / (2 * h)  # periodic BCs
    df[-1] = (f[0] - f[-2]) / (2 * h)  # periodic BCs
    return df


def fdm1_c3p_npbc(f, h):
    # For first order derivative.
    # Centered, three-point stencil, for non-periodic BCs.
    # second order approximation in interior points,
    # first order approximation at boundaries
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    df[0] = (f[1] - f[0]) / h  # first order forward
    df[-1] = (f[-1] - f[-2]) / h  # first order backward
    return df


def pd2d(f, axis, h, method='forward', bc='periodic'):
    """
    Calculate partial derivative of 2D array f along axis direction.
    parameters:
        f: 2D array
        axis: 0 -> ∂/∂x，1 -> ∂/∂y
        h: grid spacing
        method: 'forward', 'backward', 'centered'
        bc: 'periodic', 'non-periodic'
    return:
        df: 2D array, same shape as f
    """
    if method == 'forward' and bc == 'periodic':
        df = np.apply_along_axis(fdm1_f2p_pbc, axis=axis, arr=f, h=h)
    elif method == 'forward' and bc == 'non-periodic':
        df = np.apply_along_axis(fdm1_f2p_npbc, axis=axis, arr=f, h=h)
    elif method == 'backward' and bc == 'periodic':
        df = np.apply_along_axis(fdm1_b2p_pbc, axis=axis, arr=f, h=h)
    elif method == 'backward' and bc == 'non-periodic':
        df = np.apply_along_axis(fdm1_b2p_npbc, axis=axis, arr=f, h=h)
    elif method == 'centered' and bc == 'periodic':
        df = np.apply_along_axis(fdm1_c3p_pbc, axis=axis, arr=f, h=h)
    elif method == 'centered' and bc == 'non-periodic':
        df = np.apply_along_axis(fdm1_c3p_npbc, axis=axis, arr=f, h=h)
    else:
        raise ValueError(f"invalid parameters: method='{method}', bc='{bc}'")
        
    return df


import numpy as np
from scipy.linalg import expm
from abcd_dim import abcd_dim

def lsym(A, B, C, D, u, t, x0=None, ntrp='foh'):
    """
    Transient response of a continuous-time linear system to arbitrary inputs.
    
    State-space system:
        dx/dt = Ax + Bu
        y     = Cx + Du
    
    Parameters:
        A    : dynamics matrix (n x n)
        B    : input matrix (n x r)
        C    : output matrix (m x n)
        D    : feedthrough matrix (m x r)
        u    : matrix of sampled inputs (r x p)
        t    : vector of uniformly spaced points in time (1 x p)
        x0   : vector of states at the first point in time (n x 1)
               default: zeros
        ntrp : interpolation method
               'zoh' = zero order hold
               'foh' = first order hold (default)
    
    Returns:
        y    : matrix of the system outputs (m x p)
    """
    
    # Get matrix dimensions and verify compatibility
    n, r, m = abcd_dim(A, B, C, D)
    
    # Convert to numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    u = np.asarray(u)
    t = np.asarray(t)
    
    if u.ndim == 1:
        u = u[np.newaxis, :]
    
    points = u.shape[1]  # number of data points
    
    dt = t[1] - t[0]  # uniform time-step value
    
    # Continuous-time to discrete-time conversion
    if ntrp.lower() == 'zoh':  # zero-order hold on inputs
        M = np.block([
            [A, B],
            [np.zeros((r, n+r))]
        ])
    else:  # first-order hold on inputs (foh)
        M = np.block([
            [A, B, np.zeros((n, r))],
            [np.zeros((r, n+r)), np.eye(r)],
            [np.zeros((r, n+2*r))]
        ])
    
    eMdt = expm(M * dt)  # matrix exponential
    Ad = eMdt[:n, :n]    # discrete-time dynamics matrix
    Bd = eMdt[:n, n:n+r] # discrete-time input matrix
    
    if ntrp.lower() == 'zoh':
        Bd0 = Bd
        Bd1 = np.zeros((n, r))
    else:  # foh
        Bd_ = eMdt[:n, n+r:n+2*r]  # discrete-time input matrix
        Bd0 = Bd - Bd_ / dt         # discrete-time input matrix for time p
        Bd1 = Bd_ / dt              # discrete-time input matrix for time p+1
    
    # Initial conditions
    if x0 is None:
        x0 = np.zeros(n)
    else:
        x0 = np.asarray(x0).flatten()
    
    # Memory allocation for the output
    y = np.zeros((m, points))
    
    # State at t[0] ... Kjell Ahlin
    x = x0 + Bd1 @ u[:, 0]
    
    # Output for the initial condition
    y[:, 0] = C @ x + D @ u[:, 0]
    
    # Time-stepping loop
    for p in range(1, points):
        x = Ad @ x + Bd0 @ u[:, p-1] + Bd1 @ u[:, p]
        y[:, p] = C @ x + D @ u[:, p]
    
    # Strip out round-off imaginary parts
    if np.max(np.abs(np.imag(y))) < 1e-12:
        y = np.real(y)
    
    return y


# ----------------------------------------------------------------- LSYM

import numpy as np
from multivarious.lti import abcd_dim

def dlsym(A, B, C, D, u, t=None, x0=None):
    """
    Simulate the response of a discrete-time linear system to arbitrary inputs.
    
    Discrete-time system:
        x[k+1] = A*x[k] + B*u[k]
        y[k]   = C*x[k] + D*u[k]
    
    Parameters:
        A  : dynamics matrix (n x n)
        B  : input matrix (n x m)
        C  : output matrix (l x n)
        D  : feedthrough matrix (l x m)
        u  : matrix of sampled inputs (m x p)
        t  : vector of uniformly spaced points in time (1 x p), not used
        x0 : vector of initial states (n x 1), defaults to zero
    
    Returns:
        y : matrix of system outputs (l x p)
    
    Author: HP Gavin, 2021-07-19, 2023-10-01, 2023-12-22
    """
    
    # Convert to numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    u = np.asarray(u)
    
    # Check dimensions
    n, m, l = abcd_dim(A, B, C, D)
    
    # Ensure u is 2D
    if u.ndim == 1:
        u = u[np.newaxis, :]
    
    points = u.shape[1]  # number of data points
    
    # Initial conditions
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0).flatten()
    
    # Memory allocation for the output
    y = np.full((l, points), np.nan)
    
    # Initial output
    y[:, 0] = C @ x + D @ u[:, 0]
    
    # Time-stepping loop
    for p in range(1, points):
        
        y[:, p] = C @ x + D @ u[:, p]
        
        x = A @ x + B @ u[:, p]
        
        # Safety check for divergence
        if np.any(np.abs(x) > 1e2):
            break
    
    return y


# --------------------------------------------------------- dlsym.py
# 2021-07-19 ...
#   replaced ...  x = A * x  +  B * u[:, p]
#   ... with ...  x = A * x  +  B * u[:, p-1]
# 2023-10-01
#  ... switch order of dynamics and output eqn calc's
# 2023-12-22
#  ... comment-out time

# Example usage and test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing dlsym: Discrete-time linear system simulation\n")
    
    # Define a simple discrete-time system (from continuous-time damped oscillator)
    # Using sampling time dt = 0.1
    wn = 10  # natural frequency (rad/s)
    zeta = 0.1  # damping ratio
    dt = 0.01
    
    # Continuous system
    Ac = np.array([[0, 1], [-wn**2, -2*zeta*wn]])
    Bc = np.array([[0], [1]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])
    
    # Discretize (simple method for demonstration)
    from scipy.linalg import expm
    M = expm(np.block([
        [Ac, Bc],
        [np.zeros((1, 2)), np.zeros((1, 1))]
    ]) * dt)
    
    A = M[:2, :2]
    B = M[:2, 2:]
    C = Cc
    D = Dc
    
    print(f"Discrete-time system (dt = {dt}):")
    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")
    
    # Generate input signal
    points = 500
    t = np.arange(points) * dt
    
    # Step input at t=0.5
    u = np.zeros((1, points))
    u[:, int(0.5/dt):] = 1.0
    
    # Initial conditions
    x0 = np.array([0, 0])
    
    # Simulate
    y = dlsym(A, B, C, D, u, t, x0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t, u.flatten(), 'r-', linewidth=2)
    ax1.set_ylabel('Input u(t)')
    ax1.set_title('Discrete-Time Linear System Response')
    ax1.grid(True)
    ax1.set_xlim([0, t[-1]])
    
    ax2.plot(t, y.flatten(), 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Output y(t)')
    ax2.grid(True)
    ax2.set_xlim([0, t[-1]])
    
    plt.tight_layout()
    plt.show()
    
    # Test with impulse response
    print("\n" + "="*60)
    print("Impulse Response Test")
    print("="*60 + "\n")
    
    u_impulse = np.zeros((1, points))
    u_impulse[:, 0] = 1.0 / dt  # Discrete impulse
    
    y_impulse = dlsym(A, B, C, D, u_impulse, t, x0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, y_impulse.flatten(), 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output y(t)')
    ax.set_title('Impulse Response')
    ax.grid(True)
    ax.set_xlim([0, t[-1]])
    plt.show()
    
    print(f"Peak response: {np.max(np.abs(y_impulse)):.4f}")
    print(f"Final value: {y_impulse[0, -1]:.4e}")

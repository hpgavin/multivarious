import numpy as np

def WienerFilter(u, y, Hn):
    """
    Identify or simulate a MIMO system modeled by discrete-time convolution:
    
        y(j) = sum_{k=1}^N H(k) * u(j-k+1)
    
    Parameters:
        u  : sequence of input data (r x N)
        y  : sequence of outputs (m x N)
        Hn : If Hn is a scalar, it is interpreted as the order of Markov 
             parameters, n, and the fitted Markov parameters H of dimension 
             (m x r*n) are returned.
             
             Otherwise, Hn is interpreted as a set of Markov parameters H 
             of dimension (m x r*n), and the model output y of dimension 
             (m x N) is returned.
             
             Note: The Wiener filter coefficients are returned in reverse 
             order [H(n), H(n-1), ..., H(1)]
    
    Returns:
        Hy : If Hn is a scalar, Hy contains the Markov parameters (m x r*n)
             Otherwise, Hy contains the model output (m x N)
    
    Reference:
        Wiener, Norbert, The Extrapolation, Interpolation, and Smoothing of 
        Stationary Time Series, John Wiley, New York, 1949.
    
    Author: HP Gavin, System Identification, Duke University, Fall 2013
            Updated: 2013-09-04, 2017-10-06
    """
    
    u = np.asarray(u)
    y = np.asarray(y)
    
    r, Nu = u.shape  # number of inputs, duration of inputs
    m, Ny = y.shape  # number of outputs, duration of outputs
    
    N = min(Nu, Ny)
    
    # Determine if we're fitting or simulating
    if np.isscalar(Hn) or (isinstance(Hn, np.ndarray) and Hn.size == 1):
        # Fit the model to input/output data sequences
        n = int(Hn)
        if r * n > N - n + 1:
            raise ValueError('WienerFilter: not enough data to fit')
    else:
        # Simulate the model response with input data sequences
        H = np.asarray(Hn)
        n = H.shape[1] // r
    
    # Assemble the Hankel matrix of the time sequence u
    U = np.zeros((r * n, N - n + 1))
    for k in range(n):
        U[r*k:r*(k+1), :] = u[:, k:N-n+k+1]
    
    if np.isscalar(Hn) or (isinstance(Hn, np.ndarray) and Hn.size == 1):
        # Fit the model to data
        
        # Auto-correlation of input data
        Ruu = (1 / (N - n)) * U @ U.T
        
        # Cross-correlation of input/output data
        Ryu = (1 / (N - n)) * y[:, n-1:N] @ U.T
        
        # Wiener-Hopf equations
        Hy = np.linalg.solve(Ruu.T, Ryu.T).T
        
    else:
        # Evaluate the model output
        Hy = np.hstack([np.zeros((m, n-1)), H @ U])
    
    return Hy


# --------------------------------------- WienerFilter   HP Gavin
# System Identification, Duke University, Fall 2013,
# updated: 2013-09-04, 2017-10-06

# Example usage and test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing WienerFilter: System Identification via Markov Parameters\n")
    print("="*60)
    
    # Example 1: Identify a simple FIR system
    print("\nExample 1: System Identification")
    print("-"*60)
    
    # True Markov parameters for a simple system
    # y[k] = 0.5*u[k] + 0.3*u[k-1] + 0.1*u[k-2]
    H_true = np.array([[0.1, 0.3, 0.5]])  # Note: reversed order
    
    # Generate input signal
    N = 1000
    u = np.random.randn(1, N)
    
    # Generate true output using convolution
    y_true = WienerFilter(u, np.zeros((1, N)), H_true)
    
    # Add measurement noise
    noise_level = 0.1
    y_noisy = y_true + noise_level * np.random.randn(*y_true.shape)
    
    print(f"True Markov parameters:\n{H_true}\n")
    print(f"Input shape: {u.shape}")
    print(f"Output shape: {y_noisy.shape}")
    
    # Identify the system
    n_order = 3
    H_identified = WienerFilter(u, y_noisy, n_order)
    
    print(f"\nIdentified Markov parameters:\n{H_identified}\n")
    print(f"Identification error:\n{H_true - H_identified}\n")
    print(f"Max absolute error: {np.max(np.abs(H_true - H_identified)):.4f}\n")
    
    # Example 2: Simulate using identified model
    print("="*60)
    print("\nExample 2: Model Simulation")
    print("-"*60)
    
    # New input for validation
    u_test = np.random.randn(1, 500)
    
    # Generate true output
    y_test_true = WienerFilter(u_test, np.zeros((1, 500)), H_true)
    
    # Generate identified output
    y_test_identified = WienerFilter(u_test, np.zeros((1, 500)), H_identified)
    
    # Compare
    error = y_test_true - y_test_identified
    print(f"Validation RMS error: {np.sqrt(np.mean(error**2)):.4f}")
    print(f"Validation max error: {np.max(np.abs(error)):.4f}\n")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Input signal
    t_plot = np.arange(200)
    axes[0].plot(t_plot, u_test[0, :200])
    axes[0].set_ylabel('Input u(t)')
    axes[0].set_title('System Identification via Wiener Filter')
    axes[0].grid(True)
    
    # Plot 2: Output comparison
    axes[1].plot(t_plot, y_test_true[0, :200], 'b-', label='True', linewidth=2)
    axes[1].plot(t_plot, y_test_identified[0, :200], 'r--', label='Identified', linewidth=1.5)
    axes[1].set_ylabel('Output y(t)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Error
    axes[2].plot(t_plot, error[0, :200], 'g-')
    axes[2].set_ylabel('Error')
    axes[2].set_xlabel('Time step')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Example 3: MIMO system (2 inputs, 2 outputs)
    print("="*60)
    print("\nExample 3: MIMO System (2 inputs, 2 outputs)")
    print("-"*60)
    
    # True MIMO Markov parameters
    # 2 outputs, 2 inputs, order 3
    H_mimo_true = np.array([
        [0.2, 0.1, 0.3, 0.2, 0.1, 0.05],  # output 1
        [0.15, 0.25, 0.1, 0.3, 0.2, 0.1]   # output 2
    ])  # shape: (2, 6) = (m, r*n) where m=2, r=2, n=3
    
    # Generate MIMO input
    N_mimo = 2000
    u_mimo = np.random.randn(2, N_mimo)
    
    # Generate true MIMO output
    y_mimo_true = WienerFilter(u_mimo, np.zeros((2, N_mimo)), H_mimo_true)
    
    # Add noise
    y_mimo_noisy = y_mimo_true + 0.05 * np.random.randn(*y_mimo_true.shape)
    
    # Identify MIMO system
    n_mimo = 3
    H_mimo_identified = WienerFilter(u_mimo, y_mimo_noisy, n_mimo)
    
    print(f"True MIMO Markov parameters:\n{H_mimo_true}\n")
    print(f"Identified MIMO Markov parameters:\n{H_mimo_identified}\n")
    print(f"MIMO identification error (max): {np.max(np.abs(H_mimo_true - H_mimo_identified)):.4f}\n")
    
    # Validate MIMO model
    u_mimo_test = np.random.randn(2, 500)
    y_mimo_test_true = WienerFilter(u_mimo_test, np.zeros((2, 500)), H_mimo_true)
    y_mimo_test_id = WienerFilter(u_mimo_test, np.zeros((2, 500)), H_mimo_identified)
    
    error_mimo = y_mimo_test_true - y_mimo_test_id
    print(f"MIMO validation RMS error: {np.sqrt(np.mean(error_mimo**2)):.4f}")
    
    plt.show()
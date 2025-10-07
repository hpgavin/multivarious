import numpy as np

def damp(a, delta_t=None):
    """
    Natural frequency and damping factor for continuous or discrete systems.
    
    damp(A) displays a table of the natural frequencies and 
    damping ratios for the continuous-time dynamics matrix.
    
    damp(A, delta_t) displays a table of the natural frequencies 
    damping ratios for the discrete-time dynamics matrix
    with a sample time step of delta_t.
    
    Parameters:
        a       : Can be in one of several formats:
                  (1) If a is square, it is assumed to be the state-space 
                      dynamics matrix.
                  (2) If a is a row vector (1D array), it is assumed to be 
                      a vector of polynomial coefficients from a transfer function.
                  (3) If a is a column vector, it is assumed to contain 
                      root locations.
        delta_t : Optional sample time step for discrete-time systems
        
    Returns:
        wn : vector of natural frequencies (cyc/sec) if return values requested
        z  : vector of damping ratios if return values requested
        
    If no return values are requested, displays a table of results.
    """
    
    a = np.asarray(a)
    
    # Handle empty input
    if a.size == 0:
        return np.array([0.0]), np.array([0.0])
    
    # Determine the format of input and compute eigenvalues/roots
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        # Square matrix - compute eigenvalues
        r = np.linalg.eigvals(a)
        n = len(r)
    elif a.ndim == 1 or (a.ndim == 2 and (a.shape[0] == 1 or a.shape[1] == 1)):
        # Vector input
        a = a.flatten()
        n = len(a)
        if n > 1:
            # Row/column vector with multiple elements - assume polynomial coefficients
            r = np.roots(a)
        else:
            # Single element - treat as root location
            r = a
    else:
        raise ValueError('The variable A must be a vector or a square matrix.')
    
    # Discrete time system conversion
    if delta_t is not None:
        r = np.log(r) / delta_t
    
    # Compute natural frequencies and damping ratios
    wn = np.abs(r)
    z = -(np.real(r) - 2*np.finfo(float).eps) / (wn + 2*np.finfo(float).eps)
    
    # Sort by increasing natural frequency
    idx = np.argsort(np.abs(wn))
    wn = wn[idx]
    z = z[idx]
    r = r[idx]
    
    # Damped frequency
    wd = wn * np.sqrt(np.abs(z**2 - 1))
    
    # Display results if no output arguments requested
    # In Python, we'll display if called without assignment or if explicitly desired
    # For now, we'll create a display version that can be called
    print('\n')
    print('     Natural               Damped  ')
    print('    Frequency             Frequency      Eigenvalue ')
    print('    (cyc/sec)    Damping  (cyc/sec)   real        imag ')
    print('    -----------------------------------------------------')
    
    for i in range(len(wn)):
        print(f'    {wn[i]/(2*np.pi):10.5f}  {z[i]:10.5f}    '
              f'{wd[i]/(2*np.pi):10.5f}   {np.real(r[i]):10.5f}  '
              f'{np.imag(r[i]):10.5f}')
    
    return wn, z

# ----------------------------------------------------------------------- DAMP

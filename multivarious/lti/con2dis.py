import numpy as np
from scipy.linalg import expm
from multivarious.lti import abcd_dim 
# from abcd_dim import abcd_dim

def con2dis(Ac, Bc, Cc, Dc, dt, method='foh'):
    """
    Continuous-time to discrete-time transformation for state-space systems.
    
    Converts a continuous-time system:
        dx/dt = Ac*x + Bc*u
        y     = Cc*x + Dc*u
    
    To a discrete-time system:
        x[k+1] = Ad*x[k] + Bd*u[k]
        y[k]   = Cd*x[k] + Dd*u[k]
    
    Parameters:
        Ac     : continuous-time dynamics matrix (n x n)
        Bc     : continuous-time input matrix (n x r)
        Cc     : continuous-time output matrix (m x n)
        Dc     : continuous-time feedthrough matrix (m x r)
        dt     : sample period
        method : interpolation method
               'foh' - first order hold (default)
               'zoh' - zero order hold
    
    Returns:
        Ad : discrete-time dynamics matrix (n x n)
        Bd : discrete-time input matrix (n x r)
        Cd : discrete-time output matrix (m x n)
        Dd : discrete-time feedthrough matrix (m x r)
    
    Author: HP Gavin, 2020-11-01, 2021-07-19, 2024-01-23
    """
    
    # Convert to numpy arrays
    Ac = np.asarray(Ac)
    Bc = np.asarray(Bc)
    Cc = np.asarray(Cc)
    Dc = np.asarray(Dc)
    
    # Check dimensions
    n, r, m = abcd_dim(Ac, Bc, Cc, Dc)
    
    # Continuous-time to discrete-time conversion
    if method.lower() == 'zoh':  # Zero-order hold on inputs
        M = np.block([
            [Ac, Bc],
            [np.zeros((r, n+r))]
        ])
    else:  # First-order hold on inputs
        M = np.block([
            [Ac, Bc, np.zeros((n, r))],
            [np.zeros((r, n+r)), np.eye(r)],
            [np.zeros((r, n+2*r))]
        ])
    
    eMdt = expm(M * dt)  # Matrix exponential
    Ad = eMdt[:n, :n]    # Discrete-time dynamics matrix
    Bd = eMdt[:n, n:n+r] # Discrete-time input matrix
    
    if method.lower() == 'zoh':
        Cd = Cc.copy()
        Dd = Dc.copy()
    else:  # FOH
        Bdp = eMdt[:n, n+r:n+2*r]  # Additional discrete-time input matrix
        Bd = Bd + (Ad - np.eye(n)) @ Bdp / dt
        Cd = Cc.copy()
        Dd = Dc + Cc @ Bdp / dt
    
    return Ad, Bd, Cd, Dd


# ======================================================================= con2dis
# HP Gavin, 2020-11-01, 2021-07-19, 2024-01-23

# Example usage and test
if __name__ == "__main__":
    print("Testing con2dis with round-trip conversion\n")
    
    # Define a continuous-time system (simple 2nd order system)
    Ac = np.array([[0, 1], [-10, -2]])
    Bc = np.array([[0], [1]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])
    
    dt = 0.1
    
    print("Original continuous-time system:")
    print(f"Ac =\n{Ac}\n")
    print(f"Bc =\n{Bc}\n")
    print(f"Cc =\n{Cc}\n")
    print(f"Dc =\n{Dc}\n")
    
    # Test ZOH conversion
    print("="*60)
    print("Zero-Order Hold (ZOH) Conversion")
    print("="*60 + "\n")
    
    Ad_zoh, Bd_zoh, Cd_zoh, Dd_zoh = con2dis(Ac, Bc, Cc, Dc, dt, method='zoh')
    
    print(f"Discrete-time system (ZOH, dt={dt}):")
    print(f"Ad =\n{Ad_zoh}\n")
    print(f"Bd =\n{Bd_zoh}\n")
    print(f"Cd =\n{Cd_zoh}\n")
    print(f"Dd =\n{Dd_zoh}\n")
    
    # Test FOH conversion
    print("="*60)
    print("First-Order Hold (FOH) Conversion")
    print("="*60 + "\n")
    
    Ad_foh, Bd_foh, Cd_foh, Dd_foh = con2dis(Ac, Bc, Cc, Dc, dt, method='foh')
    
    print(f"Discrete-time system (FOH, dt={dt}):")
    print(f"Ad =\n{Ad_foh}\n")
    print(f"Bd =\n{Bd_foh}\n")
    print(f"Cd =\n{Cd_foh}\n")
    print(f"Dd =\n{Dd_foh}\n")
    
    # Compare eigenvalues
    eig_c = np.linalg.eigvals(Ac)
    eig_d_zoh = np.log(np.linalg.eigvals(Ad_zoh)) / dt
    eig_d_foh = np.log(np.linalg.eigvals(Ad_foh)) / dt
    
    print("="*60)
    print("Eigenvalue Comparison")
    print("="*60 + "\n")
    print(f"Continuous eigenvalues:\n{eig_c}\n")
    print(f"Recovered from ZOH:\n{eig_d_zoh}\n")
    print(f"Recovered from FOH:\n{eig_d_foh}\n")
    
    print(f"Error (ZOH): {np.max(np.abs(np.sort(eig_c) - np.sort(eig_d_zoh))):.2e}")
    print(f"Error (FOH): {np.max(np.abs(np.sort(eig_c) - np.sort(eig_d_foh))):.2e}")
    
    # Test round-trip conversion with dis2con
    try:
        from dis2con import dis2con
        
        print("\n" + "="*60)
        print("Round-Trip Test: Continuous → Discrete → Continuous")
        print("="*60 + "\n")
        
        # ZOH round-trip
        Ac_rec_zoh, Bc_rec_zoh, Cc_rec_zoh, Dc_rec_zoh = dis2con(
            Ad_zoh, Bd_zoh, Cd_zoh, Dd_zoh, dt, method='zoh'
        )
        
        print("ZOH Round-trip errors:")
        print(f"  max|Ac - Ac_recovered| = {np.max(np.abs(Ac - Ac_rec_zoh)):.2e}")
        print(f"  max|Bc - Bc_recovered| = {np.max(np.abs(Bc - Bc_rec_zoh)):.2e}")
        
        # FOH round-trip
        Ac_rec_foh, Bc_rec_foh, Cc_rec_foh, Dc_rec_foh = dis2con(
            Ad_foh, Bd_foh, Cd_foh, Dd_foh, dt, method='foh'
        )
        
        print("\nFOH Round-trip errors:")
        print(f"  max|Ac - Ac_recovered| = {np.max(np.abs(Ac - Ac_rec_foh)):.2e}")
        print(f"  max|Bc - Bc_recovered| = {np.max(np.abs(Bc - Bc_rec_foh)):.2e}")
        
    except ImportError:
        print("\n(dis2con not available for round-trip test)")

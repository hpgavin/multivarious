import numpy as np
from scipy.linalg import logm
from abcd_dim import abcd_dim

def dis2con(Ad, Bd, Cd, Dd, dt, ntrp='foh'):
    """
    Discrete-time to continuous-time transformation for state-space systems.
    
    Converts a discrete-time system:
        x[k+1] = Ad*x[k] + Bd*u[k]
        y[k]   = Cd*x[k] + Dd*u[k]
    
    To a continuous-time system:
        dx/dt = Ac*x + Bc*u
        y     = Cc*x + Dc*u
    
    Note: Ac must be invertible.
    
    Parameters:
        Ad   : discrete-time dynamics matrix (n x n)
        Bd   : discrete-time input matrix (n x r)
        Cd   : discrete-time output matrix (m x n)
        Dd   : discrete-time feedthrough matrix (m x r)
        dt   : sample period
        ntrp : interpolation method
               'foh' - first order hold (default)
               'zoh' - zero order hold
    
    Returns:
        Ac : continuous-time dynamics matrix (n x n)
        Bc : continuous-time input matrix (n x r)
        Cc : continuous-time output matrix (m x n)
        Dc : continuous-time feedthrough matrix (m x r)
    
    Author: HP Gavin, 2023-03-06, 2024-01-23, 2024-03-25
    """
    
    # Convert to numpy arrays
    Ad = np.asarray(Ad)
    Bd = np.asarray(Bd)
    Cd = np.asarray(Cd)
    Dd = np.asarray(Dd)
    
    # Check dimensions
    n, r, m = abcd_dim(Ad, Bd, Cd, Dd)
    
    # First solve for zero-order hold
    # M = logm([Ad, Bd; 0, I]) / dt
    M_block = np.block([
        [Ad, Bd],
        [np.zeros((r, n)), np.eye(r)]
    ])
    
    M = logm(M_block) / dt
    
    Ac = M[:n, :n]
    Bc = M[:n, n:n+r]
    Cc = Cd.copy()
    Dc = Dd.copy()
    
    # If first-order hold, adjust Bc and Dc
    if ntrp.lower() == 'foh':
        
        invAc = np.linalg.inv(Ac)
        In = np.eye(n)
        
        A1 = invAc @ (Ad - In)
        A2 = invAc @ invAc @ (Ad - In - Ac * dt)
        
        # Solve for Bc: (A1 + (Ad-I)*A2/dt) * Bc = Bd
        Bc = np.linalg.solve(A1 + (Ad - In) @ A2 / dt, Bd)
        
        Bdp = A2 @ Bc
        Dc = Dd - Cc @ Bdp / dt
    
    return Ac, Bc, Cc, Dc


# ================================================================= dis2con
# HP Gavin, 2023-03-06, 2024-01-23, 2024-03-25

# Example usage and test
if __name__ == "__main__":
    from scipy.linalg import expm
    
    print("Testing dis2con with round-trip conversion\n")
    
    # Define a continuous-time system
    Ac = np.array([[0, 1], [-10, -2]])
    Bc = np.array([[0], [1]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])
    
    dt = 0.1
    
    print("Original continuous-time system:")
    print(f"Ac =\n{Ac}\n")
    print(f"Bc =\n{Bc}\n")
    
    # Convert to discrete-time using expm (ZOH)
    M = expm(np.block([
        [Ac, Bc],
        [np.zeros((1, 2)), np.zeros((1, 1))]
    ]) * dt)
    
    Ad_zoh = M[:2, :2]
    Bd_zoh = M[:2, 2:]
    Cd_zoh = Cc
    Dd_zoh = Dc
    
    print(f"\nDiscrete-time system (ZOH, dt={dt}):")
    print(f"Ad =\n{Ad_zoh}\n")
    print(f"Bd =\n{Bd_zoh}\n")
    
    # Convert back to continuous-time
    Ac_rec_zoh, Bc_rec_zoh, Cc_rec_zoh, Dc_rec_zoh = dis2con(
        Ad_zoh, Bd_zoh, Cd_zoh, Dd_zoh, dt, ntrp='zoh'
    )
    
    print("\nRecovered continuous-time system (ZOH):")
    print(f"Ac =\n{Ac_rec_zoh}\n")
    print(f"Bc =\n{Bc_rec_zoh}\n")
    
    print("\nError in Ac (ZOH):")
    print(f"  max|Ac - Ac_recovered| = {np.max(np.abs(Ac - Ac_rec_zoh)):.2e}")
    print("\nError in Bc (ZOH):")
    print(f"  max|Bc - Bc_recovered| = {np.max(np.abs(Bc - Bc_rec_zoh)):.2e}")
    
    # Test FOH conversion
    print("\n" + "="*60)
    print("Testing FOH conversion")
    print("="*60 + "\n")
    
    # For FOH, we need discrete system created with FOH
    M_foh = expm(np.block([
        [Ac, Bc, np.zeros((2, 1))],
        [np.zeros((1, 2)), np.zeros((1, 1)), np.eye(1)],
        [np.zeros((1, 3))]
    ]) * dt)
    
    Ad_foh = M_foh[:2, :2]
    Bd_foh = M_foh[:2, 2:3]
    Bd_foh_prime = M_foh[:2, 3:4]
    
    # Reconstruct Bd for FOH
    Bd_discrete = Bd_foh - Bd_foh_prime / dt
    
    print(f"Discrete-time system (FOH, dt={dt}):")
    print(f"Ad =\n{Ad_foh}\n")
    print(f"Bd =\n{Bd_discrete}\n")
    
    Ac_rec_foh, Bc_rec_foh, Cc_rec_foh, Dc_rec_foh = dis2con(
        Ad_foh, Bd_discrete, Cd_zoh, Dd_zoh, dt, ntrp='foh'
    )
    
    print("Recovered continuous-time system (FOH):")
    print(f"Ac =\n{Ac_rec_foh}\n")
    print(f"Bc =\n{Bc_rec_foh}\n")
    
    print("\nError in Ac (FOH):")
    print(f"  max|Ac - Ac_recovered| = {np.max(np.abs(Ac - Ac_rec_foh)):.2e}")
    print("\nError in Bc (FOH):")
    print(f"  max|Bc - Bc_recovered| = {np.max(np.abs(Bc - Bc_rec_foh)):.2e}")

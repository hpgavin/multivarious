import numpy as np

def ctrb(A, B, p=None):
    """
    Form the controllability matrix for a discrete-time system.
    
    Q = ctrb(A, B, p)
    
    Forms the controllability matrix:
        Q = [B  AB  A^2*B  A^3*B  ...  A^(p-1)*B]
    
    For the discrete-time system:
        x[k+1] = A*x[k] + B*u[k]
    
    The system is controllable if rank(Q) = n, where n is the number of states.
    
    Parameters:
        A : state matrix (n x n)
        B : input matrix (n x r)
        p : number of blocks in Q (default: n, or max(p, n) if specified)
    
    Returns:
        Q : controllability matrix (n x r*p)
    """
    
    A = np.asarray(A)
    B = np.asarray(B)
    
    n, r = B.shape  # number of states and inputs
    
    if p is None:
        p = n
    else:
        p = max(p, n)
    
    Q = np.zeros((n, r * p))  # dimension of Q
    Q[:, :r] = B  # first block is input matrix
    
    for k in range(1, p):
        AkB = Q[:, (k-1)*r:(k-1)*r+r]
        Q[:, k*r:k*r+r] = A @ AkB
    
    return Q


# --------------------------------------------------------------------- CTRB

# Example usage and test
if __name__ == "__main__":
    print("Testing controllability matrix\n")
    print("="*60)
    
    # Example 1: Controllable system
    print("\nExample 1: Controllable system")
    print("-"*60)
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    
    Q = ctrb(A, B)
    
    print(f"A =\n{A}\n")
    print(f"B =\n{B}\n")
    print(f"Controllability matrix Q =\n{Q}\n")
    print(f"Rank of Q: {np.linalg.matrix_rank(Q)}")
    print(f"Number of states: {A.shape[0]}")
    
    if np.linalg.matrix_rank(Q) == A.shape[0]:
        print("✓ System is CONTROLLABLE")
    else:
        print("✗ System is NOT controllable")
    
    # Example 2: Uncontrollable system
    print("\n" + "="*60)
    print("\nExample 2: Uncontrollable system")
    print("-"*60)
    A2 = np.array([[1, 0], [0, 2]])
    B2 = np.array([[1], [0]])
    
    Q2 = ctrb(A2, B2)
    
    print(f"A =\n{A2}\n")
    print(f"B =\n{B2}\n")
    print(f"Controllability matrix Q =\n{Q2}\n")
    print(f"Rank of Q: {np.linalg.matrix_rank(Q2)}")
    print(f"Number of states: {A2.shape[0]}")
    
    if np.linalg.matrix_rank(Q2) == A2.shape[0]:
        print("✓ System is CONTROLLABLE")
    else:
        print("✗ System is NOT controllable")
    
    # Example 3: MIMO system
    print("\n" + "="*60)
    print("\nExample 3: MIMO system (2 inputs)")
    print("-"*60)
    A3 = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]])
    B3 = np.array([[0, 0], [0, 1], [1, 0]])
    
    Q3 = ctrb(A3, B3)
    
    print(f"A =\n{A3}\n")
    print(f"B =\n{B3}\n")
    print(f"Controllability matrix Q shape: {Q3.shape}")
    print(f"Rank of Q: {np.linalg.matrix_rank(Q3)}")
    print(f"Number of states: {A3.shape[0]}")
    
    if np.linalg.matrix_rank(Q3) == A3.shape[0]:
        print("✓ System is CONTROLLABLE")
    else:
        print("✗ System is NOT controllable")

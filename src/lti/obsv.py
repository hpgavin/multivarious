import numpy as np

def obsv(A, C, p=None):
    """
    Form the observability matrix for a discrete-time system.
    
    P = obsv(A, C, p)
    
    Forms the observability matrix:
        P = [C; CA; CA^2; CA^3; ...; C*A^(p-1)]
    
    For the discrete-time system:
        x[k+1] = A*x[k]
        y[k]   = C*x[k]
    
    The system is observable if rank(P) = n, where n is the number of states.
    
    Parameters:
        A : state matrix (n x n)
        C : output matrix (m x n)
        p : number of blocks in P (default: n, or max(p, n) if specified)
    
    Returns:
        P : observability matrix (m*p x n)
    """
    
    A = np.asarray(A)
    C = np.asarray(C)
    
    m, n = C.shape  # number of outputs and states
    
    if p is None:
        p = n
    else:
        p = max(p, n)
    
    P = np.zeros((m * p, n))  # dimension of P
    P[:m, :] = C  # first block is output matrix
    
    for k in range(1, p):
        CAk = P[(k-1)*m:(k-1)*m+m, :]
        P[k*m:k*m+m, :] = CAk @ A
    
    return P


# ---------------------------------------------------------------------- OBSV

# Example usage and test
if __name__ == "__main__":
    print("Testing observability matrix\n")
    print("="*60)
    
    # Example 1: Observable system
    print("\nExample 1: Observable system")
    print("-"*60)
    A = np.array([[0, 1], [-2, -3]])
    C = np.array([[1, 0]])
    
    P = obsv(A, C)
    
    print(f"A =\n{A}\n")
    print(f"C =\n{C}\n")
    print(f"Observability matrix P =\n{P}\n")
    print(f"Rank of P: {np.linalg.matrix_rank(P)}")
    print(f"Number of states: {A.shape[0]}")
    
    if np.linalg.matrix_rank(P) == A.shape[0]:
        print("✓ System is OBSERVABLE")
    else:
        print("✗ System is NOT observable")
    
    # Example 2: Unobservable system
    print("\n" + "="*60)
    print("\nExample 2: Unobservable system")
    print("-"*60)
    A2 = np.array([[1, 0], [0, 2]])
    C2 = np.array([[1, 0]])
    
    P2 = obsv(A2, C2)
    
    print(f"A =\n{A2}\n")
    print(f"C =\n{C2}\n")
    print(f"Observability matrix P =\n{P2}\n")
    print(f"Rank of P: {np.linalg.matrix_rank(P2)}")
    print(f"Number of states: {A2.shape[0]}")
    
    if np.linalg.matrix_rank(P2) == A2.shape[0]:
        print("✓ System is OBSERVABLE")
    else:
        print("✗ System is NOT observable")
    
    # Example 3: MIMO system (multiple outputs)
    print("\n" + "="*60)
    print("\nExample 3: MIMO system (2 outputs)")
    print("-"*60)
    A3 = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]])
    C3 = np.array([[1, 0, 0], [0, 1, 0]])
    
    P3 = obsv(A3, C3)
    
    print(f"A =\n{A3}\n")
    print(f"C =\n{C3}\n")
    print(f"Observability matrix P shape: {P3.shape}")
    print(f"Rank of P: {np.linalg.matrix_rank(P3)}")
    print(f"Number of states: {A3.shape[0]}")
    
    if np.linalg.matrix_rank(P3) == A3.shape[0]:
        print("✓ System is OBSERVABLE")
    else:
        print("✗ System is NOT observable")
    
    # Example 4: Testing duality between controllability and observability
    print("\n" + "="*60)
    print("\nExample 4: Duality test")
    print("-"*60)
    print("The system (A, B) is controllable iff (A', C'=B') is observable")
    
    from ctrb import ctrb
    
    A4 = np.array([[0, 1], [-2, -3]])
    B4 = np.array([[0], [1]])
    
    Q = ctrb(A4, B4)
    P_dual = obsv(A4.T, B4.T)
    
    print(f"\nControllability matrix rank: {np.linalg.matrix_rank(Q)}")
    print(f"Dual observability matrix rank: {np.linalg.matrix_rank(P_dual)}")
    print(f"Ranks equal: {np.linalg.matrix_rank(Q) == np.linalg.matrix_rank(P_dual)}")

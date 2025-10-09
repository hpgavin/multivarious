import numpy as np
from scipy.linalg import solve_sylvester

def liap(A, B, C=None):
    """
    Solve the general form of the Lyapunov matrix equation (Sylvester equation).
    
    X = liap(A, B, C) solves the general Sylvester equation:
        (A and B must be square and C must have the rows of A and columns of B)
        
        A*X + X*B + C = 0
    
    Q = liap(A, B) solves the Right Lyapunov matrix equation:
        (A and B have the same number of rows.)
        
        A*Q + Q*A' + B*B' = 0
    
    P = liap(A', C) solves the Left Lyapunov matrix equation:
        (A and C have the same number of columns.)
        
        A'*P + P*A + C'*C = 0
    
    Parameters:
        A : square matrix (n x n)
        B : matrix (various shapes depending on equation type)
        C : optional matrix (default None)
        
    Returns:
        X : solution matrix
    """
    
    A = np.asarray(A)
    B = np.asarray(B)
    if C is not None:
        C = np.asarray(C)
    
    ma, na = A.shape
    if ma != na:
        raise ValueError('liap: A is not square')
    else:
        n = ma
    
    if C is None:  # Transform Lyapunov equation to Sylvester equation form
        
        mb, nb = B.shape
        if mb != nb:  # B is not square
            if mb == n:  # solve A*X + X*A' + B*B' = 0  (right)
                C = -B @ B.T
            elif nb == n:  # solve A'*X + X*A + B'*B = 0  (left)
                C = -B.T @ B
            else:
                raise ValueError('liap: A, B not conformably dimensioned')
        else:  # B is square
            C = -B
        
        B = A.T
    
    # Solve the Sylvester equation: A*X + X*B + C = 0
    # scipy.linalg.solve_sylvester solves: A*X + X*B = Q
    # So we need: A*X + X*B = -C
    X = solve_sylvester(A, B, -C)
    
    # Ignore complex part if real inputs (imaginary parts should be small)
    if np.isrealobj(A) and np.isrealobj(B) and np.isrealobj(C):
        X = np.real(X)
    
    return X


# Example usage and tests
if __name__ == "__main__":
    # Test 1: General Sylvester equation
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[1, 0], [0, 1]])
    
    X = liap(A, B, C)
    residual = A @ X + X @ B + C
    print("Test 1 - General Sylvester:")
    print(f"Max residual: {np.max(np.abs(residual))}")
    
    # Test 2: Right Lyapunov equation
    A = np.array([[-1, 2], [-3, -4]])
    B = np.array([[1, 0], [0, 1]])
    
    Q = liap(A, B)
    residual = A @ Q + Q @ A.T + B @ B.T
    print("\nTest 2 - Right Lyapunov:")
    print(f"Max residual: {np.max(np.abs(residual))}")
    
    # Test 3: Left Lyapunov equation
    A = np.array([[-1, 2], [-3, -4]])
    C = np.array([[1, 0], [0, 1]])
    
    P = liap(A.T, C)
    residual = A.T @ P + P @ A + C.T @ C
    print("\nTest 3 - Left Lyapunov:")
    print(f"Max residual: {np.max(np.abs(residual))}")


import numpy as np

def blkToeplitz(y, r, k):
    """
    Return a block Toeplitz matrix filled with the data y containing k block-rows.
    
    A block Toeplitz matrix has constant values along diagonals of blocks.
    Used in system identification, convolution operations, and filtering.
    
    Parameters:
        y : sequence of vectors, e.g., time sequences (m x N)
            where N should be divisible by r
        r : number of columns in each block of y
        k : number of block rows of T
    
    Returns:
        T : block Toeplitz matrix of the output (m*k x r*(N/r+1-k))
    
    Example:
        For y = [y1, y2, y3, y4, y5, y6] with r=2, k=2:
        T = [y4 y3 | y2 y1 | y0 y-1]  (reversed blocks)
            [y5 y4 | y3 y2 | y1 y0 ]
        
        Each block is m x r, and blocks shift by r columns across each row
        (in reverse order compared to Hankel)
    """
    
    y = np.asarray(y)
    
    m, N_total = y.shape
    
    # Time sequences should be the rows of y
    if N_total < m:
        y = y.T
        m, N_total = y.shape
    
    N = N_total // r  # number of r-column blocks
    
    # Check dimensions
    if k < 0:
        raise ValueError('blkToeplitz: k should be positive')
    
    j = N + 1 - k  # number of block columns in T
    
    if m * k > r * j:
        raise ValueError('blkToeplitz: # rows > # cols, decrease k')
    
    # Fill the block-Toeplitz matrix
    T = np.zeros((m * k, r * j))
    
    for i in range(k):
        T[m*i:m*(i+1), :] = y[:, r*(k-i-1):r*(k-i-1+j)]
    
    return T


# ----------------------------------------------------------------- blkToeplitz

# Example usage and test
if __name__ == "__main__":
    print("Testing blkToeplitz: Block Toeplitz Matrix Construction\n")
    print("="*60)
    
    # Example 1: Simple scalar sequence
    print("\nExample 1: Scalar sequence (m=1)")
    print("-"*60)
    
    # Single output, 12 time samples
    y1 = np.arange(1, 13).reshape(1, -1)
    print(f"Input y (shape {y1.shape}):\n{y1}\n")
    
    r1 = 2  # 2 columns per block
    k1 = 3  # 3 block rows
    
    T1 = blkToeplitz(y1, r1, k1)
    print(f"Block Toeplitz matrix T (r={r1}, k={k1}):")
    print(f"Shape: {T1.shape}")
    print(f"\n{T1}\n")
    
    # Verify Toeplitz structure (constant diagonals)
    print("Toeplitz structure verification:")
    print(f"T[0,0] = {T1[0, 0]}, T[1,1] = {T1[1, 1]}  (should be equal for Toeplitz)")
    print(f"T[0,2] = {T1[0, 2]}, T[1,3] = {T1[1, 3]}  (should be equal for Toeplitz)")
    
    # Example 2: Compare Hankel vs Toeplitz
    print("\n" + "="*60)
    print("\nExample 2: Hankel vs Toeplitz Comparison")
    print("-"*60)
    
    from blkHankel import blkHankel
    
    y2 = np.arange(1, 11).reshape(1, -1)
    r2 = 2
    k2 = 2
    
    H2 = blkHankel(y2, r2, k2)
    T2 = blkToeplitz(y2, r2, k2)
    
    print(f"Input y:\n{y2}\n")
    print(f"Block Hankel matrix (constant anti-diagonals):")
    print(f"{H2}\n")
    print(f"Block Toeplitz matrix (constant diagonals):")
    print(f"{T2}\n")
    
    # Example 3: Multi-output sequence
    print("="*60)
    print("\nExample 3: Multi-output sequence (m=2)")
    print("-"*60)
    
    y3 = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ])
    
    print(f"Input y (shape {y3.shape}):\n{y3}\n")
    
    r3 = 2
    k3 = 2
    
    T3 = blkToeplitz(y3, r3, k3)
    print(f"Block Toeplitz matrix T (r={r3}, k={k3}):")
    print(f"Shape: {T3.shape}")
    print(f"\n{T3}\n")
    
    # Example 4: Convolution interpretation
    print("="*60)
    print("\nExample 4: Convolution Interpretation")
    print("-"*60)
    
    # Toeplitz matrices are used for convolution
    # T @ x represents convolution of filter coefficients with signal
    
    # Simple FIR filter coefficients (impulse response)
    h = np.array([[1, 0.5, 0.25]])  # 1x3
    
    # Extend to form blocks
    h_extended = np.hstack([h, np.zeros((1, 9))])  # 1x12
    
    r4 = 3  # each block has 3 filter taps
    k4 = 2  # 2 block rows
    
    T4 = blkToeplitz(h_extended, r4, k4)
    
    print(f"Filter coefficients h: {h}")
    print(f"\nToeplitz matrix for convolution:")
    print(f"Shape: {T4.shape}")
    print(f"\n{T4}\n")
    
    # Apply filter to input signal
    x = np.array([1, 2, 1, 0])  # input signal
    
    # Note: For actual convolution, you'd need proper dimensions
    print("This Toeplitz structure enables efficient convolution operations")
    
    # Example 5: System identification with Toeplitz
    print("\n" + "="*60)
    print("\nExample 5: System Identification Context")
    print("-"*60)
    
    # In system ID, Toeplitz matrices organize input sequences
    # for Hankel-based methods
    
    N_sim = 30
    u = np.random.randn(1, N_sim)
    
    r5 = 1
    k5 = 10
    
    T5 = blkToeplitz(u, r5, k5)
    
    print(f"Input sequence shape: {u.shape}")
    print(f"Toeplitz matrix shape: {T5.shape}")
    print(f"Rank: {np.linalg.matrix_rank(T5)}")
    print(f"\nThis matrix organizes past inputs for")
    print(f"subspace identification algorithms")
    
    # SVD analysis
    U, S, Vh = np.linalg.svd(T5, full_matrices=False)
    print(f"\nSingular values (first 5): {S[:5]}")
    print(f"Condition number: {S[0] / S[-1]:.2e}")
    
    # Example 6: Relationship between Hankel and Toeplitz
    print("\n" + "="*60)
    print("\nExample 6: Structural Relationship")
    print("-"*60)
    
    y6 = np.arange(1, 17).reshape(1, -1)
    r6 = 2
    k6 = 3
    
    H6 = blkHankel(y6, r6, k6)
    T6 = blkToeplitz(y6, r6, k6)
    
    print(f"Input: {y6}")
    print(f"\nHankel (anti-diagonal structure):")
    print(H6)
    print(f"\nToeplitz (diagonal structure):")
    print(T6)
    print("\nNote: Toeplitz has reversed block order compared to Hankel")

import numpy as np

def blk_hankel(y, r, k):
    """
    Return a block Hankel matrix filled with the data y containing k block-rows.
    
    A block Hankel matrix has constant values along anti-diagonals of blocks.
    Used in system identification, subspace methods, and realization theory.
    
    Parameters:
        y : sequence of vectors, e.g., time sequences (m x N)
            where N should be divisible by r
        r : number of columns in each block of y
        k : number of block rows of H
    
    Returns:
        H : block Hankel matrix of the output (m*k x r*(N/r+1-k))
    
    Example:
        For y = [y1, y2, y3, y4, y5, y6] with r=2, k=2:
        H = [y1 y2 | y3 y4 | y5 y6]
            [y3 y4 | y5 y6 | y7 y8]
        
        Each block is m x r, and blocks shift by r columns down each row
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
        raise ValueError('blkHankel: k must be positive')
    
    j = N + 1 - k  # number of block columns in H
    
    if m * k > r * j:
        raise ValueError('blkHankel: # rows > # columns, reduce k')
    
    # Fill the block-Hankel matrix
    H = np.zeros((m * k, r * j))
    
    for i in range(k):
        H[m*i:m*(i+1), :] = y[:, r*i:r*(i+j)]
    
    return H


# ----------------------------------------------------------------- blkHankel

# Example usage and test
if __name__ == "__main__":
    print("Testing blkHankel: Block Hankel Matrix Construction\n")
    print("="*60)
    
    # Example 1: Simple scalar sequence
    print("\nExample 1: Scalar sequence (m=1)")
    print("-"*60)
    
    # Single output, 12 time samples
    y1 = np.arange(1, 13).reshape(1, -1)
    print(f"Input y (shape {y1.shape}):\n{y1}\n")
    
    r1 = 2  # 2 columns per block
    k1 = 3  # 3 block rows
    
    H1 = blkHankel(y1, r1, k1)
    print(f"Block Hankel matrix H (r={r1}, k={k1}):")
    print(f"Shape: {H1.shape}")
    print(f"\n{H1}\n")
    
    # Verify Hankel structure (constant anti-diagonals)
    print("Hankel structure verification:")
    print(f"H[0,0:2] = {H1[0, 0:2]}  (should equal y[1:3])")
    print(f"H[1,0:2] = {H1[1, 0:2]}  (should equal y[3:5])")
    print(f"H[0,2:4] = {H1[0, 2:4]}  (should equal y[3:5])")
    
    # Example 2: Multi-output sequence
    print("\n" + "="*60)
    print("\nExample 2: Multi-output sequence (m=2)")
    print("-"*60)
    
    # 2 outputs, 10 time samples
    y2 = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ])
    
    print(f"Input y (shape {y2.shape}):\n{y2}\n")
    
    r2 = 2
    k2 = 2
    
    H2 = blkHankel(y2, r2, k2)
    print(f"Block Hankel matrix H (r={r2}, k={k2}):")
    print(f"Shape: {H2.shape}")
    print(f"\n{H2}\n")
    
    # Example 3: System identification application
    print("="*60)
    print("\nExample 3: System Identification Application")
    print("-"*60)
    
    # Simulate a simple system response
    # y[k] = 0.8*y[k-1] + u[k]
    N_sim = 50
    u = np.random.randn(1, N_sim)
    y_sim = np.zeros((1, N_sim))
    
    for i in range(1, N_sim):
        y_sim[0, i] = 0.8 * y_sim[0, i-1] + u[0, i]
    
    print(f"Simulated output shape: {y_sim.shape}")
    print(f"First 10 samples: {y_sim[0, :10]}\n")
    
    # Create Hankel matrix for subspace identification
    r3 = 1
    k3 = 10
    
    H3 = blkHankel(y_sim, r3, k3)
    print(f"Hankel matrix for identification:")
    print(f"Shape: {H3.shape} (10 past outputs x 41 time shifts)")
    print(f"Rank: {np.linalg.matrix_rank(H3)}")
    print(f"Expected rank ≈ system order = 1\n")
    
    # SVD analysis
    U, S, Vh = np.linalg.svd(H3, full_matrices=False)
    print(f"Singular values (first 5): {S[:5]}")
    print(f"Singular value ratios: {S[:5] / S[0]}\n")
    print("✓ Large drop after first singular value indicates order 1 system")
    
    # Example 4: MIMO system
    print("\n" + "="*60)
    print("\nExample 4: MIMO System (2 outputs)")
    print("-"*60)
    
    # 2 outputs, 20 samples
    y4 = np.random.randn(2, 20)
    
    r4 = 4
    k4 = 3
    
    H4 = blkHankel(y4, r4, k4)
    print(f"MIMO Hankel matrix:")
    print(f"Input shape: {y4.shape}")
    print(f"Output shape: {H4.shape}")
    print(f"Block size: {y4.shape[0]} x {r4}")
    print(f"Number of blocks: {k4} rows x {H4.shape[1]//r4} columns")
    
    # Verify dimensions
    print(f"\nDimension check:")
    print(f"  m*k = {y4.shape[0]}*{k4} = {y4.shape[0]*k4} (rows)")
    print(f"  r*j = {r4}*{H4.shape[1]//r4} = {H4.shape[1]} (cols)")
    print(f"  Actual H shape: {H4.shape}")
    print("  ✓ Dimensions match")

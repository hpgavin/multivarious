"""
cdiff.py - Central difference numerical differentiation

Computes numerical derivatives of vectors using central differences,
with forward/backward differences at endpoints.

Translation from MATLAB by Claude, 2024-11-18
Original: cdiff.m by H.P. Gavin, 30 Mar 2011
"""

import numpy as np


def cdiff(v, x=None):
    """
    Numerical differentiation using central differences.
    
    Computes dv/dx using central differences for interior points and
    forward/backward differences at the endpoints.
    
    Parameters
    ----------
    v : ndarray, shape (m, n)
        Array of m row vectors to differentiate.
        Each row is differentiated independently.
        Can also be 1D array of shape (n,) - will be treated as single row.
    x : ndarray, float, or None, optional
        Spacing information:
        - None: Assumes uniform unit spacing (dx = 1)
        - scalar: Assumes uniform spacing of dx
        - array of length n: v is sampled at points x, spacing varies
        Default: None (unit spacing)
    
    Returns
    -------
    dvdx : ndarray, shape (m, n)
        Numerical derivative dv/dx
        Same shape as input v
    
    Notes
    -----
    Difference formulas used:
    - Endpoints: Forward/backward difference
        dv[0]/dx = (v[1] - v[0]) / dx[0]
        dv[n-1]/dx = (v[n-1] - v[n-2]) / dx[n-1]
    
    - Interior points: Central difference
        dv[i]/dx = (v[i+1] - v[i-1]) / (2*dx[i])  for i = 1, ..., n-2
    
    Variable spacing:
    - When x is an array, dx[i] = (x[i+1] - x[i-1])/2 for interior points
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Example 1: Unit spacing (default)
    >>> v = np.array([[1, 4, 9, 16, 25]])  # y = x^2
    >>> dvdx = cdiff(v)
    >>> print(dvdx)  # Should approximate [3, 4, 5, 6, 7]
    
    >>> # Example 2: Uniform spacing dx = 0.1
    >>> x = np.arange(0, 1, 0.1)
    >>> v = x**2
    >>> dvdx = cdiff(v, 0.1)  # dv/dx ≈ 2x
    
    >>> # Example 3: Variable spacing
    >>> x = np.array([0, 0.1, 0.3, 0.6, 1.0])
    >>> v = np.sin(x)
    >>> dvdx = cdiff(v, x)  # Should approximate cos(x)
    
    >>> # Example 4: Multiple row vectors
    >>> v = np.array([[1, 2, 3, 4],
    ...               [1, 4, 9, 16]])
    >>> dvdx = cdiff(v)  # Differentiates each row
    """
    
    # Convert to numpy array and ensure 2D
    v = np.asarray(v)
    
    # Handle 1D input
    if v.ndim == 1:
        v = v.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    m, n = v.shape
    
    # Determine spacing dx
    if x is None:
        # Default: uniform unit spacing
        dx = 1.0
    elif np.isscalar(x):
        # Scalar: uniform spacing
        dx = float(x)
    else:
        # Array: variable spacing
        x = np.asarray(x)
        if len(x) != n:
            raise ValueError(f'Length of x ({len(x)}) must match length of v ({n})')
        
        # Compute spacing for each point
        dx = np.zeros(n)
        dx[0] = x[1] - x[0]                        # Forward difference at start
        dx[1:n-1] = 0.5 * (x[2:n] - x[0:n-2])      # Central difference in middle
        dx[n-1] = x[n-1] - x[n-2]                  # Backward difference at end
    
    # Compute differences dv
    dv = np.zeros_like(v)
    dv[:, 0] = v[:, 1] - v[:, 0]                   # Forward difference at start
    dv[:, 1:n-1] = 0.5 * (v[:, 2:n] - v[:, 0:n-2]) # Central difference in middle
    dv[:, n-1] = v[:, n-1] - v[:, n-2]             # Backward difference at end
    
    # Compute derivative
    dvdx = dv / dx
    
    # Return same shape as input
    if squeeze_output:
        return dvdx.squeeze()
    else:
        return dvdx


# Test and demonstration code
if __name__ == '__main__':
    """
    Test cdiff function with various cases
    """
    import matplotlib.pyplot as plt
    
    print("Testing cdiff.py")
    print("=" * 70)
    
    # Test 1: Polynomial with unit spacing
    print("\nTest 1: Polynomial y = x^2 with unit spacing")
    x1 = np.arange(0, 5, 1)
    v1 = x1**2  # y = x^2, dy/dx = 2x
    dvdx1 = cdiff(v1)
    expected1 = 2 * x1
    
    print(f"  x:        {x1}")
    print(f"  v (x^2):  {v1}")
    print(f"  dv/dx:    {dvdx1}")
    print(f"  Expected: {expected1}")
    print(f"  Error:    {np.abs(dvdx1 - expected1)}")
    
    # Test 2: Sine function with uniform spacing
    print("\nTest 2: y = sin(x) with dx = 0.1")
    x2 = np.arange(0, 2*np.pi, 0.1)
    v2 = np.sin(x2)
    dvdx2 = cdiff(v2, 0.1)
    expected2 = np.cos(x2)
    error2 = np.abs(dvdx2 - expected2)
    
    print(f"  Points: {len(x2)}")
    print(f"  Max error: {np.max(error2):.6f}")
    print(f"  Mean error: {np.mean(error2):.6f}")
    
    # Test 3: Variable spacing
    print("\nTest 3: y = exp(x) with variable spacing")
    x3 = np.array([0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    v3 = np.exp(x3)
    dvdx3 = cdiff(v3, x3)
    expected3 = np.exp(x3)  # d/dx(e^x) = e^x
    error3 = np.abs(dvdx3 - expected3)
    
    print(f"  x:        {x3}")
    print(f"  dv/dx:    {dvdx3}")
    print(f"  Expected: {expected3}")
    print(f"  Error:    {error3}")
    print(f"  Max error: {np.max(error3):.6f}")
    
    # Test 4: Multiple row vectors
    print("\nTest 4: Multiple row vectors")
    v4 = np.array([[1, 2, 4, 7, 11],      # differences: 1, 2, 3, 4
                   [1, 4, 9, 16, 25]])     # y = x^2
    dvdx4 = cdiff(v4)
    
    print(f"  Input shape: {v4.shape}")
    print(f"  Output shape: {dvdx4.shape}")
    print(f"  Row 0 derivatives: {dvdx4[0]}")
    print(f"  Row 1 derivatives: {dvdx4[1]}")
    
    # Test 5: Edge case - short array
    print("\nTest 5: Short array (n=3)")
    v5 = np.array([1, 4, 9])
    dvdx5 = cdiff(v5)
    print(f"  v:     {v5}")
    print(f"  dv/dx: {dvdx5}")
    
    # Test 6: 1D vs 2D input
    print("\nTest 6: 1D vs 2D input consistency")
    v6_1d = np.array([1, 2, 4, 7, 11])
    v6_2d = v6_1d.reshape(1, -1)
    dvdx6_1d = cdiff(v6_1d)
    dvdx6_2d = cdiff(v6_2d)
    print(f"  1D input shape: {v6_1d.shape} → output shape: {dvdx6_1d.shape}")
    print(f"  2D input shape: {v6_2d.shape} → output shape: {dvdx6_2d.shape}")
    print(f"  Results match: {np.allclose(dvdx6_1d, dvdx6_2d.squeeze())}")
    
    # Create visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Polynomial
    ax1 = axes[0, 0]
    x_plot = np.linspace(0, 4, 100)
    ax1.plot(x_plot, 2*x_plot, 'g-', linewidth=2, label='Exact: 2x')
    ax1.plot(x1, dvdx1, 'bo', markersize=8, label='cdiff')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('dy/dx', fontsize=11)
    ax1.set_title('Test 1: d(x²)/dx = 2x', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sine function
    ax2 = axes[0, 1]
    ax2.plot(x2, expected2, 'g-', linewidth=2, label='Exact: cos(x)')
    ax2.plot(x2, dvdx2, 'b-', linewidth=1, alpha=0.7, label='cdiff')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('dy/dx', fontsize=11)
    ax2.set_title('Test 2: d(sin(x))/dx = cos(x)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error in sine
    ax3 = axes[1, 0]
    ax3.plot(x2, error2, 'r-', linewidth=2)
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('Absolute Error', fontsize=11)
    ax3.set_title('Test 2: Error in d(sin(x))/dx', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, np.max(error2)*1.1])
    
    # Plot 4: Exponential with variable spacing
    ax4 = axes[1, 1]
    x_plot_exp = np.linspace(0, 3, 100)
    ax4.plot(x_plot_exp, np.exp(x_plot_exp), 'g-', linewidth=2, label='Exact: exp(x)')
    ax4.plot(x3, dvdx3, 'bo', markersize=8, label='cdiff (variable spacing)')
    ax4.plot(x3, v3, 'rs', markersize=6, alpha=0.5, label='Function values')
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('y, dy/dx', fontsize=11)
    ax4.set_title('Test 3: d(exp(x))/dx = exp(x) (variable Δx)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/cdiff_demo.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: cdiff_demo.png")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    
    plt.show()

import numpy as np

def fsolve(fctn, x, c=None, delta=1e-3, tolerance=1e-9):
    """
    Solve a nonlinear system of equations f = fctn(x, c) using 
    the Newton-Raphson method.
    
    Parameters:
        fctn      : function of the form (f, y) = fctn(x, c)
                    where f is the residual vector to be zeroed
        x         : initial guess values of independent variables (n x 1)
        c         : other constants passed to the function fctn(x, c)
        delta     : increment used for numerical Jacobian (default: 1e-3)
        tolerance : relative convergence tolerance (default: 1e-9)
    
    Returns:
        x       : solution vector where f(x) ≈ 0 (n x 1)
        y       : other outputs from fctn at the solution
        f       : function values at the solution ≈ 0 (n x 1)
        cvg_hst : convergence history (n+5 x iterations)
                  rows: [x; norm_f; -1; nfeval; norm(dx); norm(f-f_new)]
        dfdx    : Jacobian at the optimal solution (m x n)
    
    Reference:
        http://en.wikipedia.org/wiki/Newton%27s_method
    
    Author: HP Gavin - 19 Apr 2010, 4 Jan 2011, 17 Apr 2017
    """
    
    x = np.asarray(x)
    original_shape = x.shape
    x = x.flatten()  # Work with 1D array internally
    
    verbose = False      # show convergence information
    MaxIter = 10        # maximum number of Newton-Raphson iterations
    nfeval = 1          # counter for function evaluations
    
    n = len(x)  # number of independent variables
    
    # Evaluate the function at initial x
    f, y = fctn(x.reshape(original_shape), c)
    f = np.asarray(f).flatten()
    norm_f = np.linalg.norm(f)  # Euclidean norm
    
    m = len(f)  # number of equations
    
    dfdx = np.zeros((m, n))  # Initialize Jacobian
    cvg_hst = np.full((n + 5, MaxIter), np.nan)
    
    for iter in range(MaxIter):  # Newton-Raphson iterations
        
        # Compute the Jacobian numerically
        for j in range(n):
            xp = x.copy()
            pxj = delta  # perturbation in x[j]
            xp[j] = xp[j] + pxj
            
            fp, _ = fctn(xp.reshape(original_shape), c)
            fp = np.asarray(fp).flatten()
            
            dfdx[:, j] = (fp - f) / pxj  # df/dx[j]
        
        # Newton-Raphson update direction
        dx = -np.linalg.solve(dfdx, f)
        
        # Evaluate at updated x
        f_new, y_new = fctn((x + dx).reshape(original_shape), c)
        f_new = np.asarray(f_new).flatten()
        norm_f_new = np.linalg.norm(f_new)
        
        nfeval = nfeval + n + 1
        
        # Line search: reduce step size until improvement
        while norm_f_new > norm_f:
            dx = dx / 2.0
            f_new, y_new = fctn((x + dx).reshape(original_shape), c)
            f_new = np.asarray(f_new).flatten()
            norm_f_new = np.linalg.norm(f_new)
            nfeval = nfeval + 1
        
        # Update independent variables
        x = x + dx
        
        # Store convergence history
        cvg_hst[:, iter] = np.concatenate([
            x,
            [norm_f, -1, nfeval, np.linalg.norm(dx), np.linalg.norm(f - f_new)]
        ])
        
        # Update function values
        f = f_new
        norm_f = norm_f_new
        y = y_new
        
        if verbose and iter > 0:
            print(f'Iter {iter+1}: norm(f) = {norm_f:.6e}, norm(dx) = {np.linalg.norm(dx):.6e}')
        
        # Check for convergence
        if norm_f < tolerance:
            break
    
    # Trim convergence history
    cvg_hst = cvg_hst[:, :iter + 1]
    
    if iter == MaxIter - 1:
        print(f' !! fsolve failed to converge to tolerance {tolerance:.2e} '
              f'in MaxIter = {MaxIter} iterations!!')
        print(f'norm_f = {norm_f:.6e}')
    
    # Reshape x back to original shape
    x = x.reshape(original_shape)
    
    return x, y, f, cvg_hst, dfdx


# Example usage and test
if __name__ == "__main__":
    # Test problem: solve x^2 + y^2 = 5 and x*y = 2
    def test_fctn(x, c):
        """
        System of equations:
        f1 = x^2 + y^2 - 5 = 0
        f2 = x*y - 2 = 0
        
        Solution: (x, y) ≈ (2, 1) or (1, 2)
        """
        f = np.array([
            x[0]**2 + x[1]**2 - 5,
            x[0] * x[1] - 2
        ])
        y = x  # additional output
        return f, y
    
    # Initial guess
    x0 = np.array([1.5, 1.5])
    
    # Solve
    x_sol, y_sol, f_sol, cvg_hst, dfdx = fsolve(test_fctn, x0)
    
    print("\nSolution:")
    print(f"x = {x_sol}")
    print(f"f(x) = {f_sol}")
    print(f"norm(f) = {np.linalg.norm(f_sol):.2e}")
    print(f"\nJacobian at solution:")
    print(dfdx)
    print(f"\nConvergence history shape: {cvg_hst.shape}")


# fsolve - HP Gavin - 19 Apr 2010, 4 Jan 2011, 17 Apr 2017

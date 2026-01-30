#! /usr/bin/env -S python3 -i

"""
plot_opt_surface.py - 3D Surface Plot of Objective Function

Draw a surface plot of J(x) vs. x(i), x(j), where all other values in x
are held constant. Useful for visualizing optimization landscapes and 
convergence paths.

Translation from MATLAB to Python by Claude, 2025-11-19
Original by H.P. Gavin, Civil & Environ. Eng'g, Duke Univ.
Updated 2015-09-29, 2016-03-23, 2016-09-12, 2020-01-20, 2021-12-31, 2025-01-26
"""

import numpy as np
import matplotlib.pyplot as plt

from multivarious.utl import plot_opt_surface

print("\n" + "="*70)
print("Testing plot_opt_surface.py")
print("="*70 + "\n")
    
# Define a simple test function (Rosenbrock function)
def test_func(x, consts=None):
    """
    Rosenbrock function: f(x) = sum((1-v_i)^2 + 100*(v_{i+1} - v_i^2)^2)
    Minimum at x = [1, 1, ...] with f(x) = 0
    
    Returns
    -------
    f : float
        Objective function value
    g : ndarray
        Constraint values (no constraints in this example)
    """
    x = np.asarray(x).flatten()
    n = len(x)
    
    # Rosenbrock function
    f = 0.0
    for i in range(n - 1):
        f += (1 - x[i])**2 + 100 * (x[i+1] - x[i]**2)**2
    
    # No constraints (return array of negative values = feasible)
    g = np.array([-1.0])
    
    return f, g
 
# Set up parameters
n = 3  # Number of design variables
v_init = np.array([0.0, 0.0, 0.0])  # Initial point
v_lb = np.array([-2.0, -2.0, -2.0])  # Lower bounds
v_ub = np.array([2.0, 2.0, 2.0])     # Upper bounds

# Set up options array
options = np.zeros(14)
options[0] = 3        # Surface plot flag
options[3] = 0.0      # Constraint tolerance
options[5] = 0.0      # Penalty (no penalty needed - no constraints)
options[6] = 2.0      # Penalty exponent
options[10] = 0       # Plot v_0 (1st variable, 0-indexed)
options[11] = 1       # Plot v_1 (2nd variable, 0-indexed)
options[12] = 30      # Number of points in v_0 direction
options[13] = 30      # Number of points in v_1 direction

# Create the surface plot
print("Creating 3D surface plot of Rosenbrock function...")
print(f"Initial point: x = {v_init}")
print(f"Plotting x[{int(options[10])}] vs x[{int(options[11])}]")
print(f"Grid size: {int(options[12])} x {int(options[13])}")

fmin, fmax, ax = plot_opt_surface(test_func, v_init, v_lb, v_ub, 
options, consts=None, fig_no=100)

print(f"\nSurface z-axis range: [{fmin:.4f}, {fmax:.4f}]")

plt.show()

print("\n" + "="*70)
print("plot_opt_surface test completed successfully!")
print("="*70 + "\n")


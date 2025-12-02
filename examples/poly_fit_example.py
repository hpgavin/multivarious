#! /usr/bin/python3 -i 

"""
poly_fit_example.py - example for the use of poly_fit.py
2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Test poly_fit with example data
"""
    
print("\n" + "="*70)
print("Testing poly_fit.py")
print("="*70)
    
# Generate test data
np.random.seed(42)
x_l = -1
x_h = 1
Nd = 40
    
x = np.linspace(x_l, x_h, Nd)
    
measurement_error = 0.20
y = -np.cos(4*x) + 1.0 * x**3 * np.exp(-x/3) + measurement_error * np.random.randn(Nd)
    
print(f"\nGenerated {Nd} data points")
print(f"x range: [{x_l}, {x_h}]")
print(f"Measurement error (RMS): {measurement_error}")
    
# Test 1: Full polynomial basis
print("\n" + "-"*70)
print("Test 1: Powers [0, 1, 2, 3, 4]")
print("-"*70)
    
p1 = np.array([0, 1, 2, 3, 4])
c1, x_fit1, y_fit1, Sc1, Sy_fit1, Rc1, R2_1, Vr1, AIC1, cond1 = \
        poly_fit(x, y, p1, figNo=10)
    
# Test 2: Reduced polynomial basis (no linear term)
print("\n" + "-"*70)
print("Test 2: Powers [0, 2, 3, 4]")
print("-"*70)
    
p2 = np.array([0, 2, 3, 4])
c2, x_fit2, y_fit2, Sc2, Sy_fit2, Rc2, R2_2, Vr2, AIC2, cond2 = \
        poly_fit(x, y, p2, figNo=20)
    
# Comparison
print("\n" + "="*70)
print("Comparison of Models")
print("="*70)
print(f"{'Metric':<20} {'Model 1 (5 terms)':<20} {'Model 2 (4 terms)':<20}")
print("-"*70)
print(f"{'R²':<20} {R2_1:<20.4f} {R2_2:<20.4f}")
print(f"{'AIC':<20} {AIC1:<20.2f} {AIC2:<20.2f}")
print(f"{'Cond. Number':<20} {cond1:<20.1f} {cond2:<20.1f}")
print(f"{'Residual Var':<20} {Vr1:<20.4f} {Vr2:<20.4f}")
print("="*70)

if AIC2 < AIC1:
    print("\n✓ Model 2 is preferred (lower AIC)")
else:
    print("\n✓ Model 1 is preferred (lower AIC)")
    
plt.show()
    
print("\n" + "="*70)
print("poly_fit test completed successfully!")
print("="*70)

#! /usr/bin/env -S python3 -i 

import numpy as np
from multivarious.utl import fix_R

n = 15

v = 2.0 * np.random.rand( int(n*(n-1)/2) ) - 1.0

upper_tri_idx = np.triu_indices(n, k=1)

Ro = np.eye(n,n)
Ro[upper_tri_idx] = v
Ro[(upper_tri_idx[1], upper_tri_idx[0])] = v

R = fix_R(Ro,1e-2)

eValRo, _ = np.linalg.eigh(Ro)
eValR, _  = np.linalg.eigh(R)

print('\n Ro')
for row in Ro:
    print(" ".join(f"{num:10.3f}" for num in row))

print('\n eValRo')
print(" ".join(f"{num:10.3f}" for num in eValRo))

print('\n R')
for row in R:
    print(" ".join(f"{num:10.3f}" for num in row))

print('\n evalR')
print(" ".join(f"{num:10.3f}" for num in eValR))

print('\n R-Ro')
for row in (R-Ro):
    print(" ".join(f"{num:10.3f}" for num in row))

# -------------------------------------------------------------------

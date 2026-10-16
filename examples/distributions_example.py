#! /usr/bin/env -S /usr/bin/python3 -i 

import numpy as np
from collections import namedtuple

from multivarious.rvs import exponential, normal, rayleigh 
from multivarious.utl import plot_ECDF_ci
from multivarious.fit import distributions

# Generate sample data
#np.random.seed(42)  # Set a seed for reproducibility
n = 50          # Sample size
mean = 10         # population mean
stdv = mean / 3   # population standard deviation

normal_data   = normal.rnd(mean, stdv, n, 1, seed=None)
rayleigh_data = rayleigh.rnd(mean, n, 1, seed=None)
weight = 0.99
combined_data = weight * normal_data + (1-weight ) *rayleigh_data

# Show empirical CDF of the data ============================================
plot_ECDF_ci( combined_data, 99, 201, 'combined data', False )

# Fit distributions to the combined data ====================================
coeffs, criteria, tabular_output = distributions(combined_data)

# Print the tabular output ==================================================
print(tabular_output)


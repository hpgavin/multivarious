#! /usr/bin/env -S python3 -i

"""
plot_ECDF_ci_exampple.py - script for testing plot_ECDF_ci.py
a demonstration of plot_ECDF_ci
2025-12-02, 2026-02-02
"""

import numpy as np
from multivarious.utl import plot_ECDF_ci

N = 40 # x is a sample of 3N observations 
x = np.array([2 + 0.4*np.random.randn(N), 
              5 + 1.0*np.random.randn(N), 
             10 + 2.0*np.random.randn(N)])

fig_no = 100  # figure number 
# an empirical CDF and PDF (Histogram) from the sample x
plot_ECDF_ci( x , confidence_level =  95 , # confidence level  (times 100)
                 fig_no = fig_no ,         # figure number 
                 x_label = 'samples from a Gaussian mixture' , 
                 save_plots = True )

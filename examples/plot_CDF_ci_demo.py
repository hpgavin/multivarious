#! /usr/bin/python3 -i

import numpy as np
import matplotlib.pyplot as plt
from lognormal import rnd as ln_rnd
from plot_CDF_ci import plot_CDF_ci

# a demonstration of plot_CDF_ci

# x is a sample of N observations 
x = ln_rnd( medX =   5.0  ,  # the median of the population 
            covX =   0.3  ,  # the coefficient of variation of the population
               N = 100 )     # number of observations in the sample

# make an empirical CDF plot from the sample x
plot_CDF_ci( x , confidence_level =  95 , # confidence level  (times 100)
                 fig_no           = 100 , # figure number 
                 x_label = 'samples from a log-normal distribution' , 
                 norm_inv_CDF = False )
                 # False: plot F(x) vs. x,  True: plot norm_inv(F(x)) vs. x

#plt.figure(101)
#plt.hist(x.T,50)

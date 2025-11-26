import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaincinv

def plot_CDF_ci(data, confidence_level, figNo, x_label='x'):
    """
    Plot empirical CDF of data with confidence intervals.
    
    Parameters
    ----------
    data : array_like
        Data values
    confidence_level : float
        Confidence level (e.g., 95 for 95%)
    figNo : int
        Figure number
    x_label : string
        label for the x axis
    """
    
    data = np.asarray(data).flatten()
    N = len(data)
    
    # Sort data
    sorted_data = np.sort(data)
    
    # Empirical CDF
    Fx = np.arange(1, N + 1) / ( N + 1 )    # Gumbel (1958)

    # Confidence intervals on Fx
    alpha = 1 - confidence_level / 100
    P = 1 - alpha/2 

    # Confidence intervals on Fx using Gumble (p.47) and normal distribution
    #sFx = np.sqrt(Fx*(1-Fx)/(N+2))             # standard error of Fx Gumbel p.47
    #z   = -5.531 * ( ((1-P)/P)**0.1193 - 1.0 ) # norm_inv, Shore, 1982 P > 0.5
    #epsilon = z*sFx

    # Confidence intervals on Fx using Dvoretzky-Kiefer-Wolfowitz inequality
    # https://en.wikipedia.org/wiki/Dvoretzky-Kiefer-Wolfowitz_inequality
    # epsilon = np.sqrt(np.log(2 / alpha) / (2 * N))

    #upper_ci = np.minimum(Fx + epsilon, 1.0 - 0.001 / N)
    #lower_ci = np.maximum(Fx - epsilon, 0.0 + 0.001 / N)
    
    # Confidence interval on Fx using the Beta distribution 
    # variance of the Beta distribion is (F)(1-F)/(N+2)
    q = np.arange(1,N+1)
    p = N+1-q
    lower_ci = betaincinv(q, p,   P)  # lower quantile, near zero ( ci > 0 )
    upper_ci = betaincinv(q, p, 1-P)  # upper quantile, near one  ( ci < 1 )

    # normal CDF
    mu, std = np.mean(data), np.std(data)
    x_theory = np.linspace(min(sorted_data), max(sorted_data), 200)
    z_theory = (x_theory - mu) / std
    ee = np.exp( 2 * np.sqrt( 2 / np.pi / np.abs(z_theory) )  )
    cdf_theory = ee / ( 1 + ee )    # approx to norm CDF, Torcher (1963)
    
    # Plotting ---------------------------------------------------
    plt.ion() # plot interactive mode: on
    fig = plt.figure(figNo, figsize=(10, 7))
    fig.clf()
    
    plt.fill_between(sorted_data, lower_ci, upper_ci, 
                     color=[0.8, 0.9, 1.0], alpha=0.5,
                     label=f'{confidence_level}% confidence band')
    plt.step(sorted_data, Fx, '-', color=[0.2, 0.4, 0.8], 
             linewidth=2, markersize=4, label='Empirical CDF')
    #plt.plot(sorted_data, upper_ci_B, '-r')
    #plt.plot(sorted_data, lower_ci_B, '-r')
    #plt.plot(sorted_data, upper_ci_DKW, '-g')
    #plt.plot(sorted_data, lower_ci_DKW, '-g')
    #plt.plot(x_theory, cdf_theory, 'r-', linewidth=2, 
    #         label=f'Normal CDF (μ={mu:.3f}, σ={std:.3f})')
    #plt.plot(sorted_data, sorted_data * 0 + 0.5, '--k', alpha=0.3, linewidth=1)
    
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Cumulative probability', fontsize=16)
    plt.title(f'Empirical CDF with {confidence_level}% Confidence Intervals', 
              fontsize=16)
    plt.legend(fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.draw()

    '''
    Dvoretzky, A., Kiefer, J., Wolfowitz, J. (1956),
    "Asymptotic minimax character of the sample distribution function and of
    the classical multinomial estimator",
    Annals of Mathematical Statistics, 27 (3): 642-669.
    doi:10.1214/aoms/1177728174, MR 0083864

    Gumbel, Emil Julius (1958)
    Statistics of Extremes. Columbia University Press, New York. 

    Shore, H. (1982).
    Approximations to the inverse cumulative normal function for use
    on hand calculator. Applied Statistics, 28, 175-176.

    Tocher K. D. (1963).
    The Art of Simulation. English Universities Press, London.
    '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaincinv
from scipy.stats import norm
from multivarious.utl import format_plot

def plot_ECDF_ci(data, confidence_level, fig_no, x_label='sorted sample values',save_plots=True ):
    """
    Plot empirical CDF of data with confidence intervals.
    
    Parameters
    ----------
    data : array_like
        Data values
    confidence_level : float
        Confidence level (e.g., 95 for 95%)
    fig_no : int
        Figure number
    x_label : string
        label for the x axis
    pdf_too : boolean
        True: make two subplots, histogram and ECDF , False: CDF only (default)
    save_plots : boolean
        True: save plots to file (default) , False: don't

    Parameters
    ----------
    x : array_like
        orderd values of x
    Fx : array_like
        the empirical cumulative distribution
    lower_ci: array_like
        empirical distribution lower confidence interval
    upper_ci: array_like
        empirical distribution upper confidence interval
    
    """
    
    x = np.asarray(data).flatten()

    N = len(x)                                 # number of values in the sample
    x = np.sort(x)                             # sort the sample
    x_avg = np.sum(x) / N                      # average of the sample
    x_med = x[int(round(N / 2))]               # median of the sample
    x_sdv = np.sqrt((x-x_avg)@(x-x_avg)/(N-1)) # stndrd dev of the sample
    x_cov = abs(x_sdv / x_avg)                 # coefficient of variation

    plt.ion() # interactive plotting mode: on

    # Empirical PDF - histogram 
    #fx,xx = np.histogram(x, bins=nBins, density=True)  # EPDF

    # Empirical CDF - Gumbel p.47 "contrary to what one might expect"
    Fx = np.arange(1, N + 1) / ( N + 1 )    # ECDF Gumbel (1958)

    # Confidence intervals on the ECDF (Fx)
    # The variance of the emperical CDF is (F)(1-F)/(N+2)   # Gumbel (1958)
    alpha = 1 - confidence_level / 100
    P = 1 - alpha/2 

    # --- simultaneous, conservative: using Dvoretzky-Kiefer-Wolfowitz inequality
    # https://en.wikipedia.org/wiki/Dvoretzky-Kiefer-Wolfowitz_inequality
    '''
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * N))
    upper_ci_DKW = np.minimum(Fx + epsilon, 1.0 - 0.001 / N)
    lower_ci_DKW = np.maximum(Fx - epsilon, 0.0 + 0.001 / N)
    '''
    
    # --- simultaneous, exact: Dirichlet-based simultaneous emperical CDF bands
    '''
    M = 30000
    G = np.random.gamma(1.0, 1.0, size=(M, N+1))
    D = G / G.sum(axis=1, keepdims=True)
    U = D.cumsum(axis=1)[:, :N]
    lower_ci_D = np.quantile(U, alpha/2, axis=0)
    upper_ci_D = np.quantile(U, 1-alpha/2, axis=0)
    '''

    # --- pointwise, approximate: using Gumble (p.47) and normal distribution
    '''
    sFx = np.sqrt(Fx*(1-Fx)/(N+2))             # standard error of Fx Gumbel p.47
    z   = norm.ppf(P) 
    epsilon = z*sFx
    upper_ci_A = np.minimum(Fx + epsilon, 1.0 - 0.001 / N)
    lower_ci_A = np.maximum(Fx - epsilon, 0.0 + 0.001 / N)
    '''
    
    # --- pointwise, exact: Beta-based empirical CDF bands
    #'''
    q = np.arange(1,N+1)
    p = N+1-q
    upper_ci_B = betaincinv(q, p,   P)  # upper quantile, near zero ( ci > 0 )
    lower_ci_B = betaincinv(q, p, 1-P)  # lower quantile, near one  ( ci < 1 )
    #'''

    # select which confidence interval to plot
    lower_ci = lower_ci_B
    upper_ci = upper_ci_B

    # a normal CDF for testing 
    #mu, std = np.mean(data), np.std(data)
    #x_theory = np.linspace(min(x), max(x), 200)
    #z_theory = (x_theory - mu) / std
    #cdf_theory = norm.cdf(z_theory)  
    
    # Plotting ---------------------------------------------------
    fs = 14
    format_plot(line_width = 2, font_size = fs, marker_size = 4)

    fig_cdf = plt.figure(fig_no) 
    fig_cdf.set_size_inches(8, 5)  

    
    plt.fill_between(x, lower_ci, upper_ci, 
                     color='royalblue', alpha=0.6,
                     label=f'{confidence_level}% confidence band')

    plt.step(x, Fx, '-', color='darkblue', label='Empirical CDF')

    aa = 0.25
    xp = aa*x[0] + (1-aa)*x[-1]
    plt.text(xp, 0.4, rf'$x_{{avg}}$ = {x_avg:.3f}')
    plt.text(xp, 0.3, rf'$x_{{med}}$ = {x_med:.3f}')
    plt.text(xp, 0.2, rf'$x_{{sdv}}$ = {x_sdv:.3f}')
    if x_cov < 10:
        plt.text(xp, 0.1, rf'$x_{{cov}}$ = {x_cov:.3f}')

    plt.title(f'Empirical CDF with {confidence_level}% confidence Intervals')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(r'Cumulative Distribution, $F_X(x)$', fontsize=15)
    plt.tight_layout()
    #plt.legend()
    #plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    """
    # Empirical PDF - histogram ... with density=True
    fig_pdf = plt.figure(fig_no+1)
    fig_pdf.set_size_inches(6, 4)  
    plt.hist(x.T, bins=20, density=True, color='royalblue', alpha=0.7, edgecolor='black')
    plt.title(f'Histogram (scaled)')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'Probability Density, $f_X(x)$')
    plt.xlabel(x_label)
    plt.tight_layout()
    """
    
    plt.show()

    # Save plots 
    if save_plots:
        filename = f'plot_ECDF-{fig_no:04d}.pdf'
        fig_cdf.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"    Saved: {filename}")
#       filename = f'plot_EPDF-{fig_no:04d}.pdf'
#       fig_pdf.savefig(filename, bbox_inches='tight', dpi=300)
#       print(f"    Saved: {filename}")

    # plot other CI's 
    '''
    plt.plot(x, upper_ci, '-r')
    plt.plot(x, lower_ci, '-r')
    plt.step(x, upper_ci_D, '-r' ) 
    plt.step(x, lower_ci_D, '-r' ) 
    plt.plot(x, upper_ci_DKW, '-g')
    plt.plot(x, lower_ci_DKW, '-g')
    '''
    # plot another CDF
    '''
    plt.plot(x_theory, cdf_theory, '-', color='darkblue',linewidth=2, 
             label=f'Normal CDF (μ={mu:.3f}, σ={std:.3f})')
    plt.plot(x, x * 0 + 0.5, '--k', alpha=0.3, linewidth=1)
    '''
    
    return x, Fx, lower_ci, upper_ci
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

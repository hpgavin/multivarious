import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaincinv
from scipy.stats import norm

def plot_CDF_ci(data, confidence_level, fig_num, x_label='sorted sample values', norm_inv_CDF=False ):
    """
    Plot empirical CDF of data with confidence intervals.
    
    Parameters
    ----------
    data : array_like
        Data values
    confidence_level : float
        Confidence level (e.g., 95 for 95%)
    fig_num : int
        Figure number
    x_label : string
        label for the x axis
    norm_inv_CDF : boolean
        True: plot norm inverse of F , False: plot F (default)
    """

    #plt.rcParams['text.usetex'] = True # Set to True if LaTeX is installed

    pdf_plots = True  # Set to True to save PDF files
    interactive = True # Enable interactive mode for matplotlib
    
    if interactive:
        plt.ion() # plot interactive mode: on

    data = np.asarray(data).flatten()
    N = len(data)
    
    # Empirical CDF
    Fx = np.arange(1, N + 1) / ( N + 1 )    # Gumbel (1958)

    # Sort data
    sorted_data = np.sort(data)
    
    # Confidence intervals on Fx
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
    q = np.arange(1,N+1)
    p = N+1-q
    upper_ci_B = betaincinv(q, p,   P)  # upper quantile, near zero ( ci > 0 )
    lower_ci_B = betaincinv(q, p, 1-P)  # lower quantile, near one  ( ci < 1 )

    # a normal CDF for testing 
    #mu, std = np.mean(data), np.std(data)
    #x_theory = np.linspace(min(sorted_data), max(sorted_data), 200)
    #z_theory = (x_theory - mu) / std
    #cdf_theory = norm.cdf(z_theory)  
    
    # Plotting ---------------------------------------------------
    fig = plt.figure(fig_num, figsize=(10, 7))
    fig.clf()

    # select which confidence interval to plot
    lower_ci = lower_ci_B
    upper_ci = upper_ci_B
    
    if norm_inv_CDF:

       # apply the standard normal inverse function to Fx and its CI's 
       lower_ci = norm.ppf(lower_ci)   
       upper_ci = norm.ppf(upper_ci)
       Fx       = norm.ppf(Fx)
 
       maxCI = max(upper_ci)
       minCI = min(lower_ci)
       norm_F_val = np.arange( np.floor(minCI) , np.ceil(maxCI)+1 )
       norm_F_val = norm_F_val[ (-4 < norm_F_val) & (norm_F_val < 4) ]
       norm_F_str = np.char.mod('%6.4f',  norm.cdf(norm_F_val))
 
       #print(f' minCI = {minCI} ')  # debug
       #print(f' maxCI = {maxCI} ')  # debug
       #print(f' norm_F_val = {norm_F_val}')
       #print(f' norm_F_str = {norm_F_str}')

    plt.fill_between(sorted_data, lower_ci, upper_ci, 
                     color=[0.8, 0.9, 1.0], alpha=0.5,
                     label=f'{confidence_level}% confidence band')
    plt.step(sorted_data, Fx, '-', color=[0.2, 0.4, 0.8], 
             linewidth=2, markersize=4, label='Empirical CDF')
    # plot other CI's 
    '''
    plt.plot(sorted_data, upper_ci, '-r')
    plt.plot(sorted_data, lower_ci, '-r')
    plt.step(sorted_data, upper_ci_D, '-r' ) 
    plt.step(sorted_data, lower_ci_D, '-r' ) 
    plt.plot(sorted_data, upper_ci_DKW, '-g')
    plt.plot(sorted_data, lower_ci_DKW, '-g')
    '''
    # plot another CDF
    '''
    plt.plot(x_theory, cdf_theory, 'r-', linewidth=2, 
             label=f'Normal CDF (μ={mu:.3f}, σ={std:.3f})')
    plt.plot(sorted_data, sorted_data * 0 + 0.5, '--k', alpha=0.3, linewidth=1)
    '''
    
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Cumulative probability', fontsize=16)
    if norm_inv_CDF == True:
        if maxCI > 3.5 or minCI < -3.5:
            plt.ylabel(r'$\Phi^{-1}$( Cumulative probability )', fontsize=16)
        else: 
            plt.ylabel('Cumulative probability', fontsize=16)
            plt.yticks(norm_F_val, norm_F_str)
    plt.title(f'Empirical CDF with {confidence_level}% Confidence Intervals', 
              fontsize=16)
    plt.legend(fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    if norm_inv_CDF == False:
        plt.ylim([0, 1])
    
    plt.tight_layout()

    # Display plots
    if not interactive:
        plt.show()

    # Save plots to .pdf
    if pdf_plots:
        filename = f'plot_CDF_ci-{fig_num:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")

    if interactive: 
        input("Press Enter to close all figures...")
        plt.close('all')

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

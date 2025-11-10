import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_CDF_ci(data, confidence_level, figNo):
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
    """
    
    data = np.asarray(data).flatten()
    n = len(data)
    
    # Sort data
    sorted_data = np.sort(data)
    
    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    
    # Confidence intervals using Dvoretzky-Kiefer-Wolfowitz inequality
    alpha = 1 - confidence_level / 100
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))
    
    upper_ci = np.minimum(ecdf + epsilon, 1.0)
    lower_ci = np.maximum(ecdf - epsilon, 0.0)
    
    # Theoretical normal CDF
    mu, std = np.mean(data), np.std(data)
    x_theory = np.linspace(min(sorted_data), max(sorted_data), 200)
    cdf_theory = norm.cdf(x_theory, mu, std)
    
    # Plotting
    fig = plt.figure(figNo, figsize=(10, 7))
    fig.clf()
    
    plt.fill_between(sorted_data, lower_ci, upper_ci, 
                     color=[0.8, 0.9, 1.0], alpha=0.5,
                     label=f'{confidence_level}% confidence band')
    plt.plot(sorted_data, ecdf, 'o-', color=[0.2, 0.4, 0.8], 
             linewidth=2, markersize=4, label='Empirical CDF')
    plt.plot(x_theory, cdf_theory, 'r-', linewidth=2, 
             label=f'Normal CDF (μ={mu:.3f}, σ={std:.3f})')
    plt.plot(sorted_data, sorted_data * 0 + 0.5, '--k', 
             alpha=0.3, linewidth=1)
    
    plt.xlabel('Residual value', fontsize=13)
    plt.ylabel('Cumulative probability', fontsize=13)
    plt.title(f'Empirical CDF with {confidence_level}% Confidence Intervals', 
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()



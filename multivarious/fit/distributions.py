
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, laplace, rayleigh, gamma, gumbel_r, chi2, uniform, beta
from scipy.optimize import minimize
from collections import namedtuple

def fit_distributions(data):
    n = len(data)
    x_avg = np.mean(data)
    x_var = np.var(data, ddof=1)
    x_sdv = np.sqrt(x_var)
    x_cov = x_sdv / x_avg

    # Normal distribution
    normal_mu = x_avg
    normal_sigma = x_sdv * (1 + 1 / (4 * (n - 1)))

    # Lognormal distribution
    lognormal_mu = np.log(x_avg) - 0.5 * np.log(1 + x_cov ** 2)
    lognormal_sigma = np.sqrt(np.log(1 + x_cov ** 2))

    # Exponential distribution
    exponential_lambda = (n - 1) / (n * x_avg)

    # Laplace distribution
    laplace_mu = np.median(data)
    laplace_b = np.mean(np.abs(data - laplace_mu)) * (1 + 5 / (8 * n))

    # Rayleigh distribution
    rayleigh_sigma = np.sqrt(np.sum(data ** 2) / (2 * n)) * (1 + 1 / (8 * n))

    # Gamma distribution
    def gamma_nll(params):
        alpha, beta = params
        return -np.sum(gamma.logpdf(data, alpha, scale=1/beta))

    gamma_res = minimize(gamma_nll, x0=[1, 1], method='Nelder-Mead')
    gamma_alpha, gamma_beta = gamma_res.x
    gamma_alpha *= (1 - 2 / (n - 1))
    gamma_beta *= (1 - 2 / (n - 1) - 4 / (gamma_alpha * n))

    # Gumbel distribution
    gumbel_mu = x_avg - np.euler_gamma * x_sdv * np.sqrt(6) / np.pi
    gumbel_beta = x_sdv * np.sqrt(6) / np.pi * (1 + 3 / (10 * n) + 1 / (4 * (n - 1)))

    # Chi-squared distribution
    chi2_df = 2 * x_avg

    # Uniform distribution
    uniform_a = np.min(data) - (np.max(data) - np.min(data)) / (n - 1)
    uniform_b = np.max(data) + (np.max(data) - np.min(data)) / (n - 1)


 # Beta distribution (4-parameter)
    def beta_mle_equations(params):
        alpha, beta = params
        eq1 = digamma(alpha) - digamma(alpha + beta) - np.mean(np.log(data))
        eq2 = digamma(beta) - digamma(alpha + beta) - np.mean(np.log(1 - data))
        return np.array([eq1, eq2])

    beta_mle_res = opt.fsolve(beta_mle_equations, x0=[1, 1])
    beta_mle_alpha, beta_mle_beta = beta_mle_res

    # Cox-Snell bias correction for MLE estimates
    beta_alpha_tilde = beta_mle_alpha - (beta_mle_alpha - 1) / n + (2 * beta_mle_alpha - 1) / (n * (beta_mle_alpha + beta_mle_beta - 2))
    beta_beta_tilde = beta_mle_beta - (beta_mle_beta - 1) / n + (2 * beta_mle_beta - 1) / (n * (beta_mle_alpha + beta_mle_beta - 2))

    # Unbiased estimates for a and b
    beta_a_tilde = np.min(data) - beta_beta_tilde / beta_alpha_tilde * x_var / (x_avg * (1 - x_avg)) / (n + 1)
    beta_b_tilde = np.max(data) + beta_beta_tilde / beta_alpha_tilde * x_var / (x_avg * (1 - x_avg)) / (n + 1)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data, (np.arange(1, n+1) - 0.5) / n, 'ko', label='Data')

    x = np.linspace(np.min(data), np.max(data), 100)
    # ... (previous distribution plotting code remains the same) ...
    ax.plot(x, beta.cdf((x - beta_a_tilde) / (beta_b_tilde - beta_a_tilde), beta_alpha_tilde, beta_beta_tilde), 'r--', label='Beta')

    ax.set_xlabel('Data')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Fitted Distributions')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Tabular output
    tabular_output = f"Distribution | Parameters\n"
    # ... (previous tabular output code remains the same) ...
    tabular_output += f"Beta        | a = {beta_a_tilde:.3f}, b = {beta_b_tilde:.3f}, p = {beta_alpha_tilde:.3f}, q = {beta_beta_tilde:.3f}\n"

    # Return results as a namedtuple
    Results = namedtuple('Results', ['normal', 'lognormal', 'exponential', 'laplace', 'rayleigh',
                                     'gamma', 'gumbel', 'chi2', 'uniform', 'beta', 'tabular_output'])
    results = Results(
        normal=(normal_mu, normal_sigma),
        lognormal=(lognormal_mu, lognormal_sigma),
        exponential=exponential_lambda,
        laplace=(laplace_mu, laplace_b),
        rayleigh=rayleigh_sigma,
        gamma=(gamma_alpha, gamma_beta),
        gumbel=(gumbel_mu, gumbel_beta),
        chi2=chi2_df,
        uniform=(uniform_a, uniform_b),
        beta=(beta_a_tilde, beta_b_tilde, beta_alpha_tilde, beta_beta_tilde),
        tabular_output=tabular_output
    )

    return results

# distributions.py
# compute unbiased parameter estimates for various probability distributions
# to fit the distributions to a sample of data
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.special import digamma

from multivarious.rvs import beta, chi2, exponential, extreme_value_I, gamma, laplace, lognormal, normal, rayleigh, uniform
from multivarious.opt import fsolve

def distributions(data):
    """
    Fit various probability distributions to a given dataset and compute unbiased parameter estimates.

    Parameters:
    -----------
    data : numpy.ndarray
        One-dimensional array of sample data.

    Returns:
    --------
    coeffs : namedtuple
        Namedtuple containing the unbiased parameter estimates for each distribution.
        Fields: uniform, beta, exponential, laplace, normal, lognormal, rayleigh, gamma, extI, chi2.

    criteria : namedtuple
        Namedtuple containing the assessment metrics (log-likelihood, AIC, BIC) for each distribution.
        Fields: uniform, beta, exponential, laplace, normal, lognormal, rayleigh, gamma, extI, chi2.

    tabular_output : str
        Formatted string containing a tabular summary of the unbiased parameter estimates and assessment metrics.

    Raises:
    -------
    TypeError
        If 'data' is not a NumPy array.
    ValueError
        If 'data' is not one-dimensional, contains less than two elements, or contains non-finite values.

    Examples:
    ---------
    >>> data = np.random.normal(10, 3, 100)
    >>> coeffs, criteria, tabular_output = distributions(data)
    >>> print(tabular_output)
    """

    # Convert data to a NumPy array
    data = np.asarray(data)

    # Check if data is one-dimensional
    if data.ndim != 1:
        # Try to squeeze the array if it has only one dimension
        data = np.squeeze(data)
        if data.ndim != 1:
            raise ValueError("Input 'data' must be a one-dimensional array.")

    # Check if data contains only finite values
    if not np.isfinite(data).all():
        raise ValueError("Input 'data' must contain only finite values.")

    # Check if data contains negative values
    if np.any(data <= 0):
        raise ValueError("Input 'data' has negative values.  Some distributions are not appropriate.") 

    n = len(data)
    x_avg = np.mean(data)
    x_var = np.var(data, ddof=1)
    x_sdv = np.std(data, ddof=1)
    x_cov = x_sdv / x_avg


    # Uniform distribution unbiased estimates -----------------------------
    uniform_a_ube = np.min(data) - (np.max(data) - np.min(data)) / (n - 1)
    uniform_b_ube = np.max(data) + (np.max(data) - np.min(data)) / (n - 1)

    # assessment metrics
    n_param = 2
    uniform_ll = np.sum(np.log(uniform.pdf(data, uniform_a_ube, uniform_b_ube)))
    uniform_aic = 2 * n_param - 2 * uniform_ll
    uniform_bic = n_param * np.log(len(data)) -  2 * uniform_ll


    # Beta distribution (4-parameter) -----------------------------------
    def beta_mle(params, data_01):
        q, p = params
        # gradient of log likelihood w.r.t. alpha and bta 
        eq1 = digamma(q) - digamma(q + p) - np.mean(np.log(data_01))
        eq2 = digamma(p) - digamma(q + p) - np.mean(np.log(1 - data_01)) 
        return ( eq1, eq2 ), -1.0

    # scale data to the range [ 0.001 <= data <= 0.999 ]
    data_01 = 0.001 + 0.998 * (data - np.min(data)) / (np.max(data) - np.min(data))
    beta_qp_mle, _, _, _, _ = fsolve(beta_mle, [1, 1], data_01)

    beta_q_mle, beta_p_mle = beta_qp_mle

    # Cox-Snell bias correction for MLE estimates of q and p
    beta_q_ube = beta_q_mle - (beta_q_mle - 1) / n + (2 * beta_q_mle - 1) / (n * (beta_q_mle + beta_q_mle - 2))
    beta_p_ube = beta_p_mle - (beta_p_mle - 1) / n + (2 * beta_p_mle - 1) / (n * (beta_p_mle + beta_p_mle - 2))

    # Unbiased estimates (ube) for a and b
    beta_a_ube = np.min(data) - beta_p_ube / beta_q_ube * x_var / (x_avg * (1 - x_avg)) / (n + 1)
    beta_b_ube = np.max(data) + beta_p_ube / beta_q_ube * x_var / (x_avg * (1 - x_avg)) / (n + 1)

    # assessment metrics
    n_param = 4
    data_ll = data
    data_ll[ data <= beta_a_ube ] = beta_a_ube + 1e-3 
    data_ll[ data >= beta_b_ube ] = beta_b_ube - 1e-3 
    beta_ll = np.sum(np.log(beta.pdf(data_ll, beta_a_ube, beta_b_ube, beta_q_ube, beta_p_ube)))
    beta_aic = 2 * n_param - 2 * beta_ll
    beta_bic = n_param * np.log(len(data_ll)) - 2 * beta_ll 


    # Exponential distribution ------------------------------------
    exponential_mean_ube = x_avg * n / (n - 1) 

    # assessment metrics
    n_param = 1
    exponential_ll = np.sum(np.log(exponential.pdf(data, exponential_mean_ube))) 
    exponential_aic = 2 * n_param - 2 * exponential_ll
    exponential_bic = n_param * np.log(len(data)) - 2 * exponential_ll


    # Laplace distribution ----------------------------------------- 
    # (scipy does not agree with wikipedia definition of laplace pdf)
    laplace_mean_ube = np.median(data)
    laplace_stdv_ube = np.sqrt(2)*np.mean(np.abs(data - laplace_mean_ube)) * (1 + 5 / (8 * n))

    # assessment metrics
    n_param = 2
    laplace_ll = np.sum(np.log(laplace.pdf(data, laplace_mean_ube, laplace_stdv_ube))) 
    laplace_aic = 2 * n_param - 2 * laplace_ll
    laplace_bic = n_param * np.log(len(data)) - 2 * laplace_ll


    # Normal distribution ------------------------------------
    normal_mean_ube = x_avg
    normal_stdv_ube = x_sdv * (1 + 1 / (4 * (n - 1)))

    # assessment metrics
    n_param = 2
    normal_ll = np.sum(np.log(normal.pdf(data, normal_mean_ube, normal_stdv_ube))) 
    normal_aic = 2 * n_param - 2 * normal_ll
    normal_bic = n_param * np.log(len(data)) - 2 * normal_ll


    # Lognormal distribution -----------------------------------
    # The sample mean and variance of the logarithm of the data ...
    log_data = np.log(data)        # log(x)
    m = np.mean(log_data)          # mean of log(x)
    s2 = np.var(log_data, ddof=1)  # variance of log(x), ddof=1 for the unbiased sample variance

    # Bias-corrected estimators for m and s^2 (s2) 
    m_ube  = m - s2 / (2 * n)    # corrected mean of log(x)
    s2_ube = s2 * (n / (n - 1))  # corrected variance of log(x)

    # bias-corrected estimator for mean and coefficient of variation
    lognormal_mean_ube = np.exp( m_ube + 0.5*s2_ube) 
    lognormal_covn_ube = np.sqrt( np.exp(s2_ube) - 1.0 )

    # assessment metrics
    n_param = 2
    lognormal_ll = np.sum(np.log(lognormal.pdf(data, lognormal_mean_ube, lognormal_covn_ube))) 
    lognormal_aic = 2 * n_param - 2 * lognormal_ll
    lognormal_bic = n_param * np.log(len(data)) - 2 * lognormal_ll


    # Rayleigh distribution ----------------------------------------
    rayleigh_mean_ube = x_avg * (1 + 1 / (4 * n))

    # assessment metrics
    n_param = 2
    rayleigh_ll = np.sum(np.log(rayleigh.pdf(data, rayleigh_mean_ube))) 
    rayleigh_aic = 2 * n_param - 2 * rayleigh_ll
    rayleigh_bic = n_param * np.log(len(data)) - 2 * rayleigh_ll

    # Gamma distribution --------------------------------------------
    def gamma_mle(params, data):
        k, bta = params   # shape, scale
        # gradient of log likelihood w.r.t. k and bta 
        n = len(data)
        eq1 = n * np.log(bta) - n * digamma(k) + np.sum(np.log(data)) 
        eq2 = n * k / bta - np.sum(data) 
        return ( eq1, eq2 ), -1.0

    # Minimize the negative log-likelihood
    #              k          bta
    v_init = [  1.0/x_cov,  1.0/x_sdv ]  # Initial guess for the parameters
    # --- not yet working ---- 
    #gamma_mle_kb, _, _, _, _ = fsolve(gamma_mle, v_init, data)
    # --- use v_init instead of gamma_mle_kb ---
    gamma_k, gamma_bta = v_init # gamma_mle_kb # v_init 

    # unbiased estimator
    #gamma_k  *= (1 - 2 / (n - 1))
    gamma_k_ube   = gamma_k - (gamma_k - 1) / n 
    gamma_bta_ube = gamma_bta * (1 - 2 / (n - 1) - 4 / (gamma_k_ube * n))

    gamma_mean_ube = gamma_k_ube / gamma_bta_ube
    gamma_covn_ube = 1.0 / gamma_k_ube

    # assessment metrics
    n_param = 2
    gamma_ll = np.sum(np.log(gamma.pdf(data, gamma_mean_ube, gamma_covn_ube))) 
    gamma_aic = 2 * n_param - 2 * gamma_ll
    gamma_bic = n_param * np.log(len(data)) - 2 * gamma_ll

    # Extreme type I distribution ----------------------------------------
    def extI_mle(params, data):
        mm, ss = params
        n = len(data)
        exp_z = np.exp(-(data - mm) / ss)
        eq1 = n - np.sum(exp_z)
        eq2 = np.sum((data - mm) * (1 - exp_z)) - n * ss
        return [eq1, eq2], -1.0

    v_init = [x_avg, x_sdv]
    extI_mle_params, _, _, _, _ = fsolve(extI_mle, v_init, data)
    extI_mm, extI_ss = extI_mle_params 

    # Cox Snell corrections to ExtI distribution parameters mm and ss
    extI_mm_ube = extI_mm - extI_ss * (np.euler_gamma  / n)
    extI_ss_ube = extI_ss * ( 1 + (1 + np.euler_gamma) / n)

    # convert to distribution mean and standard deviation 
    extI_mean_ube = extI_mm_ube + np.euler_gamma * extI_ss_ube 
    extI_stdv_ube = extI_ss_ube * np.pi / np.sqrt(6.0)
    extI_covn_ube = extI_stdv_ube / extI_mean_ube * ( 1 + 1/(4*n) )


    # assessment metrics
    n_param = 2
    extI_ll = np.sum(np.log(extreme_value_I.pdf(data, extI_mean_ube, extI_stdv_ube/extI_mean_ube))) 
    extI_aic = 2 * n_param - 2 * extI_ll
    extI_bic = n_param * np.log(len(data)) - 2 * extI_ll


    # Chi-squared distribution -----------------------------------------
    chi2_df = x_avg

    # Wilson-Hilferty bias correction 
    chi2_df_ube = chi2_df * (1 - 2 / (9 * chi2_df))**3

    # assessment metrics
    n_param = 1
    chi2_ll = np.sum(np.log(chi2.pdf(data, chi2_df_ube))) # log likelihood
    chi2_aic = 2 * n_param - 2 * chi2_ll
    chi2_bic = n_param * np.log(len(data)) - 2 * chi2_ll

    # Tabular output ==========================================================
    tabular_output  = f"Distribution | Parameter Unbiased Estimates       LL       AIC     BIC \n"
    tabular_output += f"Uniform      | a = {uniform_a_ube:7.3f}, b = {uniform_b_ube:7.3f},        {uniform_ll:7.3f} {uniform_aic:7.3f} {uniform_bic:7.3f}\n"
    tabular_output += f"Beta         | a = {beta_a_ube:7.3f}, b = {beta_b_ube:7.3f}, p = {beta_q_ube:7.3f}, q = {beta_p_ube:7.3f}, {beta_ll:7.3f} {beta_aic:7.3f} {beta_bic:7.3f} \n"
    tabular_output += f"Exponential  | mean = {exponential_mean_ube:7.3f}                   {exponential_ll:7.3f} {exponential_aic:7.3f} {exponential_bic:7.3f}\n"
    tabular_output += f"Laplace      | mean = {laplace_mean_ube:7.3f}, covn = {laplace_stdv_ube/laplace_mean_ube:7.3f}   {laplace_ll:7.3f} {laplace_aic:7.3f} {laplace_bic:7.3f}\n"
    tabular_output += f"Normal       | mean = {normal_mean_ube:7.3f}, covn = {normal_stdv_ube/normal_mean_ube:7.3f}   {normal_ll:7.3f} {normal_aic:7.3f} {normal_bic:7.3f}\n"
    tabular_output += f"Lognormal    | mean = {lognormal_mean_ube:7.3f}, covn = {lognormal_covn_ube:7.3f}   {lognormal_ll:7.3f} {lognormal_aic:7.3f} {lognormal_bic:7.3f}\n"
    tabular_output += f"Rayleigh     | mean = {rayleigh_mean_ube:7.3f}                   {rayleigh_ll:7.3f} {rayleigh_aic:7.3f} {rayleigh_bic:7.3f}\n"
    tabular_output += f"Gamma        | mean = {gamma_mean_ube:7.3f}, covn = {gamma_covn_ube:7.3f}   {gamma_ll:7.3f} {gamma_aic:7.3f} {gamma_bic:7.3f}\n"
    tabular_output += f"Extreme I    | mean = {extI_mean_ube:7.3f}, covn = {extI_stdv_ube/extI_mean_ube:7.3f}   {extI_ll:7.3f} {extI_aic:7.3f} {extI_bic:7.3f}\n"
    tabular_output += f"Chi Squared  | mean = {chi2_df_ube:7.3f}                   {chi2_ll:7.3f} {chi2_aic:7.3f} {chi2_bic:7.3f}\n"


    # Return results as a namedtuple =============================================================
    Results = namedtuple('Results', ['uniform', 'beta', 'exponential', 'laplace',
                                     'normal', 'lognormal', 'rayleigh', 'gamma',
                                     'extI', 'chi2'])

    coeffs = Results(
        uniform=(uniform_a_ube, uniform_b_ube), 
        beta=(beta_a_ube, beta_b_ube, beta_q_ube, beta_p_ube), 
        exponential=(exponential_mean_ube), 
        laplace=(laplace_mean_ube, laplace_stdv_ube), 
        normal=(normal_mean_ube, normal_stdv_ube), 
        lognormal=(lognormal_mean_ube, lognormal_covn_ube), 
        rayleigh=(rayleigh_mean_ube), 
        gamma=(gamma_mean_ube, gamma_covn_ube), 
        extI=(extI_mean_ube, extI_covn_ube), 
        chi2=(chi2_df_ube)
    )

    criteria = Results(
        uniform=(uniform_ll, uniform_aic, uniform_bic),
        beta=(beta_ll, beta_aic, beta_bic),
        exponential=(exponential_ll, exponential_aic, exponential_bic), 
        laplace=(laplace_ll, laplace_aic, laplace_bic),
        normal=(normal_ll, normal_aic, normal_bic),
        lognormal=(lognormal_ll, lognormal_aic, lognormal_bic),
        rayleigh=(rayleigh_ll, rayleigh_aic, rayleigh_bic),
        gamma=(gamma_ll, gamma_aic, gamma_bic),
        extI=(extI_ll, extI_aic, extI_bic),
        chi2=(chi2_ll, chi2_aic, chi2_bic)
    )


    # Plots ============================================================

    # Subplots for the histogram and CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the histogram of the combined data
    ax1.hist(data, bins=30, density=True, alpha=0.7, color='gray')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Probbility Density')
    ax1.set_title('Histogram of Combined Data')

    # Plot the empirical CDF and fitted distribution CDFs
    x = np.linspace(np.min(data) + 1e-3, np.max(data) - 1e-3, 100)

    ax2.plot(x, uniform.cdf(x, coeffs.uniform), 'g--', label='Uniform')
    ax2.plot(x, beta.cdf(x,  coeffs.beta), 'r--', label='Beta')
    ax2.plot(x, exponential.cdf(x, coeffs.exponential), 'r-', label='Exponential')
    ax2.plot(x, laplace.cdf( x, coeffs.laplace ), 'c-', label='Laplace')
    ax2.plot(x, normal.cdf(x, coeffs.normal), 'b-', label='Normal')
    ax2.plot(x, lognormal.cdf(x, coeffs.lognormal), 'g-', label='Lognormal')
    ax2.plot(x, rayleigh.cdf(x, coeffs.rayleigh), 'm-', label='Rayleigh')
    ax2.plot(x, gamma.cdf(x, coeffs.gamma), 'y-', label='Gamma')
    ax2.plot(x, extreme_value_I.cdf(x, coeffs.extI), 'k-', label='Extreme I')
    ax2.plot(x, chi2.cdf(x, coeffs.chi2), 'b--', label='Chi-squared')
    ax2.plot(np.sort(data), (np.arange(1, n+1)) / (n+1), 'ok', label='Empirical CDF')

    ax2.set_xlabel('Data')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Fitted Distributions')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    return coeffs, criteria, tabular_output

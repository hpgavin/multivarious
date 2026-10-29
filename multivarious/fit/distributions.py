# distributions.py
# compute unbiased parameter estimates for various probability distributions
# to fit the distributions to a sample of data

# Voinov and Nikulin, Unbiased Estimators and Their Applications
# (Vol. 1: Univariate, 1993; Vol. 2: Multivariate, 1996)
# https://archive.org/details/unbiasedestimato0002voin
#
# D.R. Cox and E.J. Snell, A General Description of Residuals,
# Proc Royal Statistical Society , Series B (Methodological)
# Vol. 30, No 2 (1968) pp 248 - 275
# https://www.jstor.org/stable/2984505

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from multivarious.rvs import beta, chi2, exponential, extreme_value_I, gamma, laplace, lognormal, normal, rayleigh, uniform
from multivarious.opt import fsolve
from collections import namedtuple

def initialize_params():
    Parameters = namedtuple('Parameters', ['uniform', 'beta', 'exponential', 'laplace', 'normal', 'lognormal', 'rayleigh', 'gamma', 'extreme_value_I', 'chi2'])
    
    params = Parameters(
        uniform=(np.nan, np.nan),
        beta=(np.nan, np.nan, np.nan, np.nan),
        exponential=(np.nan,),
        laplace=(np.nan, np.nan),
        normal=(np.nan, np.nan),
        lognormal=(np.nan, np.nan),
        rayleigh=(np.nan,),
        gamma=(np.nan, np.nan),
        extreme_value_I=(np.nan, np.nan),
        chi2=(np.nan,)
    )
    
    criteria = Parameters(
        uniform=(np.nan, np.nan, np.nan),
        beta=(np.nan, np.nan, np.nan),
        exponential=(np.nan, np.nan, np.nan),
        laplace=(np.nan, np.nan, np.nan),
        normal=(np.nan, np.nan, np.nan),
        lognormal=(np.nan, np.nan, np.nan),
        rayleigh=(np.nan, np.nan, np.nan),
        gamma=(np.nan, np.nan, np.nan),
        extreme_value_I=(np.nan, np.nan, np.nan),
        chi2=(np.nan, np.nan, np.nan)
    )
    
    return params, criteria

def fit_uniform(x):
    uniform_a_ube = x.min - (x.max - x.min) / (len(x.data) - 1)
    uniform_b_ube = x.max + (x.max - x.min) / (len(x.data) - 1)

    n_param = 2
    uniform_ll = np.sum(np.log(uniform.pdf(x.data, uniform_a_ube, uniform_b_ube)))
    uniform_aic = 2 * n_param - 2 * uniform_ll
    uniform_bic = n_param * np.log(len(x.data)) - 2 * uniform_ll

    return {'a': uniform_a_ube, 'b': uniform_b_ube}, {'ll': uniform_ll, 'aic': uniform_aic, 'bic': uniform_bic}

def fit_beta(x):
    def beta_mle(params, data_01):
        q, p = params
        eq1 = digamma(q) - digamma(q + p) - np.mean(np.log(data_01))
        eq2 = digamma(p) - digamma(q + p) - np.mean(np.log(1 - data_01))
        return (eq1, eq2), -1.0

    data_01 = 0.001 + 0.998 * (x.data - x.min) / (x.max - x.min)
    beta_qp_mle, _, _, _, _ = fsolve(beta_mle, [1, 1], data_01)
    beta_q_mle, beta_p_mle = beta_qp_mle

    n = len(x.data)
    beta_q_ube = beta_q_mle - (beta_q_mle - 1) / n + (2 * beta_q_mle - 1) / (n * (beta_q_mle + beta_p_mle - 2))
    beta_p_ube = beta_p_mle - (beta_p_mle - 1) / n + (2 * beta_p_mle - 1) / (n * (beta_q_mle + beta_p_mle - 2))

    beta_a_ube = x.min - beta_p_ube / beta_q_ube * x.var / (x.avg * (1 - x.avg)) / (n + 1)
    beta_b_ube = x.max + beta_p_ube / beta_q_ube * x.var / (x.avg * (1 - x.avg)) / (n + 1)

    n_param = 4
    data_ll = x.data
    data_ll[data_ll <= beta_a_ube] = beta_a_ube + 1e-3
    data_ll[data_ll >= beta_b_ube] = beta_b_ube - 1e-3
    beta_ll = np.sum(np.log(beta.pdf(data_ll, beta_a_ube, beta_b_ube, beta_q_ube, beta_p_ube)))
    beta_aic = 2 * n_param - 2 * beta_ll
    beta_bic = n_param * np.log(len(data_ll)) - 2 * beta_ll

    return {'a': beta_a_ube, 'b': beta_b_ube, 'q': beta_q_ube, 'p': beta_p_ube}, {'ll': beta_ll, 'aic': beta_aic, 'bic': beta_bic}

def fit_exponential(x):
    exponential_mean_ube = x.avg * len(x.data) / (len(x.data) - 1)

    n_param = 1
    exponential_ll = np.sum(np.log(exponential.pdf(x.data, exponential_mean_ube)))
    exponential_aic = 2 * n_param - 2 * exponential_ll
    exponential_bic = n_param * np.log(len(x.data)) - 2 * exponential_ll

    return {'mean': exponential_mean_ube}, {'ll': exponential_ll, 'aic': exponential_aic, 'bic': exponential_bic}

def fit_laplace(x):
    laplace_mean_ube = np.median(x.data)
    laplace_sdvn_ube = np.sqrt(2) * np.mean(np.abs(x.data - laplace_mean_ube)) * (1 + 5 / (8 * len(x.data)))

    n_param = 2
    laplace_ll = np.sum(np.log(laplace.pdf(x.data, laplace_mean_ube, laplace_sdvn_ube)))
    laplace_aic = 2 * n_param - 2 * laplace_ll
    laplace_bic = n_param * np.log(len(x.data)) - 2 * laplace_ll

    return {'mean': laplace_mean_ube, 'sdvn': laplace_sdvn_ube}, {'ll': laplace_ll, 'aic': laplace_aic, 'bic': laplace_bic}

def fit_normal(x):
    normal_mean_ube = x.avg
    normal_sdvn_ube = x.sdv

    n_param = 2
    normal_ll = np.sum(np.log(normal.pdf(x.data, normal_mean_ube, normal_sdvn_ube)))
    normal_aic = 2 * n_param - 2 * normal_ll
    normal_bic = n_param * np.log(len(x.data)) - 2 * normal_ll

    return {'mean': normal_mean_ube, 'sdvn': normal_sdvn_ube}, {'ll': normal_ll, 'aic': normal_aic, 'bic': normal_bic}

def fit_lognormal(x):
    log_data = np.log(x.data)
    m = np.mean(log_data)
    s2 = np.var(log_data, ddof=1)

    m_ube = m - s2 / (2 * len(x.data))
    s2_ube = s2 * (len(x.data) / (len(x.data) - 1))

    lognormal_medn_ube = np.exp(m_ube)
    lognormal_covn_ube = np.sqrt(np.exp(s2_ube) - 1.0)

    n_param = 2
    lognormal_ll = np.sum(np.log(lognormal.pdf(x.data, lognormal_medn_ube, lognormal_covn_ube)))
    lognormal_aic = 2 * n_param - 2 * lognormal_ll
    lognormal_bic = n_param * np.log(len(x.data)) - 2 * lognormal_ll

    return {'medn': lognormal_medn_ube, 'covn': lognormal_covn_ube}, {'ll': lognormal_ll, 'aic': lognormal_aic, 'bic': lognormal_bic}

def fit_rayleigh(x):
    rayleigh_mean_ube = x.avg * (1 + 1 / (4 * len(x.data)))

    n_param = 2
    rayleigh_ll = np.sum(np.log(rayleigh.pdf(x.data, rayleigh_mean_ube)))
    rayleigh_aic = 2 * n_param - 2 * rayleigh_ll
    rayleigh_bic = n_param * np.log(len(x.data)) - 2 * rayleigh_ll

    return {'mean': rayleigh_mean_ube}, {'ll': rayleigh_ll, 'aic': rayleigh_aic, 'bic': rayleigh_bic}

def fit_gamma(x):
    def gamma_mle(params, data):
        k, bta = params
        n = len(data)
        eq1 = n * np.log(bta) - n * digamma(k) + np.sum(np.log(data))
        eq2 = n * k / bta - np.sum(data)
        return (eq1, eq2), -1.0

    v_init = [1.0 / x.cov, 1.0 / x.sdv]
    gamma_mle_kb, _, _, _, _ = fsolve(gamma_mle, v_init, x.data)
    gamma_k, gamma_bta = gamma_mle_kb

    gamma_k_ube = gamma_k - (gamma_k - 1) / len(x.data)
    gamma_bta_ube = gamma_bta * (1 - 2 / (len(x.data) - 1) - 4 / (gamma_k_ube * len(x.data)))

    gamma_mean_ube = gamma_k_ube / gamma_bta_ube
    gamma_covn_ube = 1.0 / gamma_k_ube

    n_param = 2
    gamma_ll = np.sum(np.log(gamma.pdf(x.data, gamma_mean_ube, gamma_covn_ube)))
    gamma_aic = 2 * n_param - 2 * gamma_ll
    gamma_bic = n_param * np.log(len(x.data)) - 2 * gamma_ll

    return {'mean': gamma_mean_ube, 'covn': gamma_covn_ube}, {'ll': gamma_ll, 'aic': gamma_aic, 'bic': gamma_bic}

def fit_extreme_value_I(x):
    def extreme_value_I_mle(params, data):
        mm, ss = params
        n = len(data)
        exp_z = np.exp(-(data - mm) / ss)
        eq1 = n - np.sum(exp_z)
        eq2 = np.sum((data - mm) * (1 - exp_z)) - n * ss
        return [eq1, eq2], -1.0

    v_init = [x.avg, x.sdv]
    extreme_value_I_mle_params, _, _, _, _ = fsolve(extreme_value_I_mle, v_init, x.data)
    extreme_value_I_mm, extreme_value_I_ss = extreme_value_I_mle_params

    extreme_value_I_mm_ube = extreme_value_I_mm - extreme_value_I_ss * (np.euler_gamma / len(x.data))
    extreme_value_I_ss_ube = extreme_value_I_ss * (1 + (1 + np.euler_gamma) / len(x.data))

    extreme_value_I_mean_ube = extreme_value_I_mm_ube + np.euler_gamma * extreme_value_I_ss_ube
    extreme_value_I_sdvn_ube = extreme_value_I_ss_ube * np.pi / np.sqrt(6.0)
    extreme_value_I_covn_ube = extreme_value_I_sdvn_ube / extreme_value_I_mean_ube

    n_param = 2
    extreme_value_I_ll = np.sum(np.log(extreme_value_I.pdf(x.data, extreme_value_I_mean_ube, extreme_value_I_sdvn_ube / extreme_value_I_mean_ube)))
    extreme_value_I_aic = 2 * n_param - 2 * extreme_value_I_ll
    extreme_value_I_bic = n_param * np.log(len(x.data)) - 2 * extreme_value_I_ll

    return {'mean': extreme_value_I_mean_ube, 'covn': extreme_value_I_covn_ube}, {'ll': extreme_value_I_ll, 'aic': extreme_value_I_aic, 'bic': extreme_value_I_bic}

def fit_chi2(x):
    chi2_df = x.avg
    chi2_df_ube = chi2_df * (1 - 2 / (9 * chi2_df)) ** 3

    n_param = 1
    chi2_ll = np.sum(np.log(chi2.pdf(x.data, chi2_df_ube)))
    chi2_aic = 2 * n_param - 2 * chi2_ll
    chi2_bic = n_param * np.log(len(x.data)) - 2 * chi2_ll

    return {'df': chi2_df_ube}, {'ll': chi2_ll, 'aic': chi2_aic, 'bic': chi2_bic}

def distributions(data, top_models=3):
    data = np.asarray(data)

    if data.ndim != 1:
        data = np.squeeze(data)
        if data.ndim != 1:
            raise ValueError("Input 'data' must be a one-dimensional array.")

    if not np.isfinite(data).all():
        raise ValueError("Input 'data' must contain only finite values.")

    x = type('', (), {})()
    x.data = data
    x.min = np.min(data)
    x.max = np.max(data)
    x.avg = np.mean(data)
    x.var = np.var(data, ddof=1)
    x.sdv = np.sqrt(x.var) / (1 - 1 / (4 * (len(data) - 1)))
    x.cov = (x.sdv / x.avg) / (1 + (x.sdv / x.avg) ** 2.0 / len(data))

    params, criteria = initialize_params()

    positive_only_distributions = ['lognormal', 'rayleigh', 'gamma', 'chi2']
    all_distributions = ['uniform', 'beta', 'exponential', 'laplace', 'normal', 'lognormal', 'rayleigh', 'gamma', 'extreme_value_I', 'chi2']

    if np.any(data <= 0):
        distributions_to_fit = [dist for dist in all_distributions if dist not in positive_only_distributions]
    else:
        distributions_to_fit = all_distributions

    for dist in distributions_to_fit:
        fit_func = globals()[f'fit_{dist}']
        params_dict, criteria_dict = fit_func(x)
        params = params._replace(**{dist: tuple(params_dict.values())})
        criteria = criteria._replace(**{dist: tuple(criteria_dict.values())})

    sorted_criteria = sorted(criteria._asdict().items(), key=lambda x: x[1][1])  # Sort based on AIC
    top_distributions = [dist for dist, _ in sorted_criteria[:top_models]]

    tabular_output = "Distribution | Parameter Unbiased Estimates       LL       AIC     BIC\n"
    for dist in top_distributions:
        if dist == 'uniform':
            tabular_output += f"Uniform      | a = {params.uniform[0]:7.3f}, b = {params.uniform[1]:7.3f},        {criteria.uniform[0]:7.3f} {criteria.uniform[1]:7.3f} {criteria.uniform[2]:7.3f}\n"
        elif dist == 'beta':
            tabular_output += f"Beta         | a = {params.beta[0]:7.3f}, b = {params.beta[1]:7.3f}, p = {params.beta[2]:7.3f}, q = {params.beta[3]:7.3f}, {criteria.beta[0]:7.3f} {criteria.beta[1]:7.3f} {criteria.beta[2]:7.3f}\n"
        elif dist == 'exponential':
            tabular_output += f"Exponential  | mean = {params.exponential[0]:7.3f}                   {criteria.exponential[0]:7.3f} {criteria.exponential[1]:7.3f} {criteria.exponential[2]:7.3f}\n"
        elif dist == 'laplace':
            tabular_output += f"Laplace      | mean = {params.laplace[0]:7.3f}, covn = {params.laplace[1] / params.laplace[0]:7.3f}   {criteria.laplace[0]:7.3f} {criteria.laplace[1]:7.3f} {criteria.laplace[2]:7.3f}\n"
        elif dist == 'normal':
            tabular_output += f"Normal       | mean = {params.normal[0]:7.3f}, covn = {params.normal[1] / params.normal[0]:7.3f}   {criteria.normal[0]:7.3f} {criteria.normal[1]:7.3f} {criteria.normal[2]:7.3f}\n"
        elif dist == 'lognormal':
            tabular_output += f"Lognormal    | medn = {params.lognormal[0]:7.3f}, covn = {params.lognormal[1]:7.3f}   {criteria.lognormal[0]:7.3f} {criteria.lognormal[1]:7.3f} {criteria.lognormal[2]:7.3f}\n"
        elif dist == 'rayleigh':
            tabular_output += f"Rayleigh     | mean = {params.rayleigh[0]:7.3f}                   {criteria.rayleigh[0]:7.3f} {criteria.rayleigh[1]:7.3f} {criteria.rayleigh[2]:7.3f}\n"
        elif dist == 'gamma':
            tabular_output += f"Gamma        | mean = {params.gamma[0]:7.3f}, covn = {params.gamma[1]:7.3f}   {criteria.gamma[0]:7.3f} {criteria.gamma[1]:7.3f} {criteria.gamma[2]:7.3f}\n"
        elif dist == 'extreme_value_I':
            tabular_output += f"Extreme I    | mean = {params.extreme_value_I[0]:7.3f}, covn = {params.extreme_value_I[1]:7.3f}   {criteria.extreme_value_I[0]:7.3f} {criteria.extreme_value_I[1]:7.3f} {criteria.extreme_value_I[2]:7.3f}\n"
        elif dist == 'chi2':
            tabular_output += f"Chi Squared  | mean = {params.chi2[0]:7.3f}                   {criteria.chi2[0]:7.3f} {criteria.chi2[1]:7.3f} {criteria.chi2[2]:7.3f}\n"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.hist(data, bins=30, density=True, alpha=0.7, color='gray')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Histogram of Combined Data')

    x_plot = np.linspace(x.min + 1e-3, x.max - 1e-3, 100)

    for dist in top_distributions:
        ax2.plot(x_plot, globals()[dist].cdf(x_plot, getattr(params, dist)), label=dist.capitalize().replace('_', ' '))

    ax2.plot(np.sort(data), (np.arange(1, len(data) + 1)) / (len(data) + 1), 'ok', label='Empirical CDF')

    ax2.set_xlabel('Data')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Fitted Distributions')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    return params, criteria, tabular_output

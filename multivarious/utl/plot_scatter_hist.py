#! /usr/bin/env -S python3 -i

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    from multivarious.utl import format_plot
except ImportError:
    def format_plot(**kwargs):
        plt.rcParams.update({
            'lines.linewidth':  kwargs.get('line_width',  1.5),
            'font.size':        kwargs.get('font_size',   10),
            'lines.markersize': kwargs.get('marker_size', 4),
        })


def corr_ci(r, N, ci=0.95):
    '''
    Fisher z-transform confidence interval for a Pearson correlation
    coefficient.

    Apply the Fisher z-transformation
        z_r = arctanh(r) = 0.5 * ln((1+r)/(1-r))
    which is approximately normal with SE = 1/sqrt(N-3).  Back-transforming
    via tanh yields the asymmetric (1-alpha) CI for rho:
        tanh( z_r + z_{alpha/2}   / sqrt(N-3) ) < rho
                                                <= tanh( z_r + z_{1-alpha/2} / sqrt(N-3) )

    INPUTS      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    r           sample Pearson correlation coefficient                 scalar
    N           number of observations                                 scalar
    ci          confidence level, e.g. 0.95 for 95 % CI               scalar

    OUTPUTS     DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    lo          lower CI bound, clamped to [-1, 1]                     scalar
    hi          upper CI bound, clamped to [-1, 1]                     scalar
    '''
    alpha  = 1.0 - ci
    z_crit = norm.ppf(1.0 - alpha / 2.0)       # positive critical value z_{1-alpha/2}
    z_r    = np.arctanh(np.clip(r, -0.9999999, 0.9999999))
    se     = 1.0 / np.sqrt(N - 3)
    lo     = np.tanh(z_r - z_crit * se)
    hi     = np.tanh(z_r + z_crit * se)
    return float(np.clip(lo, -1.0, 1.0)), float(np.clip(hi, -1.0, 1.0))


def plot_scatter_hist(data, n_bins=20, figNo=100, var_names=None,
                      font_size=10, ci=0.95):
    '''
    Scatter plot matrix: histograms on the diagonal, scatter plots on the
    lower triangle, and Pearson correlation coefficients with Fisher
    confidence intervals on the upper triangle.

    INPUTS      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    data        matrix of data, one variable per row                   n x N
    n_bins      number of bins in the histogram                        1 x 1
    figNo       figure number for plotting (default = 100)             1 x 1
    var_names   optional dict with key 'X' containing a list of
                variable name strings for axis labels                  dict
    font_size   base font size for format_plot (default = 10)          1 x 1
    ci          confidence level for Fisher CI (default = 0.95)        1 x 1

    OUTPUTS     DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    fig         matplotlib Figure object
    '''
    n, N = data.shape

    if var_names is None:
        Names = [rf"$x_{{{i+1}}}$" for i in range(n)]
    else:
        Names = var_names.get('X', [rf"$x_{{{i+1}}}$" for i in range(n)])

    corr   = np.corrcoef(data)           # n x n
    ci_pct = int(round(ci * 100))

    format_plot(line_width=1.5, font_size=font_size, marker_size=4)

    plt.ion()
    fig = plt.figure(figNo, figsize=(2*n, 2*n))
    plt.clf()

    plotIndex = 1

    for iRow in range(n):
        for iCol in range(n):

            xData  = data[iCol, :]
            yData  = data[iRow, :]
            xLabel = Names[iCol]
            yLabel = Names[iRow]

            ax = plt.subplot(n, n, plotIndex)

            if iRow == iCol:
                # Diagonal: histogram
                ax.hist(xData, bins=n_bins, color='darkblue',
                        alpha=0.7, edgecolor='black')

            elif iRow < iCol:
                # Upper triangle: rho and Fisher CI
                r         = corr[iRow, iCol]
                lo, hi    = corr_ci(r, N, ci=ci)
                txt_color = 'darkred' if abs(r) > 0.5 else 'black'

                ax.text(0.5, 0.62,
                        rf"$\rho={r:+.3f}$",
                        ha='center', va='center',
                        fontsize=font_size + 2,
                        color=txt_color,
                        transform=ax.transAxes)
                ax.text(0.5, 0.35,
                        rf"$[{lo:+.3f},\ {hi:+.3f}]_{{{ci_pct}\%}}$",
                        ha='center', va='center',
                        fontsize=font_size - 1,
                        color='dimgray',
                        transform=ax.transAxes)

                ax.set_facecolor('#f5f5f5')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            else:
                # Lower triangle: scatter plot
                ax.plot(xData, yData, 'o', color='darkblue',
                        markersize=2, alpha=0.5)

            # Edge labels only
            if iRow == n - 1:
                ax.set_xlabel(xLabel, fontsize=font_size + 4)
                plt.setp(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])

            if iCol == 0:
                ax.set_ylabel(yLabel, fontsize=font_size + 4)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)
            plotIndex += 1

    plt.tight_layout()
    plt.show(block=False)

    return fig


def main():
    '''
    Test with 4 correlated Gaussian variables.
    Prints the Fisher CI for each pair and displays the scatter plot matrix.
    '''
    rng = np.random.default_rng(42)
    N   = 500

    # Cholesky factor giving controlled correlations:
    #   x1-x2: rho ~ +0.85   x3-x4: rho ~ -0.50   others: weak
    L = np.array([[1.00,  0.00,  0.00,  0.00],
                  [0.85,  0.53,  0.00,  0.00],
                  [0.20,  0.10,  0.97,  0.00],
                  [0.10, -0.50,  0.20,  0.84]])

    data      = L @ rng.standard_normal((4, N))
    var_names = {'X': [rf"$x_{{{i+1}}}$" for i in range(4)]}

    corr = np.corrcoef(data)
    print(f"\nFisher 95% CIs   (N = {N})")
    print("-" * 52)
    for i in range(4):
        for j in range(i+1, 4):
            r = corr[i, j]
            lo, hi = corr_ci(r, N, ci=0.95)
            print(f"  r(x{i+1},x{j+1}) = {r:+.4f}   "
                  f"95% CI: [{lo:+.4f}, {hi:+.4f}]   width = {hi-lo:.4f}")
    print()

    fig = plot_scatter_hist(data, n_bins=25, figNo=1,
                            var_names=var_names, font_size=10, ci=0.95)

    input("   Press Enter to exit ...  ")


if __name__ == '__main__':
    main()

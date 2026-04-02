#! /usr/bin/env -S python3 -i

import numpy as np
import matplotlib.pyplot as plt

from multivarious.rvs import normal
from multivarious.utl import format_plot

# Light background tints for each block type in the upper triangle
_BLOCK_TINT = {
    'XX': '#eaecf8',   # light navy
    'XY': '#e0f4f4',   # light darkcyan
    'YY': '#e8f5e9',   # light darkgreen
}

# Scatter/histogram colors per block type
_BLOCK_COLOR = {
    'XX': 'navy',
    'XY': 'darkcyan',
    'YY': 'darkgreen',
}

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
    z_crit = normal.inv(1.0 - alpha / 2.0)       # positive critical value z_{1-alpha/2}
    z_r    = np.arctanh(np.clip(r, -0.9999999, 0.9999999))
    se     = 1.0 / np.sqrt(N - 3)
    lo     = np.tanh(z_r - z_crit * se)
    hi     = np.tanh(z_r + z_crit * se)
    return float(np.clip(lo, -1.0, 1.0)), float(np.clip(hi, -1.0, 1.0))


def _block_type(iRow, iCol, nInp):
    '''Return block type string for a given (iRow, iCol) cell.'''
    if iRow < nInp and iCol < nInp:
        return 'XX'
    elif iRow >= nInp and iCol >= nInp:
        return 'YY'
    else:
        return 'XY'


def plot_scatter_hist(dataX, dataY, fig_no=100, var_names=None,
                 n_bins=20, font_size=15, ci=0.95):
    '''
    Scatter plot matrix for two sets of variables X and Y.

    Layout:
      diagonal    -- histogram of each variable
      lower tri   -- scatter plot of each pair
      upper tri   -- Pearson rho with Fisher CI displayed as three lines:
                       upper CI bound  (larger font, black)
                       < rho = value < (smaller font, black)
                       lower CI bound  (larger font, black)

    Color scheme:
      XX (input  vs input)  -- navy
      YY (output vs output) -- darkgreen
      XY (input  vs output) -- darkcyan

    INPUTS      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    dataX       matrix of input  data, one variable per row            nInp x m
    dataY       matrix of output data, one variable per row            nOut x m
    fig_no      figure number for plotting (default = 100)             1 x 1
    var_names   optional dict with keys 'X' and 'Y' containing lists
                of variable name strings for axis labels               dict
    n_bins      number of histogram bins (default = 20)                1 x 1
    font_size   base font size for format_plot (default = 10)          1 x 1
    ci          confidence level for Fisher CI (default = 0.95)        1 x 1

    OUTPUTS     DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    fig         matplotlib Figure object
    '''
    nInp, m = dataX.shape
    nOut, _ = dataY.shape
    nTotal  = nInp + nOut

    # Variable names
    if var_names is None:
        xNames = [rf"$x_{{{i+1}}}$" for i in range(nInp)]
        yNames = [rf"$y_{{{i+1}}}$" for i in range(nOut)]
    else:
        xNames = var_names.get('X', [rf"$x_{{{i+1}}}$" for i in range(nInp)])
        yNames = var_names.get('Y', [rf"$y_{{{i+1}}}$" for i in range(nOut)])

    # Stack data and names for unified indexing
    data  = np.vstack([dataX, dataY])           # nTotal x m
    names = xNames + yNames

    ci_pct = int(round(ci * 100))

    plt.ion()
    fig = plt.figure(fig_no, figsize=(3*nTotal, 3*nTotal))
    plt.clf()

    plotIndex = 1

    for iRow in range(nTotal):
        for iCol in range(nTotal):

            ax     = plt.subplot(nTotal, nTotal, plotIndex)
            btype  = _block_type(iRow, iCol, nInp)
            color  = _BLOCK_COLOR[btype]
            xData  = data[iCol, :]
            yData  = data[iRow, :]
            xLabel = names[iCol]
            yLabel = names[iRow]

            if iRow == iCol:
                # Diagonal: histogram
                ax.hist(xData, bins=n_bins, color=color,
                        alpha=0.7, edgecolor='black')

            elif iRow < iCol:
                # Upper triangle: Fisher CI displayed as three lines
                r      = float(np.corrcoef(xData, yData)[0, 1])
                lo, hi = corr_ci(r, m, ci=ci)

                ax.text(0.5, 0.72,
                        rf"${hi:+.3f}$",
                        ha='center', va='center',
                        fontsize=font_size + 3,
                        color='black', 
                        transform=ax.transAxes)
                ax.text(0.5, 0.50,
                        rf"$< \rho={r:+.3f}\ <$",
                        ha='center', va='center',
                        fontsize=font_size+2,
                        color='black',
                        transform=ax.transAxes)
                ax.text(0.5, 0.28,
                        rf"${lo:+.3f}$",
                        ha='center', va='center',
                        fontsize=font_size + 3,
                        color='black', 
                        transform=ax.transAxes)

                ax.set_facecolor(_BLOCK_TINT[btype])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            else:
                # Lower triangle: scatter plot
                ax.plot(xData, yData, 'o', color=color,
                        markersize=2, alpha=0.5)

            # Edge labels only
            if iRow == nTotal - 1:
                ax.set_xlabel(xLabel, fontsize=font_size + 4)
                plt.setp(ax.get_xticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])

            if iCol == 0:
                ax.set_ylabel(yLabel, fontsize=font_size + 4)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=font_size + 1)
            plotIndex += 1

    plt.tight_layout(pad=0.3)
    plt.show(block=False)

    return fig


def main():
    '''
    Test with 3 inputs and 2 outputs, correlated via a linear map plus noise.
    '''
    rng  = np.random.default_rng(42)
    m    = 400

    # Input covariance via Cholesky
    L = np.array([[1.00, 0.00, 0.00],
                  [0.70, 0.71, 0.00],
                  [0.30, 0.50, 0.81]])
    dataX = L @ rng.standard_normal((3, m))

    # Outputs: linear combinations of inputs plus noise
    A     = np.array([[0.8, -0.4,  0.2],
                      [0.1,  0.6, -0.7]])
    dataY = A @ dataX + 0.5 * rng.standard_normal((2, m))

    var_names = {
        'X': [rf"$x_{{{i+1}}}$" for i in range(3)],
        'Y': [rf"$y_{{{i+1}}}$" for i in range(2)],
    }

    fig = plot_scatter_hist(dataX, dataY, fig_no=1,
                       var_names=var_names, font_size=15, ci=0.95)

    input("  Press Enter to Exit ... ")


if __name__ == '__main__':
    main()

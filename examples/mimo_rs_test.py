#! /usr/bin/env -S python3 -i

import numpy as np
import matplotlib.pyplot as plt
from multivarious.fit import mimo_rs
from multivarious.rvs import lognormal 


def mimo_rs_test():
    '''
    mimo_rs_test
    
    Test mimo_rs for fitting high dimensional multi-input, multi-output data
    
    ... Is this the fundamental problem statement of all data-science?
    '''
    
    data = 'SyntheticData'  # 'SyntheticData' or 'MeasuredData'
    
    # "seed" rand() and randn() to generate a certain random sequence
    np.random.seed(30)
    
    if data == 'SyntheticData':  # apply mimo_rs to synthetic data
        
        nInp = 5      # number of input features
        nOut = 2      # number of output features
        nData = 2000  # number of observations
        pNoise = 2    # percentage random noise in the synthetic data
        
        medX = np.arange(1, nInp + 1)        # median values of X1 ... Xn
        covX = 0.4 * np.ones(nInp)           # coefficients of variation of X1 ... Xn
        
        # make up a symmetric positive definite correlation matrix
        R = np.eye(nInp)  # all diagonal terms of a correlation matrix are "1"
        for ii in range(nInp - 1):  # loop over the off-diagonal terms
            rr = 0.9 / (ii + 1)**0.2  # the value of an off-diagonal term
            if (ii + 1) % 2 == 0:
                rr = -rr  # switch +ve and -ve for off diagonals
            R = R + rr * np.diag(np.ones(nInp - ii - 1), ii + 1)  # upper off diagonal
            R = R + rr * np.diag(np.ones(nInp - ii - 1), -(ii + 1))  # lower off diagonal
        
        # a sample of m observations of n correlated lognormal variables
        xData = lognormal.rnd( medX, covX, nData, R )
        
        # presume that the output feature is a specific function of n variables ...
        yData, lgd, maxN, rmsN = test_function(xData, nOut, pNoise)
    
    elif data == 'MeasuredData':  # apply mimo_rs from data loaded from a data file
        
        # Update this path to your actual data file
        Data = np.loadtxt('/home/hpgavin/Research/Nepal-EEWS/m-files/data_20220606_csv.csv',
                          delimiter=',')
        
        r2k = np.where(Data[:, 0] != 0)[0]  # rows to keep, discard rows starting with 0
        
        xData = Data[r2k, [0, 1, 2, 3, 4]].T  # use columns 1, 2, 3, 4, 5, for example
        yData = Data[r2k, [5, 6]].T           # PGV and PGA
        
        nInp, nData = xData.shape  # nInp = number of rows
        nOut, nData = yData.shape  # nOut = number of rows
    
    number_of_data_points = nData
    print(f'Number of data points: {number_of_data_points}')
    print(f'{np.min(xData):.6f} < xData < {np.max(xData):.6f}')
    print(f'{np.min(yData):.6f} < yData < {np.max(yData):.6f}')
    
    # remove each column of xData and yData with outliers
    min_allow_data = 1e-4
    max_allow_data = 5000
    
    XY = np.vstack([xData, yData])
    XY = XY[:, np.all(XY > min_allow_data, axis=0)]
    XY = XY[:, np.all(XY < max_allow_data, axis=0)]
    
    nData = XY.shape[1]
    print(f'After outlier removal: {nData} data points')
    xData = XY[:nInp, :]
    yData = XY[nInp:nInp + nOut, :]
    
    print(f'{np.min(xData):.6f} < xData < {np.max(xData):.6f}')
    print(f'{np.min(yData):.6f} < yData < {np.max(yData):.6f}')
    
    # Plot input-output relationships before fitting
    for iy in range(nOut):  # for each output feature scatter plot the ...
        plt.figure(10 + iy)   # ... output feature w.r.t. each input feature individually
        cMap = rainbow(nOut)
        plt.clf()
        format_plot(14, 1, 2)
        for ix in range(nInp):
            plt.subplot(nInp, 1, ix + 1)
            plt.plot(xData[ix, :], yData[iy, :], 'o', color=cMap[iy, :])
            plt.xlabel(f'X_{ix + 1}')
            plt.ylabel(f'Y_{iy + 1}')
            plt.axis('tight')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    # mimo_rs parameters
    maxOrder = 5     # maximum polynomial order for the model
    pTrain = 80      # percentage of the data for training (remaining for testing)
    pCull = 80       # percentage of the model to be culled
    tol = 0.10       # maximum desired coefficient of variation
    # scaling = 0    # no scaling
    # scaling = 1    # subtract mean and divide by std.dev
    # scaling = 2    # subtract mean and decorrelate
    # scaling = 3    # log-transform, subtract mean and divide by std.dev
    scaling = 4      # log-transform, subtract mean and decorrelate
    
    order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
        mimo_rs(xData, yData, maxOrder, pTrain, pCull, tol, scaling)
    
    if data == 'SyntheticData':
        print(f'\nSynthetic data statistics:')
        print(f'Maximum noise: {maxN:.6f}')
        print(f'RMS noise: {rmsN:.6f}')
    
    # Plot results: data and model predictions
    for iy in range(nOut):  # for each output feature scatter-plot the ...
        plt.figure(20 + iy)   # ... output feature w.r.t. each input feature individually
        cMap = rainbow(nOut)
        plt.clf()
        format_plot(14, 1, 2)
        for ix in range(nInp):
            plt.subplot(nInp, 1, ix + 1)
            plt.plot(xData[ix, :], yData[iy, :], 'ok', label='Data', markersize=3)
            plt.plot(testX[ix, :], testModelY[iy, :], 'o', color=cMap[iy, :], 
                    label='Model', markersize=4)
            plt.xlabel(f'X_{ix + 1}')
            plt.ylabel(f'Y_{iy + 1}')
            plt.axis('tight')
            if ix == 0:
                plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    plt.show()
    
    return order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY


def test_function(X, nOut, pNoise):
    '''
    [Y, lgd, maxN, rmsN] = test_function(X, nOut, pNoise)
    generate synthetic data to test mimo_rs
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    X           input feature matrix                                  nInp x nData
    nOut        number of output features                             1 x 1
    pNoise      percentage noise to add                               1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Y           output feature matrix                                 nOut x nData
    lgd         legend strings for plotting                           list
    maxN        maximum noise value                                   1 x 1
    rmsN        RMS noise value                                       1 x 1
    '''
    
    nInp, nData = X.shape
    Y = np.full((nOut, nData), np.nan)
    
    for ix in range(nInp):
        if ix < 1:
            for iy in range(nOut):
                if iy == 0:
                    Y[iy, :] = 0.2 * X[0, :] + 1
                elif iy == 1:
                    Y[iy, :] = 0.5 * X[0, :] + 0
            lgd = ['0.2 X_1']
        
        if ix < 2 and nInp >= 2:
            for iy in range(nOut):
                if iy == 0:
                    Y[iy, :] = Y[iy, :] + np.sin(X[1, :]) + 1
                elif iy == 1:
                    Y[iy, :] = Y[iy, :] + np.cos(X[1, :]) + 1
            lgd = ['0.2 X_1', 'sin(X_2)']
        
        if ix < 3 and nInp >= 3:
            for iy in range(nOut):
                if iy == 0:
                    Y[iy, :] = Y[iy, :] + 0.5 * np.cos(2 * X[2, :]) + 1
                elif iy == 1:
                    Y[iy, :] = Y[iy, :] + 0.5 * np.sin(2 * X[2, :]) + 1
            lgd = ['0.2 X_1', 'sin(X_2)', 'cos(2 X_3)']
        
        if ix < 4 and nInp >= 4:
            for iy in range(nOut):
                if iy == 0:
                    Y[iy, :] = Y[iy, :] + np.tanh(0.4 * X[3, :]) - 1
                elif iy == 1:
                    Y[iy, :] = Y[iy, :] - np.tanh(0.4 * X[3, :]) + 1
            lgd = ['0.2 X_1', 'sin(X_2)', 'cos(2 X_3)', 'tanh(0.4 X_4)']
        
        if ix < 5 and nInp >= 5:
            for iy in range(nOut):
                if iy == 0:
                    Y[iy, :] = Y[iy, :] + 2 * np.exp(-(0.2 * X[4, :])**2)
                elif iy == 1:
                    Y[iy, :] = Y[iy, :] + 2 * np.exp(-(0.5 * X[4, :])**2)
            lgd = ['X_1', 'sin(X_2)', 'cos(2 X_3)', 'tanh(0.4 X_4)', '2 exp(-(X_5)^2)']
    
    minY = np.min(Y)
    maxY = np.max(Y)
    
    print(f'\nTest function output range:')
    print(f'minY = {minY:.6f}')
    print(f'maxY = {maxY:.6f}')
    
    # normally distributed noise
    noise = np.random.randn(*Y.shape) * (maxY - minY) * pNoise / 100.0
    
    maxN = np.max(np.abs(noise))  # maximum of noise
    rmsN = np.linalg.norm(noise, 'fro') / np.sqrt(np.prod(noise.shape))  # root mean square noise
    
    Y = Y + noise  # add some noise to the Y data
    
    return Y, lgd, maxN, rmsN


if __name__ == '__main__':
    '''
    Run the mimo_rs test when this script is executed directly
    '''
    print('=' * 70)
    print('mimo_rs Test Function')
    print('Testing high-dimensional multi-input multi-output polynomial fitting')
    print('=' * 70)
    
    results = mimo_rs_test()
    
    print('\n' + '=' * 70)
    print('Test completed successfully!')
    print('=' * 70)

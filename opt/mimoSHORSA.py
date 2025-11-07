import numpy as np
import matplotlib.pyplot as plt
import time as time
from datetime import datetime, timedelta
from rainbow import rainbow 
from format_plot import format_plot

def mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=50, pCull=30, tol=0.10, scaling=1, L1_pnlty=1.0, basis_fctn='H'): 
    '''
    [ order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY ] = mimoSHORSA( dataX, dataY, maxOrder, pTrain, pCull, tol, scaling )
    
    mimoSHORSA
    multi-input multi-output Stochastic High Order Response Surface Algorithm
    
    This program fits a high order polynomial to multidimensional data via
    the high order response surface (mimoSHORSA) method 
    
     mimoSHORSA approximates the data with a polynomial of arbitrary order,
       y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
              \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.
    The first stage of the algorithm determines the correct polynomial order,
    k_i in the response surace. Then the formulation of the mixed terms 
    \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq} are derived by the second stage
    based on previous results. In the third stage, the response surface 
    is approximated.
    
    INPUT       DESCRIPTION                                                DEFAULT
    --------    --------------------------------------------------------   -------
    dataX       m observations of n input  features in a (nx x m) matrix
    dataY       m observations of m output features in a (ny x m) matrix
    maxOrder    maximum allowable polynomial order                            3
    pTrain      percentage of data for training (remaining for testing)      50
    pCull       maximum percentage of model which may be culled              30 
    tol         desired maximum model coefficient of variation                0.10
    scaling     scale the data before fitting                                 1
                scaling = 0 : no scaling
                scaling = 1 : subtract mean and divide by std.dev
                scaling = 2 : subtract mean and decorrelate
                scaling = 3 : log-transform, subtract mean and divide by std.dev
                scaling = 4 : log-transform, subtract mean and decorrelate
    L1_pnlty    coefficient for L1 regularization                             1.0
    basis_fctn  basis function type                                            'H'
                'H': Hermite functions
                'L': Legendre polynomials
                'P': Power polynomials

    
    OUTPUT      DESCRIPTION
    --------    --------------------------------------------------------
     order      list of matrices of the orders of variables in each term in the polynomial 
     coeff      list of polynomial coefficients 
     meanX      mean vector of the scaled dataX
     meanY      mean vector of the scaled dataY
     trfrmX     transformation matrix from dataZx to dataX
     trfrmY     transformation matrix from dataZy to dataY
     testModelY output features for model testing
     testX      input  features for model testing 
     testY      output features for model testing 
    
    Reference:
       Gavin, HP and Yau SC, "High order limit state functions in the 
       response surface method for structural reliability analysis,"
       submitted to Structural Safety, December 2005.     
    
    Department of Civil and Environmental Engineering
    Duke University
    Siu Chung Yau, Henri P. Gavin, January 2006, 2023
    '''
    
    print('\n Multi-Input Multi-Output High Order Response Surface (mimoSHORSA)\n')
    
    # Handle default arguments and convert to appropriate types
    maxOrder = int(round(abs(maxOrder)))
    pTrain = abs(pTrain) / 100 if pTrain > 1 else abs(pTrain)
    pCull = abs(pCull) / 100 if pCull > 1 else abs(pCull)
    tol = abs(tol)
    scaling = int(round(abs(scaling)))
    L1_pnlty = abs(L1_pnlty)
    if L1_pnlty > 0: # No "culling" with L1 regularization
        pCull = 0
    
    nInp, mDataX = dataX.shape   # number of columns in dataX is mData
    nOut, mDataY = dataY.shape   # number of columns in dataY is mData
    
    if mDataX != mDataY:
        raise ValueError('the dataX and dataY matrices must have the same number of columns')
    else:
        mData = mDataX
    
    # initialize lists for ...
    B = [None] * nOut           # correlating matrix
    coeff = [None] * nOut       # model coefficient vector
    coeffCOV = [None] * nOut    # coefficient of variation of the model coefficients
    
    trainX, trainY, mTrain, testX, testY, mTest = split_data(dataX, dataY, pTrain)
    
    # scale data matrices trainX and trainY separately since using 
    # the covariance between trainX and trainY in the model is "cheating"
    trainZx, meanX, trfrmX = scale_data(trainX, scaling, 0)
    print('aaa')
    trainZy, meanY, trfrmY = scale_data(trainY, scaling, 0)
    print('bbb')
    
    if scaling > 0:  # remove each column of trainZx and trainZy with outliers
        XY = np.vstack([trainZx, trainZy])
        XY = XY[:, np.all(XY > -4, axis=0)]
        XY = XY[:, np.all(XY < 4, axis=0)]
        
        nData = XY.shape[1]
        trainZx = XY[:nInp, :]
        trainZy = XY[nInp:nInp+nOut, :]
        print(f'{np.min(trainZx):.6f} < trainZx < {np.max(trainZx):.6f}')
        print(f'{np.min(trainZy):.6f} < trainZy < {np.max(trainZy):.6f}')
    
    time.sleep(1) # if needed for debugging
    
    # separate order for each variable --- Not needed if data is already provided
    # [maxOrder, orderR2] = polynomial_orders(maxOrder)
    
    maxOrder = maxOrder * np.ones(nInp, dtype=int)  # same maximum order for all variables
    
    order, nTerm = mixed_term_orders(maxOrder, nInp, nOut)
    
    # initialize variables
    maxCull = max(1,int(round(pCull * nTerm[0]))) # maximum number of terms to cull
    condB = np.full((nOut, maxCull), np.nan) # condition number of basis as model is culled
    for io in range(nOut):
        coeffCOV[io] = np.ones(nTerm[0])
    
    trainMDcorr = np.full((nOut, maxCull), np.nan)
    testMDcorr = np.full((nOut, maxCull), np.nan)
    coeffCOVmax = np.full((nOut, maxCull), np.nan)
    
    # start a timer to measure computational time
    start_time = time.time()
    
    for iter in range(maxCull):  # cull uncertain terms from the model ------------
        
        # plot model coefficients and correlations at first and last culling iter
        if (iter == 0) or (iter == maxCull - 1) or (np.max(coeffCOVmax[:, iter]) < 2*tol):
            trainFigNo = 200
            testFigNo = 300
        else:
            trainFigNo = 0
            testFigNo = 0
        
        # fit ("train") a separate model for each output (dependent) variable
        for io in range(nOut):
            coeff[io], condB[io, iter] = fit_model(trainZx, trainZy[io, :], order[io], nTerm[io], mTrain, L1_pnlty, basis_fctn)
        
        # compute the model for the training data and the testing data
        trainModelY, B = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, trainX, scaling, basis_fctn)
        testModelY, _  = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, testX, scaling, basis_fctn)

        
        # evaluate the model for the training data and the testing data
        trainMDcorr[:, iter], coeffCOV, _, _ = evaluate_model(B, coeff, trainY, trainModelY, trainFigNo, 'train')
        testMDcorr[:, iter], _, R2adj, AIC = evaluate_model(B, coeff, testY, testModelY, testFigNo, 'test')
        
        for io in range(nOut):
            coeffCOVmax[io, iter] = np.max(coeffCOV[io])
        
        print_model_stats(iter, coeff, order, coeffCOV, testMDcorr[:, iter], R2adj, scaling, maxCull)
        
        if L1_pnlty == 0:
            plt.ion() # interactive mode: on
            for io in range(nOut):
                plt.figure(400 + io)
                format_plot(18, 4, 8)
                cMap = rainbow(nOut)
                plt.clf()
                plt.semilogy(np.arange(1, nTerm[io] + 1), coeffCOV[io], 'o', color=cMap[io, :])
                for ii in range(nTerm[io]):
                    plt.text(ii + 1, 0.85 * coeffCOV[io][ii], 
                            f' {order[io][ii, :]}', fontsize=10)
                plt.ylabel('coefficient of variation')
                plt.xlabel('term number')
                plt.title(f'Y_{io}, ρ_train = {trainMDcorr[io, iter]:.3f}, ' +
                        f'ρ_test = {testMDcorr[io, iter]:.3f}, cond(B) = {condB[io, iter]:.1f}')
            
            plt.draw()
            plt.pause(1.001)
        
        if (testMDcorr[:, iter] > 0).all() and (np.max(coeffCOVmax[:, iter]) < tol):
            maxCull = iter + 1
            break

        if L1_pnlty == 0:
            order, nTerm, coeffCOV = cull_model(coeff, order, coeffCOV, tol)

    # ------------ cull uncertain terms from the model
    
    # plot correlations and coefficients of variation
    if L1_pnlty == 0: 
        plt.ion() # interactive mode: on
        plt.figure(500)
        plt.clf()
        cMap = rainbow(nOut)
        format_plot(18, 2, 4)
    
        plt.subplot(2, 1, 1)
        for io in range(nOut):
            plt.plot(np.arange(1, maxCull + 1), trainMDcorr[io, :maxCull], 'o', color=cMap[io, :])
            plt.plot(np.arange(1, maxCull + 1), testMDcorr[io, :maxCull], 'x', color=cMap[io, :])
        plt.ylabel('model-data correlation')
        plt.legend(['train', 'test'], loc='center left')
        
        plt.subplot(2, 1, 2)
        for io in range(nOut):
            plt.semilogy(np.arange(1, maxCull + 1), condB[io, :maxCull], 'o', color=cMap[io, :])
            plt.semilogy(np.arange(1, maxCull + 1), coeffCOVmax[io, :maxCull], 'x', color=cMap[io, :])
        plt.legend(['cond(B)', 'max(c.o.v.)'], loc='center right')
        plt.ylabel('maximum c.o.v.')
        plt.xlabel('model reduction')
        
        plt.show(block=False)
    
    return order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY


# Placeholder function stubs - these will be implemented as you provide them
def split_data(dataX, dataY, pTrain):
    '''
    [trainX,trainY,mTrain, testX,testY,mTest] = split_data(dataX,dataY,pTrain)
    split data into a training set and a testing set
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     dataX      m observations of nx input  "explanatory variables     nx x m 
     dataY      m observations of ny output "explanatory variables     ny x m 
     pTrain     fraction of the m observations to use for training      1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     trainX     matrix of  input data for training                    nx x mTrain
     trainY     matrix of output data for training                    ny x mTrain
     mTrain     number of observations in the training set             1 x 1
     testX      matrix of  input data for testing                     nx x mTest
     testY      matrix of output data for testing                     ny x mTest
     mTest      number of observations in the testing set              1 x 1
    '''
    
    nInp, mData = dataX.shape   # number of columns in dataX is mData
    
    mTrain = int(np.floor(pTrain * mData))  # number of data points in training data set
    mTest = mData - mTrain                   # number of data points in testing  data set
    
    reorder = np.random.permutation(mData)        # random permutation of integers [0:mData-1]
    idtrainX = reorder[:mTrain]                   # indices of training data
    idtestX = reorder[mTrain:mData]               # indices of testing data
    
    trainX = dataX[:, idtrainX]
    trainY = dataY[:, idtrainX]
    print(f'dim_train_Y = {trainY.shape})
    
    testX = dataX[:, idtestX]
    testY = dataY[:, idtestX]
    
    return trainX, trainY, mTrain, testX, testY, mTest


def polynomial_orders(maxOrder, Zx, Zy, n, tol, scaling):
    '''
    [order, orderR2] = polynomial_orders(maxOrder)
    
    NOTE: This function is not currently used in the main workflow.
    It becomes inefficient for larger datasets.
    The main function uses uniform maxOrder for all variables instead.
    '''
    print('1st Stage: Polynomial Order Determination ...')
    
    order = np.ones(n, dtype=int)      # initial guess of response surface orders
    quality = np.zeros(n)
    orderR2 = np.zeros(n)
    no_pts = maxOrder + 15             # number of sample points (must be > ki+1)
    
    # sample points along dimension X_i within the domain [-1,1]
    # (roots of no_pts-th order Chebyshev polynomial)
    z = np.cos(np.pi * (np.arange(1, no_pts + 1) - 0.5) / no_pts)
    
    for i in range(n):  # determine the orders for each variable one by one
        
        # allocate memory for matrix of sampling points along all variables
        zAvg = np.mean(Zx, axis=1)
        zMap = np.outer(zAvg, np.ones(no_pts))
        
        # the sample points along z (-1 <= z <= 1) are linearly mapped onto the domain [zMin ... zMax]
        # only the i-th row is non-zero since all other variables 
        # are kept constants at their mean values
        zMax = np.max(Zx[i, :])
        zMin = np.min(Zx[i, :])
        zMap[i, :] = (zMax + zMin + z * (zMax - zMin)) / 2
        
        # interpolate the data at the Chebyshev sampling points
        y = IDWinterp(Zx.T, Zy.T, zMap.T, 2, 10, 0.1)
        
        for ki in range(order[i], maxOrder + 1):  # loop over possible polynomial orders
            
            # values of 0-th to ki-th degree Chebyshev polynomials at
            # the sampling points.
            # this matrix is used for the determination of coefficients
            Tx = np.cos(np.arccos(z[:, np.newaxis]) * np.arange(ki + 1))
            
            d = (Tx.T @ y) / np.diag(Tx.T @ Tx)  # coefficients by least squares method
            
            residuals = y - Tx @ d               # residuals of the 1-D curve-fit
            fit_error = np.linalg.norm(residuals) / (np.linalg.norm(y) - np.mean(y))  # error of the curve fit
            orderR2[i] = 1 - fit_error**2
            
            plt.figure(103)
            format_plot(18, 4, 8)
            plt.clf()
            plt.plot(zMap[i, :], y, 'ob', label='data')
            plt.plot(zMap[i, :], Tx @ d, '*r', label='fit')
            if scaling > 0:
                plt.xlabel(f'Z_{i+1}')
            else:
                plt.xlabel(f'X_{i+1}')
            ttl = f'k_{{{i+1}}}={ki}   R^2 = {orderR2[i]:.3f}   fit-error = {fit_error:.3f}'
            plt.legend()
            plt.title(ttl)
            plt.pause(1.1)
            
            if (orderR2[i] > 1 - tol) and (fit_error < tol):  # the 1D fit is accurate
                break
        
        order[i] = ki  # save the identified polynomial order
    
    # output the results
    print('  Variable     Determined Order    R_sq ')
    for i in range(n):
        print(f'{i+1:10.0f} {order[i]:20.0f}    {orderR2[i]:9.6f}')
    
    return order, orderR2


def scale_data(Data, scaling, flag):
    '''
    [ Z, meanD, T, maxZ, minZ ] = scale_data(Data, scaling)
     scale data in one of four or five ways ..  Z = inv(T)*(Data - meanD); 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Data        a matrix of data values                               n x m
    scaling     type of scaling ...                                   1 x 1
                 scaling = 0 : no scaling
                 scaling = 1 : subtract mean and divide by std.dev
                 scaling = 2 : subtract mean and decorrelate
                 scaling = 3 : log-transform, subtract mean and divide by std.dev
                 scaling = 4 : log-transform, subtract mean and decorrelate
    flag        unused parameter (kept for compatibility)             1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Z           scaled data                                           n x m
    meanD       the arithmetic mean of the (log-transformed) data     n x 1
    T           transformation matrix                                 n x n
    maxZ        maximum value of each data sample                     1 x n
    minZ        minimum value of each data sample                     1 x n
    '''
    
    n, m = Data.shape  # m observations of n variables
    
    if scaling == 0:  # no scaling
        Z = Data.copy()
        meanD = np.zeros((n, 1))
        T = np.eye(n)
    
    elif scaling == 1:  # subtract mean and divide by std.dev
        meanD = np.mean(Data, axis=1, keepdims=True)
        T = np.diag(np.sqrt(np.var(Data, axis=1, ddof=1)))
    
    elif scaling == 2:  # subtract mean and decorrelate
        meanD = np.mean(Data, axis=1, keepdims=True)
        covData = np.cov(Data)
        dim_cov_data = covData.shape
        print(f'dim_cov_data = {dim_cov_data}')
        eVal, eVec = np.linalg.eig(covData)
        T = eVec @ np.sqrt(np.diag(eVal))
    
    elif scaling == 3:  # log-transform, subtract mean and divide by std.dev
        Data = np.log10(Data)
        meanD = np.mean(Data, axis=1, keepdims=True)
        T = np.diag(np.sqrt(np.var(Data, axis=1, ddof=1)))
    
    elif scaling == 4:  # log-transform, subtract mean and decorrelate
        Data = np.log10(Data)
        meanD = np.mean(Data, axis=1, keepdims=True)
        covData = np.cov(Data)
        eVal, eVec = np.linalg.eig(covData)
        T = eVec @ np.sqrt(np.diag(eVal))
    
    else:
        raise ValueError(f'Invalid scaling option: {scaling}. Must be 0-4.')
    
    # apply the scaling: Z = inv(T) * (Data - meanD)
    Z = np.linalg.solve(T, Data - meanD)
    
    maxZ = np.max(Z, axis=1)
    minZ = np.min(Z, axis=1)
    
    return Z, meanD, T


def clip_data(Data, lowLimit, highLimit):
    '''
     clip_data(Data, lowLimit, highLimit)
     remove outliers from the data
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Data        matrix of data                                          n x m  
    lowLimit    remove values lower  than lowLimit                      1 x 1
    highLimit   remove values higher than highLimit                     1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     Data       matrix of data without values exceeding given limits    n x m'
    '''
    
    # remove low and high values (outliers)
    
    # Keep only columns where all values are greater than lowLimit
    idxc = np.where(np.all(Data > lowLimit, axis=0))[0]
    Data = Data[:, idxc]
    
    # Keep only columns where all values are less than highLimit
    idxc = np.where(np.all(Data < highLimit, axis=0))[0]
    Data = Data[:, idxc]
    
    return Data


def scatter_data(dataX, dataY, figNo=100, varNames=None):
    '''
    scatter_data(dataX, dataY, figNo, varNames)
    Create scatter plots of each pair of input and output variables
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    dataX       matrix of input data                                   nInp x m
    dataY       matrix of output data                                  nOut x m
    figNo       figure number for plotting (default = 100)             1 x 1
    varNames    optional dict with keys 'X' and 'Y' containing
                lists of variable names for labeling                   dict
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    None        Creates matplotlib figure with scatter plot matrix
    
    NOTE: For high-dimensional data, this can create many subplots.
          The function creates a grid showing:
          - X vs X correlations in upper left
          - Y vs X correlations in lower left
          - Y vs Y correlations in lower right
    '''
    
    nInp, m = dataX.shape
    nOut, _ = dataY.shape
    
    # Set up variable names if not provided
    if varNames is None:
        xNames = [f'X_{i+1}' for i in range(nInp)]
        yNames = [f'Y_{i+1}' for i in range(nOut)]
    else:
        xNames = varNames.get('X', [f'X_{i+1}' for i in range(nInp)])
        yNames = varNames.get('Y', [f'Y_{i+1}' for i in range(nOut)])
    
    # Calculate number of subplots needed
    nTotalVars = nInp + nOut
    
    # Create figure with subplots
    plt.ion() # interactive mode: on
    fig = plt.figure(figNo, figsize=(3*nTotalVars, 3*nTotalVars))
    plt.clf()
    
    plotIndex = 1
    
    # Create scatter plots for all pairs
    for iRow in range(nTotalVars):
        for iCol in range(nTotalVars):
            
            ax = plt.subplot(nTotalVars, nTotalVars, plotIndex)
            
            # Determine which data to plot
            if iRow < nInp and iCol < nInp:
                # X vs X
                xData = dataX[iCol, :]
                yData = dataX[iRow, :]
                xLabel = xNames[iCol]
                yLabel = xNames[iRow]
                color = 'blue'
                
            elif iRow >= nInp and iCol < nInp:
                # Y vs X
                xData = dataX[iCol, :]
                yData = dataY[iRow - nInp, :]
                xLabel = xNames[iCol]
                yLabel = yNames[iRow - nInp]
                color = 'red'
                
            elif iRow >= nInp and iCol >= nInp:
                # Y vs Y
                xData = dataY[iCol - nInp, :]
                yData = dataY[iRow - nInp, :]
                xLabel = yNames[iCol - nInp]
                yLabel = yNames[iRow - nInp]
                color = 'green'
                
            else:
                # X vs Y (upper right - leave empty or skip)
                plotIndex += 1
                continue
            
            # Create scatter plot
            if iRow == iCol:
                # Diagonal: plot histogram instead of scatter
                ax.hist(xData, bins=20, color=color, alpha=0.7, edgecolor='black')
                ax.set_ylabel('Count')
            else:
                # Off-diagonal: scatter plot
                ax.plot(xData, yData, 'o', color=color, markersize=2, alpha=0.5)
            
            # Add labels only on edges
            if iRow == nTotalVars - 1:
                ax.set_xlabel(xLabel, fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if iCol == 0:
                ax.set_ylabel(yLabel, fontsize=8)
            else:
                ax.set_yticklabels([])
            
            # Make tick labels smaller
            ax.tick_params(labelsize=6)
            
            plotIndex += 1
    
    plt.tight_layout()
    plt.suptitle('Pairwise Scatter Plots', fontsize=14, y=1.0)
    plt.show(block=False)
    
    return None


def mixed_term_orders(maxOrder, nInp, nOut):
    '''
    [ order , nTerm ] = mixed_term_orders( maxOrder, nInp, nOut )
    specify the exponents on each input variable for every term in the model,
    and the total number of terms, nTerm
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    maxOrder    maximum polynomial order of the model                   1 x nInp
    nInp        number of input  (explanatory) variables                1 x 1
    nOut        number of output  (dependent)  variables                1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    order       list of 2D matrices of the orders 
                indicating the orders of each explanatory variable in
                in each polynomial term                             {nTerm x nInp}
                one matrix for each output
                initially, these matrices are all the same 
    nTerm       number of polynomial terms in the model               nOut x 1
    
    The matrix 'order' indicates which mixed term power-products are present
    in each term of the model. 
    
    Each element in this matrix represents the order of a variable in a term.
    The columns of 'order' correspond to individual variables. 
    Each row indicates the powers present in the product of a term.
    
    Algorithm by Siu Chung Yau (2006)
    '''
    
    print('Determine the Mixed Term Power Products ...')
    
    nTerm_total = int(np.prod(maxOrder + 1))
    
    ordr = np.zeros((nTerm_total, nInp), dtype=int)  # allocate memory for the 'order' matrix
    term = np.zeros(nInp + 1, dtype=int)              # orders in a given term (extra element for carry)
    term[0] = -1                                      # starting value for the first term
    
    for t in range(nTerm_total):                      # loop over all terms
        term[0] = term[0] + 1                         # increment order of the first variable
        for v in range(nInp):                         # check every column in the row
            if term[v] > maxOrder[v]:
                term[v] = 0
                term[v + 1] = term[v + 1] + 1         # increment columns as needed
        
        ordr[t, :] = term[:nInp]  # save the orders of term t in the 'order' matrix
    
    # The power of a variable in each term can not be greater than 
    # the order of that variable alone.  
    # Remove the terms in which the total order is larger than 
    # the highest order term.
    
    it = np.where(np.sum(ordr, axis=1) <= np.max(maxOrder))[0]
    ordr = ordr[it, :]
    
    # The number of rows in the matrix 'order' is the number of
    # required terms in the model
    
    nTerm = np.full(nOut, ordr.shape[0], dtype=int)
    
    order = [None] * nOut
    for io in range(nOut):
        order[io] = ordr.copy()
    
    print(f'  Total Number of Terms: {nTerm[0]:3d}')
    print(f'  Number of Mixed Terms: {nTerm[0] - np.sum(maxOrder) - 1:3d}\n')
    
    return order, nTerm


def polynomial_product(order, Zx, max_order, basis_fctn='H'):
    '''
    psyProduct = polynomial_product( order, Zx, basis_fctn )
    compute the product of hermite functions of given orders (from 0 to 5)
    for a set of column vectors Z, where each column of Zx has a given order
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    order       expxonents present in each polynomial term             1 x nInp
     Zx         matrix of scaled input (explanatory) variables      mData x nInp
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    psyProduct  vector of product of hermite polynomials             mData x 1
    '''
    
    nInp = len(order) # number of input (explanatory) variables
    psyProduct = np.ones(Zx.shape[0])  # initialize to vector of 1

    for i in range(nInp):
        if order[i] > 0:
            if basis_fctn == 'H':
                psyProduct *= hermite( order[i], Zx[:, i], max_order )
            elif basis_fctn == 'L':
                psyProduct *= legendre( order[i], Zx[:, i] )
    
    return psyProduct


def build_basis(Zx, order, basis_fctn='H'):
    '''
    B = build_basis( Zx, order, basis_fctn )
    compute matrix of model basis vectors
    options: power-polynomial basis, Hermite function basis, or Legendre basis

    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------  -----------
     Zx         matrix of input (explanatory) variables               nInp x mData
     order      orders for each variable in each polynomial term      nTerm x nInp
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial

    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------  -----------
      B         matrix basis vectors for the polynomial model       mData x nTerm
    '''

    mData = Zx.shape[1]             # number of data points
    nTerm, nInp = order.shape       # number of terms, inputs, outputs 
    B = np.ones((mData, nTerm))     # the matrix of model basis vectors 

    max_order = int(np.max(order))

    # in the matrix of basis vectors, B, 
    # columns correspond to each term in the polynomial and 
    # rows correspond to each observation 
 
    if basis_fctn == 'P':
        # Power polynomials
        for it in range(nTerm):
            B[:, it] = np.prod(Zx.T ** order[it, :], axis=1)
    else:
        # Legendre or Hermite
        for it in range(nTerm):
            B[:, it] = polynomial_product(order[it, :], Zx.T, max_order, basis_fctn)

    return B

   
def hermite(n, z, N):
    '''
    psy = hermite(n, z, N)
    compute the Hermite function of a given order (orders from 0 to 10)
    for a vector of values of z 
    https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
    Note: These Hermite functions are index-shifted by 2, in order to 
    augment the basis with a constant (0-order) function and a linear (1-order) 
    function.  The 0-order and 1-order functions have approximately unit-area and 
    attenuate exponentially at rates comparable to the highest order Hermite 
    function in the basis.  
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    n           the polynomial order of a hermite function            1 x 1
    z           vector of input (explanatory) variables               1 x mData
    N           largest order in the full expansion                   1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     psy        a hermite function of specified order at given values 1 x mData
    '''
    
    pi4 = np.pi**(0.25)
    ez2 = np.exp(-0.5 * z**2)

    z0 = N+2;     # expand the domain of extrapolation 

    if n == 0:
        psy = exp(-(z/z0)**(6*z0))
    elif n == 1:
        psy = (z/z0) * exp(-(z/z0)**(6*z0)) 
    elif n == 2:
        psy = 1/pi4 * ez2
    elif n == 3:
        psy = np.sqrt(2)/pi4 * z * ez2
    elif n == 4:
        psy = 1/(np.sqrt(2)*pi4) * (2*z**2 - 1) * ez2
    elif n == 5:
        psy = 1/(np.sqrt(3)*pi4) * (2*z**3 - 3*z) * ez2
    elif n == 6:
        psy = 1/(2*np.sqrt(6)*pi4) * (4*z**4 - 12*z**2 + 3) * ez2
    elif n == 7:
        psy = 1/(2*np.sqrt(15)*pi4) * (4*z**5 - 20*z**3 + 15*z) * ez2
    elif n == 8:
        psy = 1/(12*np.sqrt(5)*pi4) * (8*z**6 - 60*z**4 + 90*z**2 - 15) * ez2
    elif n == 9:
        psy = 1/(6*np.sqrt(70)*pi4) * (8*z**7 - 84*z**5 + 210*z**3 - 105*z) * ez2
    elif n == 10:
        psy = 1/(24*np.sqrt(70)*pi4) * (16*z**8 - 224*z**6 + 840*z**4 - 840*z**2 + 105) * ez2
    elif n == 11:
        psy = 1/(72*np.sqrt(35)*pi4) * (16*z**9 - 288*z**7 + 1512*z**5 - 2520*z**3 + 945*z) * ez2
    elif n == 12:
        psy = 1/(720*np.sqrt(7)*pi4) * (32*z**10 - 720*z**8 + 5040*z**6 - 12600*z**4 + 9450*z**2 - 945) * ez2
    else:
        raise ValueError(f'Hermite function implemented only for orders 0-12, got order={order}')
    
    return psy


def legendre(n, z):
    '''
    Compute a Legendre polynomial of order n evaluated at points z
    Legendre polynomials are orthogonal on [-1, 1]:
    integral_{-1}^{1} P_m(z) P_n(z) dz = 2/(2n+1) * δ{mn}

    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    n           the polynomial order of a legendre polynomial          1 x 1
    z           vector of input (explanatory) variables                1 x mData
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     psy        a legendre polynomial of given order at given values  1 x mData
    '''

    z = np.asarray(z)

    if n == 0:
        return np.ones_like(z)
    elif n == 1:
        return z
    elif n == 2:
        psy = (    3*z**2 -     1  )/2;
    elif n == 3:
        psy = (    5*z**3 -     3*z)/2;
    elif n == 4:
        psy = (   35*z**4 -    30*z**2 +    3  )/8;
    elif n == 5:
        psy = (   63*z**5 -    70*z**3 +   15*z)/8;
    elif n == 6:
        psy = (  231*z**6 -   315*z**4 +  105*z**2 -    5  )/16;
    elif n == 7:
        psy = (  429*z**7 -   693*z**5 +  315*z**3 -   35*z)/16;
    elif n == 8:
        psy = ( 6435*z**8 - 12012*z**6 + 6930*z**4 - 1260*z**2+  35   )/128;
    elif n == 9:
        psy = (12155*z**9 - 25740*z**7 +18018*z**5 - 4620*z**3+ 315*z )/128;
    elif n == 10:
        psy = (46189*z**10-109395*z**8 +90090*z**6 -30030*z**4+3465.*z**2-63)/256;
    else:
        raise ValueError(f'Legendre function implemented only for orders 0-10, got order={order}')

    return psy


def fit_model(Zx, Zy, order, nTerm, mData, L1_pnlty, basis_fctn):
    '''
    [ coeff , condB ] = fit_model( Zx, Zy, order, nTerm, mData, L1_pnlty, basis_fctn )
    via singular value decomposition without regularization, or
    via quadratic programming with L1 regularization
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     Zx         scaled input (explanatory) data                      nx x mData
     Zy         scaled output (dependent) data                        1 x mData
     order      order of each explanatory variable in each term   nTerm x nx
     nTerm      number of terms in the polynomial model                  1 x 1
     mData      number of data points (not used, kept for compatibility) 1 x 1
     L1_pnlty   L1 regularization coefficient                            1 x 1
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial

    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     coeff      vector of model coefficients                           nTerm x 1  
     condB      condition number of the model basis                     1 x 1
    '''
    
    print(f'Fit The Model ... with L1_pnlty = {L1_pnlty}')

    B = build_basis( Zx, order, basis_fctn )

    if L1_pnlty > 0:
        # Use L1_fit for regularization
        try:
            from L1_fit import L1_fit
            from L1_plots import L1_plots

            # Zy needs to be column vector for L1_fit
            Zy_col = Zy.reshape(-1, 1) if Zy.ndim == 1 else Zy.reshape(-1, 1)

            coeff, mu, nu, cvg_hst = L1_fit(B, Zy_col, L1_pnlty, w=0)

            # Optional: plot L1 convergence
            # L1_plots(B, coeff, Zy_col, cvg_hst, L1_pnlty, 0, fig_no=7000)

        except ImportError:
            print('WARNING: L1_fit not found, using OLS instead')
            coeff = np.linalg.lstsq(B, Zy, rcond=None)[0]
    else:
        # Use ordinary least squares / SVD
        coeff = np.linalg.lstsq(B, Zy, rcond=None)[0]

    condB = np.linalg.cond(B)

    print(f'  condition number of model basis matrix = {condB:6.1f}')

    return coeff, condB


def compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, dataX, scaling, basis_fctn='H'):
    '''
    [ modelY, B ] = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, dataX, scaling,basis_fctn)
    compute a multivariate polynomial model 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     order      list of order of each explanatory variable in each term {nTerm x nInp}
     coeff      list of model coefficient vectors                       {nTerm x 1}
     meanX      mean of pre-scaled input  (explanatory) variables     nInp x 1 
     meanY      mean of pre-scaled output  (dependent)  variables     nOut x 1 
     trfrmX     transformation matrix for input variables             nInp x nInp 
     trfrmY     transformation matrix for output variables            nOut x nOut
     dataX      input data to evaluate model on                       nInp x mData
     scaling    scaling type ... see scale_data function                 1 x 1
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    modelY      computed model                                        nOut x mData
     B          list of basis vector matrices of the computed model  {mData x nTerm}
    '''
    
    nInp, mData = dataX.shape        # number of columns in dataX is mData
    nOut = len(order)                # number of output variables
    modelZy = np.full((mData, nOut), np.nan)  # initialize model output
    B = [None] * nOut                # initialize basis list
    
    # Transform input data according to scaling
    if scaling == 0:
        dataZx = dataX
    
    elif scaling in [1, 2]:
        dataZx = np.linalg.solve(trfrmX, dataX - meanX)
    
    elif scaling in [3, 4]:
        log10X = np.log10(dataX)
        dataZx = np.linalg.solve(trfrmX, log10X - meanX)  # standard normal variables
    
    # Compute model for each output
    for io in range(nOut):
        B[io] = build_basis(dataZx, order[io], basis_fctn)
        modelZy[:, io] = B[io] @ coeff[io]
    
    # Inverse transform to original scale
    if scaling == 0:
        modelY = modelZy.T
    
    elif scaling in [1, 2]:
        modelY = trfrmY @ modelZy.T + meanY
    
    elif scaling in [3, 4]:
        modelY = 10**(trfrmY @ modelZy.T + meanY)
    
    return modelY, B


def evaluate_model(B, coeff, dataY, modelY, figNo, txt):
    '''
    [ MDcorr, coeffCOV , R2adj, AIC ] = evaluate_model( B, coeff, dataY, modelY, figNo )
    evaluate the model statistics 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     B          list of basis vector matrices of each model          {mData x nTerm}
     coeff      list of coefficient vectors of each model            {nTerm x 1}  
     dataY      output (dependent) data                               nOut x mData
     modelY     model predictions                                     nOut x mData
     figNo      figure number for plotting (figNo = 0: don't plot)         1 x 1
       txt      annotation text
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    MDcorr      model-data correlation                                nOut x 1
    coeffCOV    list of coefficient of variation of
                each model coefficient of each model                 { 1 x nTerm }
    R2adj       adjusted R-squared for each model                     nOut x 1
    AIC         Akaike information criterion for each model           nOut x 1
    '''
    
    nOut, mData = dataY.shape
    MDcorr = np.full(nOut, np.nan)
    R2adj = np.zeros(nOut)
    AIC = np.zeros(nOut)
    coeffCOV = [None] * nOut
    
    # statistical analysis of coefficients
    residuals = dataY - modelY    # matrix of residuals for each model
    
    for io in range(nOut):
        
        nTerm = coeff[io].shape[0]    # number of terms in model "io"
        
        r = residuals[io, :]          # R-squared criterion for model "io"
        m = modelY[io, :]             # computed output data for model "io"
        d = dataY[io, :]              # measured output data for model "io"
        R2 = 1 - (np.linalg.norm(r) / np.linalg.norm(m - np.mean(m)))**2  # R-squared
        
        # R-squared criterion adjusted for the amount of data and number of coefficients
        R2adj[io] = ((mData - 1) * R2 - nTerm) / (mData - nTerm)
        
        # correlation between model "io" and the data
        MDc = np.corrcoef(np.vstack([d, m]))
        MDcorr[io] = MDc[0, 1]
        
        # standard error of each coefficient for model "io"
        BtB_inv_diag = np.diag(np.linalg.inv(B[io].T @ B[io]))
        Std_Err_Coeff = np.sqrt((r @ r.T) * BtB_inv_diag / (mData - nTerm))
        
        # coefficient of variation of each coefficient for model "io"
        coeffCOV[io] = Std_Err_Coeff / ( np.abs(coeff[io].flatten()) + 1e-6 )
        coeffCOV[io][np.abs(coeff[io]) < 1e-6] = 1e-3
        
        AIC[io] = 0   # add AIC here
    
    if figNo:
        plt.ion() # interactive mode: on
        cMap = rainbow(nOut)
        plt.figure(figNo)
        format_plot(16, 1, 3)
        ax = [ np.min(dataY), np.max(dataY), np.min(dataY), np.max(dataY) ]
        plt.clf()
        for io in range(nOut):
            plt.plot(modelY[io, :], dataY[io, :], 'o', color=cMap[io, :])
        plt.plot([ax[0], ax[1]], [ax[2], ax[3]], '-k', linewidth=0.5)
        plt.axis('square')
        plt.xlabel('Y model')
        plt.ylabel('Y data')
        
        tx = 0.00
        ty = 1.0 - 0.05 * (io+1)
        plt.text(tx*ax[1] + (1-tx)*ax[0], ty*ax[3] + (1-ty)*ax[2],
                f'{nTerm} model terms')
        
        for io in range(nOut):
            tx = 0.55
            ty = 0.5 - 0.2 * (io+1)
            plt.text(tx*ax[1] + (1-tx)*ax[0], ty*ax[3] + (1-ty)*ax[2],
                    f'ρ_{{x,y{io}}} = {MDcorr[io]:.3f}', color=cMap[io, :])
#                   '$\rho_{{x,y{io}}}$ = {MDcorr[io]:.3f}', color=cMap[io, :])
            ty = 0.5 + 0.4 * (io+1)
            plt.text(tx*ax[1] + (1-tx)*ax[0], ty*ax[3] + (1-ty)*ax[2],
                    f'{txt}', color=cMap[io, :])
        
        plt.show(block=False)
    
    return MDcorr, coeffCOV, R2adj, AIC


def print_model_stats(iter, coeff, order, coeffCOV, MDcorr, R2adj, scaling, maxCull):
    '''
    print_model_stats( iter, coeff, order, coeffCOV, MDcorr, R2adj, scaling, maxCull )
    Print model statistics during iterative culling process
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    iter        current iteration number                                  1 x 1
    coeff       list of coefficient vectors of each model            {nTerm x 1}
    order       order of each explanatory variable in each term      {nTerm x nInp}
    coeffCOV    list of coefficient of variation of each coefficient {1 x nTerm}
    MDcorr      model-data correlation for each output               nOut x 1
    R2adj       adjusted R-squared for each output                   nOut x 1
    scaling     scaling option used                                       1 x 1
    maxCull     maximum number of culling iterations                      1 x 1
    '''
    
    nOut = len(order)
    nTerm, nInp = order[0].shape
    
    print(f' model culling iteration {iter + 1}')
    for io in range(nOut):
        print(f'  Output {io + 1} ------------------------------------------')
        print(f'  Scaling Option    = {scaling}')
        print('  Response Surface Coefficients')
        
        # Print header
        header = '    i  '
        for ii in range(nInp):
            if scaling > 0:
                header += f' z{ii+1:02d} '
            else:
                header += f' x{ii+1:02d} '
        header += '   coeff    C.O.V\'s'
        print(header)
        
        # Print each term
        for it in range(nTerm):
            line = f'  {it + 1:3d} '
            for ii in range(nInp):
                line += f' {order[io][it, ii]:2d}  '
            line += f' {coeff[io][it]:8.4f}  {coeffCOV[io][it]:8.4f}'
            print(line)
        
        print()
        print(f'  scaling option            = {scaling:3d}')
        print(f'  Total Number of Terms     = {nTerm * nOut:3d}')
        print(f'  Adjusted R-square {io + 1}       = {R2adj[io]:6.3f}')
        print(f'  model-data correlation {io + 1}  = {MDcorr[io]:6.3f}')
    
    # Estimate time remaining
    # import time
    
    # Calculate estimated time remaining
    # Note: In Python, we'd need to track elapsed time separately
    # For now, provide a simplified version
    if iter > 0:
        # This would need elapsed time from start to work properly
        # For now, just print a separator
        print('  ==================================')
    else:
        print('  ==================================')


def cull_model(coeff, order, coeffCOV, tol):
    '''
    [ order, nTerm, coeffCOV ] = cull_model( coeff, order, coeffCOV, tol )
    remove the term from the model that has the largest coeffCOV
    
    INPUT       DESCRIPTION                                         DIMENSION
    --------    -------------------------------------------------   ---------
     coeff      list of coefficient vectors of each model         {nTerm x 1}  
     order      order of each explanatory variable in each term   {nTerm x nInp}
    coeffCOV    list of coefficient of variation of
                each model coefficient of each model               {1 x nTerm}
     tol        tolerance for an acceptable coeffCOV                1 x 1
    
    OUTPUT      DESCRIPTION                                         DIMENSION
    --------    -------------------------------------------------   ---------
     order      retained order of each explanatory variable in each term  {nTerm x nInp}
     nTerm      number of terms in each polynomial model                  1 x nOut
    coeffCOV    list of coefficient of variation of
                each model coefficient of each culled model               {1 x nTerm}
    '''
    
    nOut = len(order)
    nTerm = np.zeros(nOut, dtype=int)
    
    for io in range(nOut):
        
        nTerm[io], nInp = order[io].shape
        
        # model coefficient with largest coefficient of variation
        ic = np.argmax(coeffCOV[io])
        max_cov = coeffCOV[io][ic]
        
        # remove the 'ic-th' term from the model for output io
        # Create index arrays excluding ic
        keep_idx = np.concatenate([np.arange(ic), np.arange(ic + 1, nTerm[io])])
        
        order[io] = order[io][keep_idx, :]
        coeff[io] = coeff[io][keep_idx]
        coeffCOV[io] = coeffCOV[io][keep_idx]
        
        nTerm[io] = nTerm[io] - 1
    
    return order, nTerm, coeffCOV


def IDWinterp(Zx, Zy, zMap, p, k, tol):
    '''
    Inverse Distance Weighting interpolation
    NOTE: This function is used only by polynomial_orders, which is currently unused.
    '''
    pass


# updated 2006-01-29, 2007-02-21, 2007-03-06, 2009-10-14, 2022-11-19 2023-02-27, 2023-05-31 2025-11-07

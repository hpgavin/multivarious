import numpy as np
import matplotlib.pyplot as plt

def mimo_rs(dataX, dataY, max_order=2, pTrain=70, scaling=1, L1_pnlty=1.0, basis_fctn='H', var_names=None ):
    '''
    [ ordr, coeff, meanX, meanY, invTX, TY, testModelY, testX, testY ] = mimo_rs( dataX, dataY, max_order, pTrain, scaling, L1_pntly, basis_fctn, var_names=None )
    
    mimo_rs: multi-input multi-output response surface 
    
    This program fits a polynomial to multidimensional data 
    by projecting the data onto a polynomial basis of oder up to 10 (or more)
    Data may be scaled, standardized or decorrelated before fitting. 
    The polynomial basis may be Hermite, Legendre, or Power polynomials. 
    The model complexity is managed via L1 regularization. 
    
     mimo_rs approximates data with a polynomial of arbitrary order,

       y(X) = a + sum_i=1 ^n sum j=1^k_i b_ij X_i^j + 
                  sum q=1 ^m c_q prod i=1 ^n X_i^{p_iq}.
    
    INPUT       DESCRIPTION                                              DEFAULT
    --------    -------------------------------------------------------- -------
    dataX       m observations of nx input  features in a (nx x m) matrix
    dataY       m observations of ny output features in a (ny x m) matrix
    max_order   maximum allowable polynomial order                          3
    pTrain      percentage of data for training (remaining for testing)    50
    scaling     scale the X data and the Y data before fitting             [1,1] 
                scaling = 0 : no scaling
                scaling = 1 : subtract mean and divide by std.dev
                scaling = 2 : subtract mean and decorrelate
                scaling = 3 : log-transform, subtract mean and divide by std.dev
                scaling = 4 : log-transform, subtract mean and decorrelate
    L1_pnlty    penalty coefficient for L1 regularization                   1.0
    basis_fctn  basis function type                                         'H'
                'H': Hermite functions
                'L': Legendre polynomials
                'P': Power polynomials
    var_names   optional dictionary with keys 'X' and 'Y' containing
                lists of variable names for labeling                       None
    
    
    OUTPUT      DESCRIPTION
    --------    --------------------------------------------------------
     ordr       list of matrices of the orders of variables in each polynomial term 
     coeff      list of polynomial coefficients 
     meanX      mean vector of the scaled dataX
     meanY      mean vector of the scaled dataY
     invTX      inverse transformation matrix from dataX to dataZx (full col rank)
     TY         transformation matrix from dataZy to dataY (full row rank)
     testX      input  features for model testing 
     testY      output features for model testing 
     testModelY output features for model testing
    
    Reference:
       Gavin, HP and Yau SC, "High order limit state functions in the 
       response surface method for structural reliability analysis,"
       submitted to Structural Safety, December 2005.     
    
    Department of Civil and Environmental Engineering
    Duke University
    Siu Chung Yau, Henri P. Gavin, January 2006, 2023
    '''

    print('\n Multi-Input Multi-Output Response Surface (mimo_rs)\n')

    # Handle default arguments and convert to appropriate types
    max_order = int(round(abs(max_order)))
    pTrain = abs(pTrain) / 100 if pTrain > 1 else abs(pTrain)
    scaling[0] = int(round(abs(scaling[0])))
    scaling[1] = int(round(abs(scaling[1])))
    L1_pnlty = abs(L1_pnlty)

    scaling_names = {
         0: "no scaling",
         1: "standardization (mean=0, std=1)",
         2: "decorrelation", 
         3: "log transform and standardization", 
         4: "log transform and decorrelation" }
   
    if not np.isfinite(dataX).all():
        print(' mimo_rs: dataX has infinite or NaN values\n\n')
        exit(100)

    if not np.isfinite(dataY).all():
        print(' mimo_rs: dataY has infinite or NaN values\n\n')
        exit(100)

    nInp, mDataX = dataX.shape   # number of columns in dataX is mData
    nOut, mDataY = dataY.shape   # number of columns in dataY is mData

    print(f'   {min(mDataX, mDataY):6d} data values\n') 

    print(f' {np.min(dataX):11.6f} < X < {np.max(dataX):11.6f} ')
    print(f' {np.min(dataY):11.6f} < Y < {np.max(dataY):11.6f} \n')
    print(f'    X scaling option {scaling[0]}: {scaling_names[scaling[0]]} ')
    print(f'    Y scaling option {scaling[1]}: {scaling_names[scaling[1]]} \n')

    if mDataX != mDataY:
        raise ValueError('the dataX and dataY matrices must have the same number of columns')
    else:
        mData = min(mDataX,mDataY)

    scatter_data(dataX, dataY, figNo=100, var_names = var_names )

    # scale data matrices for X (explanatory) and Y (dependent) variables
    # separately since using the covariance between X and Y in the model
    # is "cheating"
    Zx, meanX, TX, invTX, RX, minZx, maxZx = scale_data(dataX, scaling[0])
    Zy, meanY, TY, invTY, RY, minZy, maxZy = scale_data(dataY, scaling[1])
    
    # print(f'  {minZx:.6f} < Zx < {maxZx:.6f}')
    # print(f'  {minZy:.6f} < Zy < {maxZy:.6f}')
    # print(f' shape Zx = {Zx.shape}')
    # print(f' shape Zy = {Zy.shape}')

    nZx = TX.shape[1]
    nZy = TY.shape[1]

    XY = np.vstack([Zx , Zy])
    # print(f' 1 shape XY = {XY.shape}')

    # remove columns of  Zx and Zy containing outliers
    XY = clip_data( XY, -1e0 , 1e0 ) 

    # print(f' 2 shape XY = {XY.shape}')

    mData = XY.shape[1]
    Zx = XY[:nZx, :]
    Zy = XY[nZx:nZx+nZy, :]

    # print(f' shape Zx = {Zx.shape}')
    # print(f' shape Zy = {Zy.shape}')

    # re-scale the data ??
    '''
    dataX = descale_data(Zx, meanX, TX, scaling[0])
    dataY = descale_data(Zy, meanY, TY, scaling[1])
    Zx, meanX, TX, invTX, RX, minZx, maxZx = scale_data(dataX, scaling[0])
    Zy, meanY, TY, invTY, RY, minZy, maxZy = scale_data(dataY, scaling[1])
    '''

    print(f'   {Zy.shape[1]:6d} data values\n') 
    print(f'  {minZx:.6f} < Zx < {maxZx:.6f}')
    print(f'  {minZy:.6f} < Zy < {maxZy:.6f}')
    print(f'\n  dataX correlation matrix  ')
    print(np.round(RX,2))
    print(f'\n  dataY correlation matrix  ')
    print(np.round(RY,2))

    if np.any(scaling == 2) or np.any(scaling == 4):

        xNames = [rf"$zX_{i+1}$" for i in range(nZx)]
        yNames = [rf"$zY_{i+1}$" for i in range(nZy)]
        z_names = { 'X': xNames  , 'Y': yNames }
        scatter_data(Zx, Zy, figNo=101, var_names = z_names)

    #import time as time
    #from datetime import datetime, timedelta
    #time.sleep(1) # if needed for debugging

    trainZx, trainZy, mTrain, testZx, testZy, mTest = split_data(Zx, Zy, pTrain)

    trainX = descale_data(trainZx, meanX, TX, scaling[0])
    trainY = descale_data(trainZy, meanY, TY, scaling[1])
    testX  = descale_data( testZx, meanX, TX, scaling[0])
    testY  = descale_data( testZy, meanY, TY, scaling[1])
    
    ordr, nTerm = mixed_term_orders(max_order, nZx)

    trainMDcorr = np.full((nOut, 1), np.nan)
    testMDcorr  = np.full((nOut, 1), np.nan)
    
    # start a timer to measure computational time
    #start_time = time.time()
    
    # fit ("train") a separate model for each output (dependent) variable
    coeff, condB = fit_model(trainZx, trainZy, ordr, nTerm, mTrain, L1_pnlty, basis_fctn)
        
    # compute the model for the training data and the testing data
    trainModelY, B = compute_model(ordr, coeff, meanX, meanY, invTX, TY, trainX, scaling, basis_fctn)
    testModelY, _  = compute_model(ordr, coeff, meanX, meanY, invTX, TY, testX,  scaling, basis_fctn)

    # evaluate the model for the training data and the testing data
    trainMDcorr, coeffCOV, _, _ = evaluate_model(B, coeff, trainY, trainModelY)
    testMDcorr, _, R2adj, AIC   = evaluate_model(B, coeff,  testY,  testModelY) 
        
    print_model_stats(coeff, ordr, coeffCOV, testMDcorr, R2adj, AIC, scaling, scaling_names)
        
    visualize_model_performance(trainY, trainModelY, 'training')
    visualize_model_performance( testY,  testModelY, 'testing')
    
    return ordr, coeff, meanX, meanY, invTX, TY, testX, testY, testModelY


def scatter_data(dataX, dataY, figNo=100, var_names=None):
    '''
    scatter_data(dataX, dataY, figNo, var_names)
    Create scatter plots of each pair of input and output variables
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    dataX       matrix of input data                                   nInp x m
    dataY       matrix of output data                                  nOut x m
    figNo       figure number for plotting (default = 100)             1 x 1
    var_names   optional dictionary with keys 'X' and 'Y' containing
                lists of variable names for labeling                   dict
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    None        Creates matplotlib figure with scatter plot matrix
    
    NOTE: For high-dimensional data, this can create many subplots.
          The function creates a grid showing:
          - X vs X correlations in upper left
          - Y vs X correlations in lower left and upper right
          - Y vs Y correlations in lower right
    '''
    nInp, m = dataX.shape
    nOut, _ = dataY.shape

    #print(var_names)
    
    # Set up variable names if not provided
    if var_names is None:
        xNames = [rf"$x_{i+1}$" for i in range(nInp)]
        yNames = [rf"$y_{i+1}$" for i in range(nOut)]
    else:
        xNames = var_names.get('X') #, [rf"$zX_{i+1}$" for i in range(nInp)])
        yNames = var_names.get('Y') #, [rf"$zY_{i+1}$" for i in range(nOut)])
        #xNames = [rf"$zX_{i+1}$" for i in range(nInp)]
        #yNames = [rf"$zY_{i+1}$" for i in range(nOut)]

    # Calculate number of subplots needed
    nTotalVars = nInp + nOut
    
    # Create figure with subplots
    plt.ion() # interactive mode: on
    fig = plt.figure(figNo, figsize=(2*nTotalVars, 2*nTotalVars))
    plt.clf()
    
    plotIndex = 1
    
    # Create scatter plots for all pairs
    for iRow in range(nTotalVars):
        for iCol in range(nTotalVars):
#           print(f' iRow = {iRow}   iCol = {iCol}')

            ax = plt.subplot(nTotalVars, nTotalVars, plotIndex)
            
            # Determine which data to plot
            if iRow < nInp and iCol < nInp:
                # X vs X
                xData = dataX[iCol, :]
                yData = dataX[iRow, :]
                xLabel = xNames[iCol]
                yLabel = xNames[iRow]
                color = 'navy'
                
            elif iRow >= nInp and iCol < nInp:
                # Y vs X
                xData = dataX[iCol, :]
                yData = dataY[iRow - nInp, :]
                xLabel = xNames[iCol]
                yLabel = yNames[iRow - nInp]
                color = 'darkcyan'
                
            elif iRow >= nInp and iCol >= nInp:
                # Y vs Y
                xData = dataY[iCol - nInp, :]
                yData = dataY[iRow - nInp, :]
                xLabel = yNames[iCol - nInp]
                yLabel = yNames[iRow - nInp]
                color = 'darkgreen'
                
            elif iRow < nInp and iCol >= nInp:
                # X vs Y (upper right - leave empty or skip)
                yData = dataX[iRow, :]
                xData = dataY[iCol - nInp, :]
                yLabel = xNames[iRow]
                xLabel = yNames[iCol - nInp]
                color = 'darkcyan'
            # else:  
                # plotIndex += 1
                # continue
            
            # Create scatter plot
            if iRow == iCol:
                # Diagonal: plot histogram instead of scatter
                ax.hist(xData, bins=20, color=color, alpha=0.7, edgecolor='black')
                # ax.set_ylabel('Count')
            else:
                # Off-diagonal: scatter plot
                ax.plot(xData, yData, 'o', color=color, markersize=2, alpha=0.5)
            
            # Add labels only on edges
            if iRow == nTotalVars - 1:
                ax.set_xlabel(xLabel, fontsize=16) #, fontweight='bold')
            else:
                ax.set_xticklabels([])
            
            if iCol == 0:
                ax.set_ylabel(yLabel, fontsize=16) #, fontweight='bold')
            else:
                ax.set_yticklabels([])
            
            # Make tick labels smaller
            ax.tick_params(labelsize=6)
            
            plotIndex += 1
    
    plt.tight_layout()
#   plt.suptitle('Pairwise Scatter Plots', fontsize=14, y=1.0)
    plt.show(block=False)
    
    return None


def scale_data(Data, scaling):
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
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Z           scaled data                                           n x m
    meanD       the arithmetic mean of the (log-transformed) data     n x 1
    T           transformation matrix with full column rank           n x <n?
    invT        inverse of the transformation matrix, full row rank <n? x n
    R           data correlation matrix                               n x n
    maxZ        maximum value of each data sample                     1 x n
    minZ        minimum value of each data sample                     1 x n
    '''
    
    n, m = Data.shape  # m observations of n variables
    
    T    = np.eye(n)
    invT = np.eye(n) 
    R    = np.eye(n)

    if scaling in [ 3 , 4 ]:  # log-transform
        if np.any(Data <= 0):
            print('  mimo_rs: Data has negative values, can not log-transform\n\n')
            exit(300)
        Data = np.log(Data);

    meanD   = np.mean(Data, axis=1, keepdims=True)
    covData =  np.cov(Data, ddof=1) # data covariance

    if not np.isfinite(covData).all():
        print('  mimo_rs: data covariance has infinite or NaN values \n\n')
        exit(200)

    if scaling <= 0: # no scaling
        meanD = np.zeros([n,1])
    
    if scaling == 1 or scaling == 3: # subtract mean and divide by std.dev
        if n > 1:
            T = np.diag(np.sqrt(np.diag(covData)))
            invT = np.diag(np.sqrt(1/np.diag(covData)))
    
    if scaling == 2 or scaling == 4: # subtract mean and decorrelate
        if n > 1:
            eVal, eVec = np.linalg.eigh(covData)      # eig decomp of symm matx
            idx = eVal > 1e-6*max(eVal)               # +'ve eigenvals
            T = (eVec[:,idx]) @ np.sqrt(np.diag(eVal[idx]))
            invT = np.diag(np.sqrt(1.0 / eVal[idx])) @ eVec[:,idx].T
            nCut = np.sum(eVal < 1e-6*max(eVal))
            if nCut > 0:
                print(f'  eVal ratio = {np.max(eVal)/np.min(eVal)}')
                print(f'  Basis reduced by {nCut} dimension(s)')
                print(f'  eVal ratio = {np.max(eVal)/eVal[nCut]}\n')
                #print(f'eVal = {eVal} , idx = {idx}')
                #print(f'T = {T} , invT = {invT} ,  I2 = {invT @ T}')
    
    # correlation matrix
    if n > 1:
        R = np.corrcoef(Data)
    
    Z = invT @ ( Data - meanD )

    #print(f' shapeZ = {Z.shape}')
    #print('T')
    #print(T)
    #print('invT')
    #print(invT)

    maxZ = np.max(Z)
    minZ = np.min(Z)
    
    return Z, meanD, T, invT, R, minZ, maxZ


def descale_data(Z, meanD, T, scaling ):
    '''
     Data  = scale_data(Z, meanD,  scaling)
     descale data in one of four or five ways ..  Z = inv(T)*(Data - meanD); 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Z           scaled data                                           n x m
    meanD       the arithmetic mean of the (log-transformed) data     n x 1
    T           transformation matrix                                 n x n
    scaling     type of scaling ...                                   1 x 1
                 scaling = 0 : no scaling
                 scaling = 1 : subtract mean and divide by std.dev
                 scaling = 2 : subtract mean and decorrelate
                 scaling = 3 : log-transform, subtract mean and divide by std.dev
                 scaling = 4 : log-transform, subtract mean and decorrelate
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Data        a matrix of data values                               n x m
    '''
    n, m = Z.shape  # m observations of n variables

    # print(f' descale shape  Z = {Z.shape}')
    # print(f' descale shape  T = {T.shape}')

    # apply the scaling: Data = T * Z + meanD
    
    if scaling <= 0:  # no scaling
        Data = Z.copy()

    if scaling == 1 or scaling == 2:  
        Data = T @ Z + meanD; 

    elif scaling == 3 or scaling == 4: 
        Data = np.exp(T @ Z + meanD); 
    
    return Data


def clip_data(Data, low_limit, high_limit):
    '''
     clip_data(Data, low_limit, high_limit)
     remove outliers from standardized data (zero mean, unit variance)
     using Chauvenet's criterion
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    Data        matrix of data                                          n x m  
    low_limit   remove values lower  than low_limit *Chauvenet crit     1 x 1
    high_limit  remove values higher than high_limit*Chauvenet crit     1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     Data       matrix of data without values exceeding given limits    n x m'
    '''
        
    mData = Data.shape[1]

    Chauvenet_criterion = 0.8 + 0.4*np.log(mData)  # an approximation
    idxc = np.where(np.any( np.abs(Data) > Chauvenet_criterion, axis=0 ))[0]
    outliers = idxc.shape[0]
    print(f'  Chauvenet outlier criterion = {Chauvenet_criterion:.2f}')  
    print(f'  number of outliers = {outliers} = {100*outliers/mData:.2f} percent of the data')

    if outliers > 0:
        low_limit  =  low_limit * Chauvenet_criterion
        high_limit = high_limit * Chauvenet_criterion
 
        # Keep only columns where all values are greater than low_limit
        idxc = np.where(np.all(Data > low_limit, axis=0))[0]
        Data = Data[:, idxc]
    
        # Keep only columns where all values are less than high_limit
        idxc = np.where(np.all(Data < high_limit, axis=0))[0]
        Data = Data[:, idxc]
    
    return Data 


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
    
    testX = dataX[:, idtestX]
    testY = dataY[:, idtestX]
    
    return trainX, trainY, mTrain, testX, testY, mTest


def mixed_term_orders( max_order, nZx ):
    '''
    [ ordr , nTerm ] = mixed_term_orders( max_order, nZx )
    specify the exponents on each input variable for every term in the model,
    and the total number of terms, nTerm
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    max_order    maximum polynomial order of the model                   1 x 1
    nZx         number of indepenent input  (explanatory) variables     1 x 1
    nZy         number of indepenent output  (dependent)  variables     1 x 1
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    ordr        the orders of each independeht explanatory variable 
                in each polynomial term                           {nTerm x nInp}
    nTerm       number of polynomial terms in the model             nOut x 1
    
    The matrix 'order' indicates which mixed term power-products are present
    in each term of the model. 
    
    Each element in this matrix represents the order of a variable in a term.
    The columns of 'order' correspond to individual variables. 
    Each row indicates the powers present in the product of a term.
    
    Algorithm by Siu Chung Yau (2006)
    '''
    
    print(f'\nDetermine the Mixed Term Power Products ...')
    
    nTerm = int(np.prod(max_order * np.ones(nZx, dtype=int)+1))

    # print(f'nTerm = {nTerm}')
    
    ordr = np.zeros((nTerm, nZx), dtype=int) # allocate memory for the 'order' matrix
    term = np.zeros(nZx + 1, dtype=int)            # orders in a given term (extra element for carry)
    term[0] = -1                                   # starting value for the first term
    
    for t in range(nTerm):                         # loop over all terms
        term[0] = term[0] + 1                      # increment order of the first variable
        for v in range(nZx):                       # check every column in the row
            if term[v] > max_order:
                term[v] = 0
                term[v + 1] = term[v + 1] + 1      # increment columns as needed
        
        ordr[t, :] = term[:nZx]  # save the orders of term t in the 'order' matrix
    
    # The power of a variable in each term can not be greater than 
    # the order of that variable alone.  
    # Remove the terms in which the total order is larger than 
    # the highest order term.
    
    # print(ordr)

    it = np.where(np.sum(ordr, axis=1) <= max_order)[0]
    ordr = ordr[it, :]
    nTerm = ordr.shape[0]
    
    # print(f'nTerm = {nTerm}')
    # print(ordr)
    
    print(f'  Total Number of Terms: {nTerm:3d}')
    print(f'  Number of Mixed Terms: {( nTerm - nZx*max_order - 1 ):3d}\n')
    
    return ordr, nTerm


def build_model_basis(Zx, ordr, basis_fctn='H'):
    '''
    B = build_model_basis( Zx, ordr, basis_fctn )
    compute matrix of model basis vectors
    options: power-polynomial basis, Hermite function basis, or Legendre basis

    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------  -----------
     Zx         matrix of input (explanatory) variables              nZx x mData
     ordr       orders for each variable in each polynomial term   nTerm x nZx
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial

    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------  -----------
      B         matrix basis vectors for the polynomial model       mData x nTerm
    '''

    mData = Zx.shape[1]             # number of data points
    nTerm, nZx = ordr.shape       # number of terms, inputs, outputs 
    B = np.ones((mData, nTerm))     # the matrix of model basis vectors 

    max_order = int(np.max(ordr))

    # in the matrix of basis vectors, B, 
    # columns correspond to each term in the polynomial and 
    # rows correspond to each observation 
 
    if basis_fctn == 'P':
        # Power polynomials
        for it in range(nTerm):
            B[:, it] = np.prod(Zx.T ** ordr[it, :], axis=1)
    else:
        # Legendre or Hermite
        for it in range(nTerm):
            B[:, it] = polynomial_product(ordr[it, :], Zx.T, max_order, basis_fctn)

    return B

   
def polynomial_product(ordr, Zx, max_order, basis_fctn='H'):
    '''
    psyProduct = polynomial_product( ordr, Zx, basis_fctn )
    compute the product of hermite functions of given orders (from 0 to 5)
    for a set of column vectors Z, where each column of Zx has a given order
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    ordr        expxonents present in each polynomial term              1 x nZx
     Zx         matrix of scaled input (explanatory) variables      mData x nZx
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    psyProduct  vector of product of hermite polynomials             mData x 1
    '''
    
    nZx  = len(ordr) # number of independent input (explanatory) variables
    psyProduct = np.ones(Zx.shape[0])  # initialize to vector of 1

    for i in range(nZx):
        if ordr[i] > 0:
            if basis_fctn == 'H':
                psyProduct *= hermite( ordr[i], Zx[:, i], max_order )
            elif basis_fctn == 'L':
                psyProduct *= legendre( ordr[i], Zx[:, i] )
    
    return psyProduct


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

    if n == 0:
        psy = np.exp(-(z/(N+2))**(6*N))
    elif n == 1:
        psy = (z/(N+2)) * np.exp(-(z/(N+2))**(6*N)) 
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


def fit_model(Zx, Zy, ordr, nTerm, mData, L1_pnlty, basis_fctn):
    '''
    [ coeff , condB ] = fit_model( Zx, Zy, ordr, nTerm, mData, L1_pnlty, basis_fctn )
    via singular value decomposition without regularization, or
    via quadratic programming with L1 regularization
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     Zx         scaled input (explanatory) data                      nx x mData
     Zy         scaled output (dependent) data                        1 x mData
     ordr       order of each explanatory variable in each term   nTerm x nx
     nTerm      number of terms in the polynomial model                  1 x 1
     mData      number of data points (not used, kept for compatibility) 1 x 1
     L1_pnlty   L1 regularization coefficient                            1 x 1
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial

    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     coeff      vector of model coefficients                           nTerm x 1  
     condB      condition number of the model basis                     1 x 1
    '''
    
    #print(f'Zy dim = {Zy.shape}')
    nZy = Zy.shape[0]
    coeff = np.zeros([nTerm,nZy])

    print(f'Fit The Model ... with {basis_fctn} polynomials and L1_pnlty = {L1_pnlty}')

    B = build_model_basis( Zx, ordr, basis_fctn )

    condB = np.linalg.cond(B)

    print(f'  Basis matrix condition number = {condB:8.2e}')
    if condB > 1e20:
        print(f'  ... is just too much - exiting \n\n')
        exit(100)
    elif condB > 1e12:
        print(f'  ... you are in serious trouble \n\n')
    elif condB > 1e6:
        print(f'  ... you are in trouble \n\n')
    elif condB > 1e2:
        print(f'  ... it could work \n\n')
    elif condB < 1e2:
        print(f'  ... looking good! \n\n')


    if L1_pnlty > 0:
        # Use L1_fit for regularization
        try:
            from multivarious.fit.L1_fit import L1_fit
            from multivarious.utl.L1_plots import L1_plots

            for io in range(nZy):
                Zy_col = Zy[io,:].T # Zy needs to be column vector for L1_fit

                #print(f'Zy_col_dim = {Zy_col.shape}')
                coeff[:,io], mu, nu, cvg_hst = L1_fit(B, Zy_col, L1_pnlty, w=0)

                # Optional: plot L1 convergence
                L1_plots(B, coeff[:,io], Zy_col, cvg_hst, L1_pnlty, 0, fig_no=700+10*io)

        except ImportError:
            print('WARNING: L1_fit not found, using OLS instead')
            coeff = np.linalg.lstsq(B, Zy, rcond=None)[0]
    else:
        # Use ordinary least squares / SVD
        coeff = np.linalg.lstsq(B, Zy, rcond=None)[0]

    return coeff, condB


def compute_model(ordr, coeff, meanX, meanY, invTX, TY, dataX, scaling, basis_fctn='H'):
    '''
    [ modelY, B ] = compute_model(ordr, coeff, meanX, meanY, invTX, TY, dataX, scaling,basis_fctn)
    compute a multivariate polynomial model 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     ordr       list of ordr of each explanatory variable in each term {nTerm x nZx}
     coeff      list of model coefficient vectors                 {nTerm x 1}
     meanX      mean of pre-scaled input  (explanatory) variables   nInp x 1 
     meanY      mean of pre-scaled output  (dependent)  variables   nOut x 1 
     invTX      inverse transformation matrix for input variables  <nZx? x nZx
     TY         transformation matrix for output variables           nZy x <nZy?
     dataX      input data to evaluate model on                     nInp x mData
     scaling    scaling type ... see scale_data function               1 x 1
     basis_fctn 'H': Hermite, 'L': Legendre, 'P': Power polynomial
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    modelY      computed model                                        nOut x mData
     B          list of basis vector matrices of the computed model  {mData x nTerm}
    '''
    
    nInp, mData = dataX.shape        # number of columns in dataX is mData
    nOut = len(meanY)                # number of output variables
    modelZy = np.full((mData, nOut), np.nan)  # initialize model output
    
    # Transform input data according to scaling
    if scaling[0] == 0:
        dataZx = dataX
    
    elif scaling[0] in [1, 2]:
        dataZx = invTX @ ( dataX - meanX )
    
    elif scaling[0] in [3, 4]:
        logX = np.log(dataX)
        dataZx = invTX @ ( logX - meanX )  # standard normal variables
    
    # Compute model for each output
    B = build_model_basis(dataZx, ordr, basis_fctn)
    for io in range(nOut):
        modelZy[:,io] = B @ coeff[:,io]
    
    # Inverse transform to original scale
    if scaling[1] == 0:
        modelY = modelZy.T
    
    elif scaling[1] in [1, 2]:
        modelY = TY @ modelZy.T + meanY
    
    elif scaling[1] in [3, 4]:
        modelY = np.exp(TY @ modelZy.T + meanY)
    
    return modelY, B


def evaluate_model(B, coeff, dataY, modelY):
    '''
    [ MDcorr, coeffCOV , R2adj, AIC ] = evaluate_model( B, coeff, dataY, modelY)
    evaluate the model statistics 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     B          list of basis vector matrices of each model        mData x nTerm
     coeff      list of coefficient vectors of each model          nTerm x 1  
     dataY      output (dependent) data                             nOut x mData
     modelY     model predictions                                   nOut x mData
    
    OUTPUT      DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    MDcorr      model-data correlation                                 1 x nOut
    coeffCOV    list of coefficient of variation of
                each model coefficient of each model               nTerm x nOut
    R2adj       adjusted R-squared for each model                      1 x nOut
    AIC         Akaike information criterion for each model            1 x nOut
    '''
    
    nTerm = B.shape[1]
    nOut, mData = dataY.shape
    MDcorr = np.zeros(nOut)
    R2adj = np.zeros(nOut)
    AIC = np.zeros(nOut)
    coeffCOV = np.zeros([nTerm,nOut])  # coefficient of variation of the model coefficients
    
    # statistical analysis of coefficients
    residuals = dataY - modelY    # matrix of residuals for each model
    
    BtB_inv_diag = np.diag(np.linalg.inv(B.T @ B))

    for io in range(nOut):
        
        nc = sum( np.abs(coeff[:,io]) > 1e-3)
        r = residuals[io, :]          # R-squared criterion for model "io"
        m = modelY[io, :]             # computed output data for model "io"
        d = dataY[io, :]              # measured output data for model "io"
        Vr = residuals @ residuals.T / mData   # covariance of the residuals
        R2 = 1 - (np.linalg.norm(r) / np.linalg.norm(m - np.mean(m)))**2  # R-squared
        
        # R-squared criterion adjusted for the amount of data and number of coefficients
        R2adj[io] = ((mData - 1) * R2 - nTerm) / (mData - nTerm)
        
        # correlation between model "io" and the data
        MDc = np.corrcoef(np.vstack([d, m]))
        MDcorr[io] = MDc[0, 1]
        
        # standard error of each coefficient for model "io"
        Std_Err_Coeff = np.sqrt((r @ r.T) * BtB_inv_diag / (mData - nTerm))
        
        # coefficient of variation of each coefficient for model "io"
        coeffCOV[:,io] = Std_Err_Coeff / ( np.abs(coeff[:,io].flatten()) + 1e-6 )
        coeffCOV[ np.abs(coeff[:,io]) < 1e-6 , io ] = 1e-4
        
        AIC[io] = np.log(2*np.pi*nc*Vr[io,io]) + nc*Vr[io,io] + 2*nc 

    #print(f'nOut = {nOut}')
    #print('coeffCOV')
    #print(coeffCOV)
    
    return MDcorr, coeffCOV, R2adj, AIC


def visualize_model_performance(dataY, modelY,  txt):
    '''
    Create visualization of model performance

    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
     dataY      output (dependent) data                               nOut x mData
     modelY     model predictions                                     nOut x mData
       txt      annotation text
    '''

    nOut = dataY.shape[0]
    
    plt.ion() # interactive mode: on 
    fig, axes = plt.subplots(1, nOut, figsize=(6*nOut, 5))
    if nOut == 1:
        axes = [axes]
    
    for io in range(nOut):
        ax = axes[io]
        
        # Scatter plot
        ax.scatter(dataY[io, :], modelY[io, :], alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(np.min(dataY[io, :]), np.min(modelY[io, :]))
        max_val = max(np.max(dataY[io, :]), np.max(modelY[io, :]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2) 
        #       label='Perfect prediction')
        
        # Correlation
        corr = np.corrcoef(dataY[io, :], modelY[io, :])[0, 1]
        
        ax.set_xlabel(f'{txt} output data', fontsize=12)
        ax.set_ylabel(f'{txt} model output', fontsize=12)
        ax.set_title(f'Output {io+1}:  correlation: {corr:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'{txt}_performance.png', dpi=150, bbox_inches='tight')
    print(f"Performance plot saved to '{txt}_performance.png'")
    
    return


def print_model_stats(coeff, order, coeffCOV, MDcorr, R2adj, AIC, scaling, scaling_names ):
    '''
    print_model_stats( coeff, order, coeffCOV, MDcorr, R2adj, scaling )
    Print model statistics 
    
    INPUT       DESCRIPTION                                           DIMENSION
    --------    ---------------------------------------------------   ---------
    coeff       list of coefficient vectors of each model            nTerm x 1
    order       order of each explanatory variable in each term      nTerm x nZx
    coeffCOV    list of coefficient of variation of each coefficient  1 x nTerm
    MDcorr      model-data correlation for each output                1 x nOut 
    R2adj       adjusted R-squared for each output                    1 x nOut 
    AIC                                                               1 x nOut
    scaling     scale the X data and the Y data before fitting        1 x 2
    scaling_names short description of each scaling type   
    '''
    
    nTerm, nZx = order.shape
    nOut = coeff.shape[1]
    
    for io in range(nOut):
        print(f'\n  Output {io + 1} ------------------------------------------')
        print('  Response Surface Coefficients')
        
        # Print header
        header = '    k  '
        for ii in range(nZx):
            if scaling[0] > 0:
                header += f' z{ii+1:02d} '
            else:
                header += f' x{ii+1:02d} '
        header += '   coeff    C.O.V\'s'
        print(header)
        
        # Print each term
        retained_terms = 0
        for it in range(nTerm):
            line = f'  {it:3d}  '
            for ii in range(nZx):
                if order[it,ii] > 0:
                   line += f' {order[it, ii]:2d}  '
                else:
                   line += f'  .  '
            line += f'{coeff[it,io]:8.4f}  {coeffCOV[it,io]:8.4f}'
            if np.abs(coeff[it,io]) > 1e-4:
                line += '  *'
                retained_terms += 1
            print(line)
        
        print()
        print(f'  X scaling option                      = {scaling[0]:3d}: {scaling_names[scaling[0]]}')
        print(f'  Y scaling option                      = {scaling[1]:3d}: {scaling_names[scaling[1]]}')
        print(f'  Total Number of Terms                 = {retained_terms:4d} out of {nTerm:4d}')
        print(f'  Akaike Information Criterion          = {AIC[io]:7.3f}')
        print(f'  Adjusted R-square                     = {R2adj[io]:7.3f}')
        print(f'  model-data correlation                = {MDcorr[io]:7.3f}')
    
    # Estimate time remaining
    # import time
    
    # Calculate estimated time remaining
    # Note: In python, we'd need to track elapsed time separately
    print('  ==================================')


# updated 2006-01-29, 2007-02-21, 2007-03-06, 2009-10-14, 2022-11-19 2023-02-27, 2023-05-31 2025-11-17

function [ order, coeff, meanX,meanY, trfrmX,trfrmY,  testModelY, testX,testY ] = mimoSHORSA ( dataX,dataY, max_order, pTrain,pCull, tol, scaling, L1_pnlty, basis_fctn )
% [ order, coeff, trfrmX,trfrmY, meanX,meanY, testModelY, testX,testY ] = mimoSHORSA( dataX,dataY, max_order, pTrain,pCull, tol, scaling, L1_pnlty, basis_fctn )
%
% mimoSHORSA
% multi-input multi-output Stochastic High Order Response Surface Algorithm
% 
% This program fits a high order polynomial to multidimensional data via
% the high order response surface (mimoSHORSA) method 
%
%  mimoSHORSA approximates the data with a polynomial of arbitrary order,
%    y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
%           \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.
% The first stage of the algorithm determines the correct polynomial order,
% k_i in the response surace. Then the formulation of the mixed terms 
% \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq} are derived by the second stage
% based on previous results. In the third stage, the response surface 
% is approximated.
%
% INPUT       DESCRIPTION                                                DEFAULT
% --------    --------------------------------------------------------   -------
% dataX       m observations of n input  features in a (nx x m) matrix
% dataY       m observations of m output features in a (ny x m) matrix
% max_order    maximum allowable polynomial order                            3
% pTrain      percentage of data for training (remaining for testing)      50
% pCull       maximum percentage of model which may be culled              30 
% tol         desired maximum model coefficient of variation                0.10
% scaling     scale the data before fitting                                 1
%             scaling = 0 : no scaling
%             scaling = 1 : subtract mean and divide by std.dev
%             scaling = 2 : subtract mean and decorrelate
%             scaling = 3 : log-transform, subtract mean and divide by std.dev
%             scaling = 4 : log-transform, subtract mean and decorrelate
% L1_pnlty    coefficient for L_1 regularization        
% basis_fctn  'H': Hermite fctn, 'L': Legendre polynomial, 'P': power polynomial
%
% OUTPUT      DESCRIPTION
% --------    --------------------------------------------------------
%  order      matrix of the orders of variables in each term in the polynomial 
%  coeff      polynomial coefficients 
%  meanX      mean vector of the scaled dataX
%  meanY      mean vector of the scaled dataY
%  trfrmX     transformation matrix from dataZx to dataX
%  trfrmY     transformation matrix from dataZy to dataY
%  testModelY output features for model testing
%  testX      input  features for model testing 
%  testY      output features for model testing 
%
% get rainbow.m from ... http://www.duke.edu/~hpgavin/m-files/rainbow.m

% Reference:
%    Gavin, HP and Yau SC, ``High order limit state functions in the 
%    response surface method for structural reliability analysis,''
%    submitted to Structural Safety, December 2005.     
%
% Department of Civil and Environmental Engineering
% Duke University
% Siu Chung Yau, Henri P. Gavin, January 2006, 2023, 

  fprintf('\n Multi-Input Multi-Output High Order Response Surface (mimoSHORSA)\n\n');

  if nargin < 2 , help mimoHOSRS; return; end
  if nargin < 3 , max_order = 3;  else max_order = round(abs(max_order)); end
  if nargin < 4 , pTrain   = 0.5; else pTrain   = abs(pTrain)/100;     end
  if nargin < 5 , pCull    = 0.3; else pCull    = abs(pCull)/100;      end
  if nargin < 6 , tol      = 0.1; else tol      = abs(tol);            end
  if nargin < 7 , scaling  = 0;   else scaling  = round(abs(scaling)); end
  if nargin < 8 , L1_pnlty = 1;   else L1_pnlty = abs(L1_pnlty);       end

  if L1_pnlty > 0 , pCull = 0; end    % no "culling" with L_1 regularization


  [nInp, mDataX] = size(dataX);   % number of columns in dataX is mData
  [nOut, mDataY] = size(dataY);   % number of columns in dataX is mData

  if (mDataX ~= mDataY) 
    error(' the dataX and dataY matrices must have the same number of colimns '); 
  else
    mData = mDataX;
  end

  % initialize cell arrays for ...
  B        = cell(nOut,1);  % correlating matrix
  coeff    = cell(nOut,1);  % model coefficient vector
  coeffCOV = cell(nOut,1);  % coefficient of variation of the model coefficients


  [trainX, trainY, mTrain, testX, testY, mTest] = split_data(dataX,dataY,pTrain);

  % scale data matrices trainX and trainY separately since using 
  % tne covariance between trainX and trainY in the model is "cheating"
  [trainZx , meanX , trfrmX] = scale_data( trainX, scaling );
  [trainZy , meanY , trfrmY] = scale_data( trainY, scaling );

  if ( scaling > 0 ) % remove each column of trainZx and trainZy with outliers
    XY  = [ trainZx ; trainZy ];
    XY  = XY(:,find(all( XY > -4 ))); 
    XY  = XY(:,find(all( XY <  4 ))); 

    nData = size(XY,2)  
    trainZx = XY(1:nInp,:);
    trainZy = XY(nInp+1:nInp+nOut,:);
    fprintf(sprintf('%f < trainZx < %f \n', min(min(trainZx)), max(max(trainZx)) ));
    fprintf(sprintf('%f < trainZy < %f \n', min(min(trainZy)), max(max(trainZy)) ));

  end

% pause(3)

% separate order for each variable --- Not needed if data is already provided
% [max_order, orderR2] = polynomial_orders(max_order);

  max_order = max_order*ones(1,nInp);   % same maximum order for all variables
 
  [ order, nTerm ] = mixed_term_orders( max_order, nInp, nOut ); 

% initialize variables
  maxCull = max(1,round(pCull*nTerm(1)));  % maximum number of terms to cull
  condB   = NaN(nOut,maxCull); % condition number of basis as model is culled
  for io = 1:nOut
    coeffCOV{io} = ones(1,nTerm(1));
  end
  trainMDcorr = NaN(nOut,maxCull);
   testMDcorr = NaN(nOut,maxCull);
  coeffCOVmax = NaN(nOut,maxCull); 

% start a timer to measure computational time
  tic
  for iter = 1:maxCull   % cull uncertain terms from the model ------------

    % plot model coefficients and correlations at first and last culling iter
    if ( ( iter == 1 ) || iter == maxCull ) || ( max(coeffCOVmax(:,iter)) < 2*tol )
       trainFigNo = 200;
        testFigNo = 300;
    else
       trainFigNo = 0;
        testFigNo = 0;
    end

    % fit ("train") a separate model for each output (dependent) variable
    for io = 1:nOut
      [ coeff{io} , condB(io,iter) ] = fit_model( trainZx, trainZy(io,:), order{io}, nTerm(io), mTrain, L1_pnlty, basis_fctn );
    end

    % compute the model for the training data and the testing data
    [trainModelY, B] = compute_model(order,coeff, meanX,meanY,trfrmX,trfrmY, trainX,scaling, basis_fctn);
    [ testModelY, ~] = compute_model(order,coeff, meanX,meanY,trfrmX,trfrmY,  testX,scaling, basis_fctn);

    % evaluate the model for the training data and the testing data
    [trainMDcorr(:,iter), coeffCOV, ~, ~ ] = evaluate_model(B,coeff, trainX, trainY, trainModelY, trainFigNo, 'test');
    [ testMDcorr(:,iter), ~, R2adj,  AIC ] = evaluate_model(B,coeff,  testX,  testY,  testModelY,  testFigNo, 'train');

    pause(1);

    for io = 1:nOut
      coeffCOVmax(io,iter) = max(coeffCOV{io});
    end

    print_model_stats( iter, coeff, order, coeffCOV, testMDcorr(:,iter), R2adj, scaling, maxCull ); 
     
    for io = 1:nOut
      figure(400+io)
        formatPlot(18,4,8)
        cMap = rainbow(nOut);
        clf
        hold on
        semilogy([1:nTerm(io)], coeffCOV{io},  'o', 'color', cMap(io,:) )
          for ii = 1:nTerm(io)
             text(ii,0.85*coeffCOV{io}(ii),sprintf(' %1d', order{io}(ii,:)), 'FontSize',10)
          end
        ylabel('coefficient of variation')
        xlabel('term number')
        title(sprintf('Y_%d, \\rho_{train} = %5.3f, \\rho_{test} = %5.3f, cond(B) = %5.1f', io, trainMDcorr(io,iter), testMDcorr(io,iter), condB(io,iter) ));
    end 
    drawnow

    if ( 0 < testMDcorr(:,iter) )  && ( max(coeffCOVmax(iter)) < tol )
      maxCull = iter;
      break
    end

    if L1_pnlty == 0
      [order, nTerm, coeffCOV] = cull_model( coeff, order, coeffCOV, tol ); 
    end

  end                    % ------------ cull uncertain terms from the model

  figure(500)            % plot correlations and coefficents of variation 
    clf
    cMap = rainbow(nOut);
    formatPlot(18,2,4)
    subplot(211)
     hold on
     for io = 1:nOut
       plot([1:maxCull], trainMDcorr(io,1:maxCull) , 'o', 'color', cMap(io,:))
       plot([1:maxCull],  testMDcorr(io,1:maxCull) , 'x', 'color', cMap(io,:))
     end
     hold off
      ylabel('model-data correlation')
      legend('train', 'test')
      legend('location','West')
    subplot(212)
     hold on
     for io = 1:nOut
       semilogy([1:maxCull], condB(io,1:maxCull),  'o', 'color', cMap(io,:) )
       semilogy([1:maxCull], coeffCOVmax(io,1:maxCull),  'x', 'color', cMap(io,:) )
     end      
     hold off
      legend('cond(B)', 'max(c.o.v.)')
      legend('location','East')
      ylabel('maximum c.o.v.')
      xlabel('model reduction')

  clear dataX dataY trainX trainY  trainZx trainZy 

end % ================================================== main function HORSMu 


function [trainX,trainY,mTrain, testX,testY,mTest] = split_data(dataX,dataY,pTrain)
% [trainX,trainY,mTrain, testX,testY,mTest] = split_data(dataX,dataY,pTrain)
% split data into a training set and a testing set
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
%  dataX      m observations of nx input  "explanatory variables     nx x m 
%  dataY      m observations of ny output "explanatory variables     ny x m 
%  pTrain     fraction of the m observations to use for training      1 x 1
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% trainX      matrix of  input data for training                    nx x mTrain
% trainY      matrix of output data for training                    ny x mTrain
% mTrain      number of observations in the training set             1 x 1
% testX       matrix of  input data for testing                     nx x mTest
% testY       matrix of output data for testing                     ny x mTest
% mTest       number of observations in the testing set              1 x 1

  [nInp, mData] = size(dataX);   % number of columns in dataX is mData

  mTrain = floor(pTrain*mData);  % number of data points in training data set
  mTest  = mData - mTrain;       % number of data points in testing  data set

  reorder  = randperm(mData);         % random permutation of integers [1:mData]
  idtrainX = reorder(1:mTrain);       % indices of training data
  idtestX  = reorder(mTrain+1:mData); % indices of testing data

  trainX = dataX(:,idtrainX);
  trainY = dataY(:,idtrainX);

  testX  = dataX(:,idtestX);
  testY  = dataY(:,idtestX);

end % ===================================================== function split_data


function [order, orderR2] = polynomial_orders(max_order) 
% [order, orderR2] = polynomial_orders(max_order) 
  fprintf('1st Stage: Polynomial Order Determination ...\n');

  order    = ones(1,n);     % initial guess of response surface orders
  quality  = zeros(1,n); 
  orderR2  = zeros(1,n);
  no_pts   = max_order+15;   % number of sample points (must be > ki+1)

  % sample points along dimension X_i within the domain [-1,1]
  % (roots of no_pts-th order Chevbyshev polynomial)
  z = cos(pi*([1:no_pts]-0.5)/no_pts); 
 
  for i = 1:n     % determine the orders for each variable one by one 
    
  % allocate memory for matrix of sampling points along all variables
    zAvg = mean(Zx,2);
    zMap = zAvg * ones(1,no_pts);

  % the sample points along z (-1 <= z <= 1) are linearly mapped onto the domain [zMin ... zMax]
  % only the i-th row is non-zero since all other variables 
  % are kept constants at their mean values 
    zMax = max(Zx(i,:));
    zMin = min(Zx(i,:)); 
    zMap(i,:) = ( zMax+zMin + z*(zMax-zMin) ) / 2;

  % interpolate the data at the Chebyshev sampling points
    y = IDWinterp( Zx', Zy', zMap', 2, 10, 0.1 );

    for ki = order(i) : max_order   % loop over possible polynomial orders
                               
  % values of 0-th to ki-th degree Chevbyshev polynomials at
  % the sampling points.
  % this matrix is used for the determination of coefficients 
      Tx = cos(acos(z')*(0:ki));   
                                    
      d = Tx'*y ./ diag(Tx'*Tx);  % coefficients by least squares method
                
      residuals = y-Tx*d;    % residuals of the 1-D curve-fit
      fit_error = norm(residuals) / (norm(y) - mean(y));  % error of the curve fit
      orderR2(i) = 1 - fit_error^2;
                                    
      figure(103)
       formatPlot(18,4,8)
       clf
       hold on
       plot ( zMap(i,:), y,     'ob' )
       plot ( zMap(i,:), Tx*d,  '*r' )
       hold off
       if scaling > 0
         xlabel(sprintf('Z_%d', i))
       else
         xlabel(sprintf('X_%d', i))
       end
       ttl = sprintf('k_{%d}=%d   R^2 = %5.3f   fit-error = %5.3f', i, ki, orderR2(i),fit_error);
%      axis([ min(zMap(i,:)), max(zMap(i,:)), min(y), max(y) ]);
       legend('data','fit')
       title(ttl);
       pause(0.1)

      if ( orderR2(i) > 1-tol && fit_error < tol ) % the 1D fit is accurate
        break;
      end

    end

    order(i) = ki;    % save the identified polynomial order

  end

  % output the results
  fprintf('  Variable     Determined Order    R_sq \n');
  for i = 1:n
      fprintf('%10.0f %20.0f    %9.6f\n', i, order(i),  orderR2(i) );
  end
  
end % =============================================== function polynomialOrder 


function [ Z, meanD, T, maxZ, minZ ] = scale_data(Data, scaling)
% [ Z, meanD, T ] = scale_data(Data, scaling)
%  scale data in one of four or five ways ..  Z = inv(T)*(Data - meanD); 
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% Data        a matrix of data values                               n x m
% scaling     type of scaling ...                                   1 x 1
%              scaling = 0 : no scaling
%              scaling = 1 : subtract mean and divide by std.dev
%              scaling = 2 : subtract mean and decorrelate
%              scaling = 3 : log-transform, subtract mean and divide by std.dev
%              scaling = 4 : log-transform, subtract mean and decorrelate
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% Z           scaled data                                           n x m
% meanD       the arithmetic mean of the (log-transformed) data     n x 1
% T           transformation matrix                                 n x n
% maxZ        maximum value of each data sample                     1 x n
% minZ        maximum value of each data sample                     1 x n

  [n,m] = size(Data);       % m observations of n variables

  switch scaling            % scaling options for mean

    case 0                  % no scaling
      Z = Data;
      meanD = zeros(n,1);
      T = eye(n);

    case 1                  % subtract mean and divide by std.dev            
      meanD = mean(Data')'; 
      T = diag(sqrt(var(Data')));

    case 2                  % subtract mean and decorrelate
      meanD = mean(Data')';       
      [eVec,eVal] = eig(cov(Data'));
      T = eVec * sqrt(eVal);

    case 3                  % log-transform, subtract mean and divide by std.dev
      Data = log10(Data);
      meanD = mean(Data')';   
      T = diag(sqrt(var(Data')));

    case 4                  % log-transform, subtract mean and decorrelate
      Data = log10(Data);
      meanD = mean(Data')'; 
      [eVec,eVal] = eig(cov(Data'));
      T = eVec * sqrt(eVal);

  end  %  ------------------------------ switch scaling

  Z = T \ (Data - meanD);   % apply the scaling 

  maxZ = max(Z');
  minZ = min(Z');

end % ===================================================== function scale_data


function Data = clip_data( Data, lowLimit, highLimit )
%  clip_data(Data, S, figNo)
%  remove outliers from the data
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% Data        matrix of data                                          n x m  
% lowLimit    remove values lower  than lowLimit                      1 x 1
% highLimit   remove values higher than highLimit                     1 x 1
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
%  Data       matrix of data without values exeeeding give limits

% remove low and high values (outliers)

  [ idxr, idxc ] = find( lowLimit <  Data);
  Data = Data(:,unique(idxc)); 

  [ idxr, idxc ] = find( Data < highLimit);
  Data = Data(:,unique(idxc)); 

end % ====================================================== function clip_data


function scatter_data(Data,S,figNo)
%  scatter_data(Data, S, figNo)
%  make a scatter plot of the data
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% Data        a matrix of data values                               n x m
%  S          a character string corresponding to the variable
% figNo       figure number

  [n,m] = size(Data);       % m observations of n variables

  figure(figNo) % scatter plot of scaled input features and the output features
  clf
  formatPlot(10,1,2)
  kk = 0;
  for ii=1:n  
    for jj = 1:n   
      kk = kk+1;
      subplot(n,n,kk)
        plot(Data(jj,:), Data(ii,:), 'o')
        if jj==1 
          ylabel(sprintf('%s_%d',S,ii));
        end
        if ii==n                 
          xlabel(sprintf('%s_%d',S,jj));
        end
        axis('square')
        axis('tight')
    end
  end

  figure(figNo+1); hist(Data',30);    % plot histograms of the data to check scaling

end % =================================================== function scatter_data


function [ order , nTerm ] = mixed_term_orders( max_order, nInp, nOut )
% [ order , nTerm ] = mixed_term_orders( max_order, nInp, nOut )
% specify the exponents on each input variable for every term in the model,
% and the total number of terms, nTerm
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% max_order   maximum polynomial order of the model                   1 x 1
% nInp        number of input  (explanatory) variables                1 x 1
% nOut        number of output  (dependent)  variables                1 x 1
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------   ---------
% order       cell array of matrices of model orders for each output
%             indicating powers present in each model terms      {nTerm x nInp}
%             initially, these matrices are all the same 
% nTerm       number of polynomial terms in the model                 1 x 1
%
% The matrix 'order' indicates which mixed term power-products are present
% in each term of the model. 
%
% Each element in this matrix represents the order of a variable in a term.
% The columns of 'order' correspond to individual variables. 
% Each row indicates the powers present in the prodcut of a term.
 
% Algorithm by Siu Chung Yau (2006)

  fprintf('Determine the Mixed Term Power Products ...\n');

  nTerm = prod(max_order+1);

  ordr = zeros(nTerm,nInp);         % allocate memory for the 'order' matrix
  term = zeros(1,nInp);              % orders in a given term
  term(1) = -1;                      % starting value for the first term

  for t = 1:nTerm                    % loop over all terms
    term(1) = term(1)+1;             % increment order of the first variable
    for v = 1:nInp                   % check every column in the row
      if term(v) > max_order(v)
         term(v) = 0;
         term(v+1) = term(v+1)+1;    % increment columns as needed
      end
      for io = 1:nOut
        ordr(t,:) = term; % save the orders of term t in the 'order' matrix
      end
    end
  end
 
  % The power of a variable in each term can not be greater than 
  % the order of that variable alone.  
  % Remove the terms in which the total order is larger than 
  % the highest order term.  

  [it, ~, ~ ] = find( sum(ordr,2) <= max(max_order) );
  ordr = ordr( it , : );

  % The number of rows is in the matrix 'order' is the number of
  % required terms in the model
 
  nTerm = size(ordr,1) * ones(nOut,1);
 
  for io = 1:nOut
    order{io} = ordr;         
  end
 
  fprintf('  Total Number of Terms: %3d\n', nTerm(1));
  fprintf('  Number of Mixed Terms: %3d\n\n', nTerm(1)-sum(max_order)-1);

end % ============================================= function mxed_term_powers


function psyProduct = polynomial_product( order, Zx, N, basis_fctn )
% psyProduct = polynomial_product( order, Zx, N, basis_fctn )
% compute the product of polynomials of given orders (from 0 to 5)
% for a set of column vectors Z, where each column of Zx has a given order
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  order      vector of powers present in one term of the model      1 x nInp
%    Zx       matrix of scaled input (explanatory) variables      nInp x mData
%    N        largest order in the full expansion                    1 x 1
% basis_fctn  'H': Hermite fctn, 'L': Legendre polynomial, 'P': power polynomial
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
% psyProduct  vector of product of polynomials                      1 x mData 
 
  nInp = length(order);            % number of input (explanatory) variables
  psyProduct = ones(size(Zx,1),1); % initialze to vector of 1

  switch basis_fctn
  case 'L' % Legendre 
    for k = 1:nInp
      psyProduct = psyProduct .* legendre( order(k), Zx(:,k) );
    end
  case 'H' % Hermite
    for k = 1:nInp
      psyProduct = psyProduct .* hermite( order(k), Zx(:,k), N );
    end
  end

end % ============================================ function polynomial_product


function psy = hermite(n,z,N)
% psy = hermite(n,z,N)
% compute the shifted Hermite function of a given order (orders from 0 to 10)
% for a vector of values of z 
% https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions
% Note: These Hermite functions are index-shifted by 2, in order to 
% augment the basis with a constant (0-order) function and a linear (1-order) 
% function.  The 0-order and 1-order functions have approximately unit-area and 
% attenuate exponentially at rates comparable to the highest order Hermite 
% function in the basis.  
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%     n       the polynoimial order of a Hermite function            1 x 1
%     z       vector of input (explanatory) variables                1 x mData  
%     N       largest order in the full expansion                    1 x 1
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  psy        a Hermite function of specified order at given values  1 x mData
 
  pi4 = pi^(0.25);
  ez2 = exp(-0.5*z.^2);

  z0 = N+2;     % expand the domain of extrapolation 
 
  switch n 
    case  0
      psy =          exp(-(z/z0).^(6*z0));
    case  1
      psy = (z/z0) .* exp(-(z/z0).^(6*z0));
    case  2
      psy = 1/pi4 * ez2;
    case  3
      psy = sqrt(2)/pi4 * z .* ez2;
    case  4
      psy = 1/(sqrt(2)*pi4) * (2*z.^2 - 1) .* ez2;
    case  5
      psy = 1/(sqrt(3)*pi4) * (2*z.^3 - 3*z) .* ez2;
    case  6
      psy = 1/(2*sqrt(6)*pi4) * (4*z.^4 - 12*z.^2 + 3) .* ez2;
    case  7
      psy = 1/(2*sqrt(15)*pi4) * (4*z.^5 - 20*z.^2 + 15*z) .* ez2;
    case  8
      psy = 1/(12*sqrt(5)*pi4) * (8*z.^6 - 60*z.^4 + 90*z.^2 - 15) .* ez2;
    case  9
      psy = 1/(6*sqrt(70)*pi4) * (8*z.^7 - 84*z.^5 + 210*z.^3 - 105*z) .* ez2;
    case  10 
      psy = 1/(24*sqrt(70)*pi4) * (16*z.^8 - 224*z.^6 + 840*z.^4 - 840*z.^2 + 105) .* ez2;
    case  11 
      psy = 1/(72*sqrt(35)*pi4) * (16*z.^9 - 288*z.^7 + 1512*z.^5 - 2520^z.^3 + 945*z) .* ez2;
    case  12
      psy = 1/(720*sqrt(7)*pi4) * (32*z.^10 - 720*z.^8 + 5040*z.^6 - 12600*z.^4 + 9450*z.^2 - 945) .* ez2;
  end

end % ======================================================= function hermite


function psy = legendre(order,z)
% psy = legendre(order,z,N)
% compute the Legendre function of a given order (orders from 0 to 10)
% for a vector of values of z 
% https://en.wikipedia.org/wiki/Legendre_polynomials#Rodrigues'_formula_and_other_explicit_formulas
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  order      the polynoimial order of a Legendre function           1 x 1
%     z       vector of input (explanatory) variables                1 x mData  
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  psy        a Hermite function of specified order at given values  1 x mData
 

  switch order
    case  0
      psy = ones(length(z),1); 
    case  1
      psy = z;
    case  2
      psy = (    3*z.^2 -     1  )/2;
    case  3
      psy = (    5*z.^3 -     3*z)/2;
    case  4
      psy = (   35*z.^4 -    30*z.^2 +    3  )/8;
    case  5
      psy = (   63*z.^5 -    70*z.^3 +   15*z)/8;
    case  6
      psy = (  231*z.^6 -   315*z.^4 +  105*z.^2 -    5  )/16;
    case  7
      psy = (  429*z.^7 -   693*z.^5 +  315*z.^3 -   35*z)/16;
    case  8
      psy = ( 6435*z.^8 - 12012*z.^6 + 6930*z.^4 - 1260*z.^2+  35   )/128;
    case  9
      psy = (12155*z.^9 - 25740*z.^7 +18018*z.^5 - 4620*z.^3+ 315*z )/128;
    case  10 
      psy = (46189*z.^10-109395*z.^8 +90090*z.^6 -30030*z.^4+3465.*z.^2-63)/256;
  end

end % ======================================================= function hermite


function B = build_basis( Zx, order, basis_fctn ) 
% B = build_basis( Zx, order, basis_fctn ) 
% compute matrix of model basis vectors
% options: power-polynomial basis  or  Hermite function basis
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  Zx         matrix of input (explanatory) variables             nInp x mData
% order       powers for each variable on each term of the model nTerm x nInp
% basis_fctn  'H': Hermite fctn, 'L': Legendre polynomial, 'P': power polynomial
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%   B         matrix basis vectors for the polynomial model      mData x nTerm

  mData = size(Zx,2);            % number of data points
  [ nTerm, nInp ] = size(order); % number of terms, inputs, outputs 
  B = ones( mData , nTerm );     % the matrix of model basis vectors 

  % in the matrix of basis vectors, B, 
  % columns correspond to each term in the polynomial and 
  % rows correspond to each observation 

  max_order = max(max(order)); 

  switch basis_fctn % select basis functions 
    case 'P' % ... power polynomials ...
      for it = 1:nTerm
        B(:,it) = prod( Zx'.^(ones(mData,1)*order(it,:)) , 2);
      end
    otherwise % Legendre or Hermite ... maybe Tschebitshiew also
      for it = 1:nTerm
        B(:,it) = polynomial_product( order(it,:), Zx', max_order, basis_fctn );
      end
  end
 
end % ==================================================== function build_basis


function [ coeff , condB ] = fit_model( Zx, Zy, order, nTerm, mData, L1_pnlty, basis_fctn )
% [ coeff , condB ] = fit_model( Zx, Zy, order, nTerm, mData )
% Fit the polynomial model from all the input data Zx to one of the output data Zy 
% via singular value decomposition without regularization
% or quadratic programming with L1 regularization 
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  Zx         scaled  input (explanatory) data                        nx x mData
%  Zy         scaled output  (dependent)  data                         1 x mData
%  order      order of each explantory variabe in each term        nTerm x nx
%  nTerm      number of terms in the polynomial model                  1 x 1
%  L1_pnlty   L_1 regularization coefficient                           1 x 1 
% basis_fctn  'H': Hermite fctn, 'L': Legendre polynomial, 'P': power polynomial
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  coeff      vector of model coefficients                           nTerm x 1  
%  condB      condition number of the model basis                     nOut x 1

  fprintf('Fit The Model ... with L1_pnlty = %f \n', L1_pnlty );

  B = build_basis( Zx, order, basis_fctn ); 


% plot the basis (for nInp == 2) 
  nBasis = size(B,2)
  for ii = 1:nBasis
    figure(1000+ii)
    clf
    hold on
    plot3(Zx(1,:), Zx(2,:),B(:,ii),'o')
    xlabel('X_1')
    ylabel('X_2')
    zlabel(sprintf('B_{%d}',ii-1))
    title(sprintf('B_{%d}',ii-1))
  end
 
  % determine the coefficients of the response surface for each output

  if L1_pnlty > 0             % ... by QP optimization for L1 regularization
    [ coeff, mu, nu, cvg_hst ] = L1_fit( B, Zy', L1_pnlty, 0 );
    L1_plots( B, coeff, Zy', cvg_hst, L1_pnlty, 0, 7000 );
  else 
    coeff = B \ Zy';          % ... by singular value decomposition
  end

  condB = cond(B)         % condition number

  fprintf('  condition number of model basis matrix = %6.1f \n', condB );

end % ====================================================== function fit_model

 
function  [ modelY, B ] = compute_model(order, coeff, meanX,meanY,trfrmX,trfrmY,dataX,scaling, basis_fctn)
% [ modelY, B ] = compute_model(order, coeff, meanX,meanY,trfrmX,trfrmY,dataX,scaling)
% compute a multivariate power-polynomial model 
%
% INPUT       DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  order      powers on each explantory variabe for each term      nTerm x nx
%  coeff      model coefficient vector                             nTerm x 1  
%  meanX      mean of pre-scaled input  (explanatory) variables     nInp x 1 
%  meanY      mean of pre-scaled output  (dependent)  variables     nOut x 1 
%  trfrmX     transformation matrix for input variables             nInp x nInp 
%  trfrmY     transformation matrix for input variables             nOut x nOut
%  scaling    scaling type ... see scale_data function                 1 x 1
% basis_fctn  'H': Hermite fctn, 'L': Legendre polynomial, 'P': power polynomial
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
% modelY      computed model                                        nOut x mData
%  B          basis vector matrix of the computed model            mData x nTerm  

  [nInp, mData]  =  size(dataX);   % number of columns in dataX is mData
  nOut    = size(order,2);         % number of columns in dataY is mData
  nTerm   = size(coeff,2);         % number of terms in the model
  modelZy = NaN(mData,nOut);       % initialize model output

  switch scaling

    case 0
      dataZx = dataX;

    case { 1 , 2 }
      dataZx = inv(trfrmX) * (dataX - meanX);

    case { 3 , 4 }
      log10X = log10( dataX ); 
      dataZx  = inv(trfrmX)*(log10X - meanX);  % standard normal variables 

  end  % --- scaling

  for io = 1:nOut

    B{io} = build_basis( dataZx, order{io}, basis_fctn ); 

    modelZy(:,io) = B{io} * coeff{io};

  end  % io

  switch scaling

    case 0
     modelY = modelZy';

   case { 1 , 2 }
      modelY = trfrmY * modelZy' + meanY;

    case { 3 , 4 }
      modelY = 10.^(trfrmY * modelZy' + meanY );

  end % --- switch scaling

 
end % ================================================== function compute_model


function [MDcorr, coeffCOV, R2adj, AIC] = evaluate_model( B, coeff, dataX, dataY, modelY, figNo, txt )
% [ MDcorr, coeffCOV , R2adj] = evaluate_model( B, coeff, dataX, dataY, modelY, figNo, txt )
% evaluate the model statistics 
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  B          cell array of basis vector matrices of each model  {mData x nTerm}
%  coeff      cell array of coefficent vectors of each model     {nTerm x 1}  
%  dataX      input  (explanatory) data                            nInp x mData
%  dataY      output  (dependent)) data                            nOut x mData
%  figNo      figure number for plotting (figNo = 0: don't plot)      1 x 1
%  txt        text annotation 
%
% OUTPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
% MDcorr      model-data correlation                                  1 x nOut
% coeffCOV    cell array of coefficient of variation of
%             each model coefficent of each model                    {1 x nTerm}
% R2adj       adjusted R-squared for each model                       1 x nOut
% AIC         Akaike information criterion for each model             1 x nOut

  [ nOut, mData ] = size(dataY);
  MDcorr = NaN(nOut,1);

  % statistical analysis of coefficients
  residuals = ( dataY - modelY );    % matrix of residuals for each model

  for io = 1:nOut

    nTerm = size(coeff{io},1)     % number of terms in model "io" 

    r = residuals(io,:);          % R-squared criterion  for model "io"
    m = modelY(io,:);             % computed output data for model "io"
    d = dataY(io,:);              % measured output data for model "io"
    R2 = 1 - ( norm(r) / norm( m - mean(m) ) )^2;  % R-squared 

    % R-squared criterion adjusted for the amount of data and number of coefficients
    R2adj(io) = ( (mData-1) * R2 - nTerm ) / (mData - nTerm);

    MDc = corrcoef( [ d ; m ]' ); % correlation between model "io" and the data
    MDcorr(io) = MDc(1,2);
 
    % standard error of each coefficient for model "io"
    Std_Err_Coeff = sqrt( ( r*r' ) * diag(inv(B{io}'*B{io})) / (mData-nTerm) ); 
 
    % coefficient of variation of each coefficient for model "io"
    coeffCOV{io} =  Std_Err_Coeff ./ ( abs(coeff{io}) + 1e-6 );
    coeffCOV{io}(find(abs(coeff{io})<1e-6)) = 1e-3; 

    AIC(io) = 0;   % add AIC here

  end

  if figNo 
    cMap = rainbow(nOut);
    figure(figNo)
      formatPlot(18,1,3)
      ax = [min(min(dataY)), max(max(dataY)), min(min(dataY)), max(max(dataY))];
      clf
      hold on
      for io = 1:nOut 
        plot( modelY(io,:), dataY(io,:), 'o', 'color', cMap(io,:) )
      end
      plot( [ ax(1),ax(2) ] , [ ax(3), ax(4) ], '-k', 'LineWidth',0.5)
      hold off
      axis('square')
      xlabel('Y model')
      ylabel('Y data')
      tx = 0.00;
      ty = 1.0-0.00*io;
      text(tx*ax(2) + (1-tx)*ax(1), ty*ax(4) + (1-ty)*ax(3), ...
           sprintf('%d model terms', nTerm(1) ));
      for io = 1:nOut
        tx = 0.75;
        ty = 0.4-0.1*io;
        text(tx*ax(2) + (1-tx)*ax(1), ty*ax(4) + (1-ty)*ax(3), ...
             sprintf('\\rho_{x,y%d} = %5.3f', io, MDcorr(io) ) , 'color', cMap(io,:) );
        ty = 0.4+0.0*io;
        text(tx*ax(2) + (1-tx)*ax(1), ty*ax(4) + (1-ty)*ax(3), sprintf('%s', txt ));
      end
  end

end % ================================================ function evaluate_model
 

function [ order, nTerm, coeffCOV ] = cull_model( coeff, order, coeffCOV, tol )
% [ order, nTerm, coeffCOV ] = cull_model( c, order, coeffCOV, tol )
% remove the term from the model that has the largest coeffCOV
%
%  INPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  coeff      cell array of coefficent vectors of each model     {nTerm x 1}  
%  order      powers on each explantory variabe for each term    {nTerm x nInp}
% coeffCOV    cell array of coefficient of variation of
%             each model coefficent of each model                    {1 x nTerm}
%  tol        tolerance for an acceptable coeffCOV                    1 x 1
%
% OUNPUT      DESCRIPTION                                           DIMENSION
% --------    ---------------------------------------------------  -----------
%  order      retrained powers on each explantory variabe for each term {nTerm x nInp}
%  nTerm      number of terms in each polynomial model                1 x nOut
% coeffCOV    cell array of coefficient of variation of
%             each model coefficent of each culled model             {1 x nTerm}
 
   nOut = length(order); 

   for io = 1:nOut

     [ nTerm(io), nInp ] = size(order{io});

     % model coefficient with largest coefficient of variation 
     [max_cov, ic]   = max(coeffCOV{io});    

     % remove the 'ic-th' term from the model for output io

     order{io}    = order{io}( [ 1:ic-1 , ic+1:nTerm(io) ] , : ) ;
     coeff{io}    = coeff{io}( [ 1:ic-1 , ic+1:nTerm(io) ] ) ;
     coeffCOV{io} = coeffCOV{io}( [ 1:ic-1 , ic+1:nTerm(io) ] ) ;

     nTerm(io) = nTerm(io) - 1;

   end

end % ===================================================== function cull_model


function print_model_stats( iter, coeff, order, coeffCOV, MDcorr, R2adj, scaling, maxCull )
% print_model_stats( iter, coeff, order, coeffCOV, MDcorr, R2adj, scaling )

  nOut = length(order)  
  [ nTerm, nInp ]  = size(order{1});

  fprintf(' model culling iteration %d \n', iter );
  for io = 1:nOut
    fprintf('  Output %d ------------------------------------------\n', io); 
    fprintf('  Scaling Option    = %d \n', scaling );
    fprintf('  Response Surface Coefficients \n' );
    fprintf('    i  ');
    for ii = 1:nInp 
      if scaling > 0
        fprintf(' z%02d ', ii); 
      else
        fprintf(' x%02d', ii); 
      end
    end
    fprintf('   coeff    C.O.V''s    \n');
 
    for it = 1:nTerm
      fprintf('  %3d ', it );
        for ii =1:nInp 
          fprintf(' %2d  ', order{io}(it,ii) );
        end
      fprintf(' %8.4f  %8.4f \n', coeff{io}(it), coeffCOV{io}(it) );
    end
    fprintf('\n');
    fprintf('  scaling option            = %3d \n', scaling );
    fprintf('  Total Number of Terms     = %3d \n', nTerm * nOut);
    fprintf('  Adjusted R-square %d       = %6.3f \n', io , R2adj(io) );
    fprintf('  model-data correlation %d  = %6.3f \n', io , MDcorr(io)); 
  end

  secs_left = round((maxCull-iter)*toc/iter);
  fprintf('  ==================================  eta %s\n', ...
                                           datestr(now+secs_left/3600/24,14));

end % ============================================= function print_model_stats
   
% ########################################################  program mimoSHORSA

%{
  fprintf('\n\n  try GP --------------------------------------------- \n') 
 
  Zygp = GaussPross(Zx',Zy',Zx', 'Gaussian', [ 5 1 0.15 0.001 ], 0 );

  if scaling > 2
    Y_gp = 10.^( C(n+1,:) * [Zx ; Zygp'] + meanD(n+1) );
  else
    Y_gp = Zygp';
  end

  residuals = ( Y_gp - dataY )';    % residuals of the fit

  R2 = 1 - ( norm(residuals) / norm(Y_gp - mean(Y_gp)) )^2;  % R-squared error
  R2adj = ((m-1) * R2 - nTerm ) / (m - (nTerm+1)); % Adjusted R2
  MDcorr = corrcoef(dataY, Y_gp);

  fprintf('  R-square          = %6.3f \n', R2 );
  fprintf('  Adjusted R-square = %6.3f \n',  R2adj );
  fprintf('  model-data correlation = %6.3f \n', MDcorr(1,2)); 
%}

% updated 2006-01-29, 2007-02-21, 2007-03-06, 2009-10-14, 2022-11-19 2023-02-27, 2023-05-31, 2025-11-06


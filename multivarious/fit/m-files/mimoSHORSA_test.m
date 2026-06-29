function momoHORS_test()
% mimoSHORSA_test

% Test mimoSHORSA for fitting high dimensional multi-input, multi-output data

% ... Is this the fundamental problem statement of all data-science? 

data = 'MeasuredData';  % 'SyntheticData'  or  'MeasuredData' 

% "seed" rand() and randn() to generate a certain random sequence 
  rand('seed',30);
  randn('seed',30);

switch data % ---------------------------------------------

  case 'SyntheticData'    % apply mimoSHORSA to synthetic data

    nInp   = 5;     % number of  input features
    nOut   = 2;     % number of output features
    nData  = 2000;  % number of observations
    pNoise = 02;    % percentage random noise in the synthetic data

    medX = [ 1 : nInp ]';         % median values of X1 ... Xn
    covX =  0.4 * ones(nInp,1);   % coefficients of variation of X1 ... Xn

    % make up a symmetric positive definite correlation matrix 
    R = eye(nInp);          % all diagonal terms of a correlation matrix are "1"
    for ii = 1:nInp-1       % loop over the off-diagonal terms
      rr = 0.9/ii^0.2;     % the value of an off-diagonal term
      if rem(ii,2) rr = -rr; end   % switch +ve and -ve for off diagonals
      R = R + rr*diag(ones(nInp-ii,1),+ii);  % upper off diagonal
      R = R + rr*diag(ones(nInp-ii,1),-ii);  % lower off diagonal
    end
    
    % a sample of m observations of n correlated lognormal variables 
    xData = corr_logn_rnd(medX, covX, R, nData );   

    % presume that the output feature is a specific function of n variables ...
    [yData,lgd,maxN,rmsN] = test_function(xData, nOut, pNoise);

  case 'MeasuredData'     % apply mimoSHORSA from data loaded from a data file

    Data = load('/home/hpgavin/Research/Nepal-EEWS/m-files/data_20220606_csv.csv'); 

    r2k = find(Data(:,1)~=0);       % rows to keep, discard rows starting with 0

    xData = Data(r2k, [1,2,3,4,5])'; % use columns 1, 2, 3, 4, 5, for example
    yData = Data(r2k, [6,7])';       % PGV and PGA

    [nInp,nData] = size(xData); % nInp = number of rows, nOut = nuber of columns
    [nOut,nData] = size(yData); % nInp = number of rows, nOut = nuber of columns

end % switch data -----------------------------------------

number_of_data_points = nData
fprintf(sprintf('%f < xData < %f \n', min(min(xData)), max(max(xData)) ));
fprintf(sprintf('%f < yData < %f \n', min(min(yData)), max(max(yData)) ));

% remove each column of xData and yData with outliers
  min_allow_data = 1e-4;
  max_allow_data = 5000;

  XY  = [ xData  ; yData ];
  XY  = XY(:,find(all( XY > min_allow_data ))); 
  XY  = XY(:,find(all( XY < max_allow_data ))); 

  nData = size(XY,2)  
  xData = XY(1:nInp,:);
  yData = XY(nInp+1:nInp+nOut,:);
  clear XY

fprintf(sprintf('%f < xData < %f \n', min(min(xData)), max(max(xData)) ));
fprintf(sprintf('%f < yData < %f \n', min(min(yData)), max(max(yData)) ));

pause(3)


% {
  for iy = 1:nOut   % for each output features scatter plot the ...
    figure(10+iy)   % ... output feature w.r.t. each input feature individually
     cMap = rainbow(nOut);
     clf
     formatPlot(14,1,2)
     for ix = 1:nInp
       subplot(nInp,1,ix)
         plot(xData(ix,:), yData(iy,:), 'o', 'color', cMap(iy,:) ); 
         xlabel(sprintf('X_%d',ix))
         ylabel(sprintf('Y_%d',iy))
        axis('tight')
     end
     drawnow
  end
% }
  

maxOrder  =  5;   % maximum polynomial order for the model
pTrain    = 80;   % percentage of the data for traning (remaining for testing)
pCull     = 80;   % percentage of the model to be culled
tol       = 0.10; % maximum desired coefficient of variation 
% scaling = 0;    % no scaling
% scaling = 1;    % subtract mean and divide by std.dev
% scaling = 2;    % subtract mean and decorrelate
% scaling = 3;    % log-transform, subtract mean and divide by std.dev
  scaling = 4;    % log-transform, subtract mean and decorrelate

[ term, c, xTest, yTest ] = mimoSHORSA (xData, yData, maxOrder, pTrain, pCull, tol, scaling );

switch data
  case 'SyntheticData'
    maxN
    rmsN
  case 'MeasuredData'
end

for iy = 1:nOut   % for each output feature scatter-plot the ...
  figure(20+iy)   % ... output feature w.r.t. each input feature individually
   cMap = rainbow(nOut);
   clf
   formatPlot(14,1,2)
   for ix = 1:nInp
     subplot(nInp,1,ix)
       plot(xData(ix,:), yData(iy,:), 'ok' );
       plot(xData(ix,:), yTest(iy,:), 'o', 'color', cMap(iy,:) ); 
       xlabel(sprintf('X_%d',ix))
       ylabel(sprintf('Y_%d',iy))
      axis('tight')
   end
   drawnow
end


%{
%Plot results using Gaussian Process segment of HORSMx.m
figure(5)   % the output feature w.r.t. each input feature
 clf
 formatPlot(14,2,2)
 for ii = 1:n
   subplot(n,1,ii)
     hold on
     plot(xData(ii,:), yData, 'o'); 
     plot(xData(ii,:), Ygp, 'o'); 
     ylabel('Y')
     xlabel(sprintf('X_%d',ii))
     axis('tight')
 end
%}

end % ========================================= main function mimoSHORSA_test.m 


function [Y,lgd,maxN,rmsN] = test_function(X, nOut, pNoise)
% [Y,lgd,maxN,rmsN] = test_function(X,pNoise)
% generate synthetic data to test mimoSHORSA

 [ nInp, nData ] = size(X);
 Y = NaN(nOut,nData);

 for ix = 1:nInp
   if ix <= 1
     for iy = 1:nOut
       if iy == 1
         Y(iy,:) = 0.2*X(1,:) + 1; 
       end
       if iy == 2
         Y(iy,:) = 0.5*X(1,:) + 0; 
       end
     end
     lgd = {'0.2 X_1'};
   end
   if ix <= 2
     for iy = 1:nOut
       if iy == 1
         Y(iy,:) = Y(iy,:) + sin(X(2,:)) + 1;
       end
       if iy == 2
         Y(iy,:) = Y(iy,:) + cos(X(2,:)) + 1;
       end
     end
     lgd = {'0.2 X_1', 'sin(X_2)'};
   end
   if ix <= 3
     for iy = 1:nOut
       if iy == 1
         Y(iy,:) =  Y(iy,:) + 0.5*cos(2*X(3,:)) + 1; 
       end
       if iy == 2
         Y(iy,:) =  Y(iy,:) + 0.5*sin(2*X(3,:)) + 1; 
       end
     end
     lgd = {'0.2 X_1', 'sin(X_2)', 'cos(2 X_3)'};
   end
   if nInp <= 4
     for iy = 1:nOut
       if iy == 1
         Y(iy,:) =  Y(iy,:) + tanh(0.4*X(4,:)) - 1;
       end
       if iy == 2
         Y(iy,:) =  Y(iy,:) - tanh(0.4*X(4,:)) + 1;
       end
     end
     lgd = {'0.2 X_1', 'sin(X_2)', 'cos(2 X_3)', 'tanh(0.4 X_4)'};
   end
   if nInp <= 5
     for iy = 1:nOut
       if iy == 1
         Y(iy,:) =  Y(iy,:) + 2*exp(-(0.2*X(5,:)).^2);
       end
       if iy == 2
         Y(iy,:) =  Y(iy,:) + 2*exp(-(0.5*X(5,:)).^2);
       end
     end
     lgd= {'X_1', 'sin(X_2)', 'cos(2 X_3)', 'tanh(0.4 X_4)', '2 exp(-(X_5)^2)'};
   end
 end

  minY  =  min(min(Y)) 
  maxY  =  max(max(Y)) 


  noise = randn(size(Y))*(maxY-minY)*pNoise/100.0;  % normally distributed noise

  maxN =   max(max(abs(noise)))                        % maximum of noise 
  rmsN =   norm(noise,'fro') / sqrt(prod(size(noise))) % root mean square noise

  Y = Y + noise;          % add some noise to the Y data


end % ================================================= function test_function

% updated:  2022-11-19 2023-10-02 2023-05-31

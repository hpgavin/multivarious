% L1_fit_test


% independent variables
 x = [ -1.2 : 0.05 : 1.2 ]';  m = length(x);   

% power-polynomial fit Basis functions (the design matrix)
  B = [  x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 ];
% B = [       x.^1 x.^2 x.^3      x.^5      x.^7 ];

noise = 0.15 * randn(m,1); 

% the data is not polynomial
%y = x.^2 + tanh(5*x) + noise;
 y = 1 - x.^2 + sin(pi*x) + noise;        
%y = 1 - x + exp(-(2*x).^2) + noise;

 w     =  1.0;         % ... 0: without weighting ... >0: with weighting
 alpha = 1e-1;       % L1 regularization parameter

 [c, mu, nu, cvg_hst] = L1_fit( B, y, alpha, w );

 L1_plots( B, c, x, y, cvg_hst, alfa, w )


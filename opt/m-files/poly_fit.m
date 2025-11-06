function [c,x_fit,y_fit,Sc,Sy_fit,Rc,R2,Vr,AIC,condNo] = poly_fit(x,y,p,figNo,Sy,rof,b)
% [c,x_fit,y_fit,Sc,Sy_fit,Rc,R2,Vr,AIC,condNo] = poly_fit(x,y,p,figNo,Sy,rof,b)
%
% fit a power-polynomial, y_fit(x;a) to data pairs (x,y) where
%
%  y_fit(x;a) = SUM_i=1^length(p)  c_i  x^p_i
%
% which minimizes the Chi-square error criterion, X2,
%
%  X2 = SUM_k=1 ^ length(x) [ ( y_fit(x_k;a) - y_k )^2 / Sy_k^2 ]
%
% where  Sy_k is the standard error of the k-th data point.  
%
%  INPUT     DESCRIPTION                                        DIM     DEFAULT
% ------     ------------------------------------------------  -----    -------
%   x        known vector of the independent variables,        (1xm)
%   y        measured vector of the dependent variables,       (1xm)
%   p        vector of the real power of x (x^p) of each term  (1xn)
%   figNo    figure number. Use 0 to suppress plotting                    0
%   Sy       measurement errors for each value of y  (1x1) or  (1xm)      1
%   rof      range of the fit                                  (1x2) [xMin,xMax]
%   b        regularization constant                                      0
%
% OUTPUT     DESCRIPTION                                         
% ------     ----------------------------------------------------
%   c        values of the polynomial coefficients
%   x_fit    values of x within the range of fit fit 
%   y_fit    poloynomial at values of x_fit
%   Sc       standard errors of polynomial coefficients 
%   Sy_fit   standard errors of the curve-fit
%   Rc       parameter correlation matrix
%   R2       R-squared error criterion
%   Vr       unbiased variance of unweighted residuals
%   AIC      Akaike information criterion
%   condNo   condition number of the regularized system matrix 
%
% HP Gavin, 
% Fitting Models to Data: Generalized Linear Least Squares and Error Analysis
% https://people.duke.edu/~hpgavin/SystemID/linear-least-sqaures.pdf
%
% updated 2007-04-10, 2013-02-24, 2024-11-04

 if ( length(x) ~= length(y) )       % error checking
   disp(' the length of x must equal the length of y ');
   return
 end

 ee = 0.05;
 rfd = [ min(x) , max(x) ]*[ 1+ee -ee ; -ee 1+ee ]; % extrapolated range of fit
 if nargin < 4,  figNo = 0;  end       % default figure number
 if nargin < 5,  Sy  = 1.0;  end       % default measurement error
 if nargin < 6,  rof = rfd;  end       % default range of the fit 
 if nargin < 7,  b   = 0.0;  end       % default regularization parameter

 if any(Sy <= 0)                       % ensure that all vaues of Sy are positve
   Sy(find(Sy<=0)) = 1.0;
%  nargin = 4;                         % use Vr to compute invVy, Vc, and Sy_fit
 end

 x = x(:);                             % make "x" a column-vector
 y = y(:);                             % make "y" a column-vector
 p = p(:);                             % make "p" a column-vector

 Nd = length(x);                       % number of data points
 Np = length(p);                       % number of parameters
 Nf = 100;                             % number of values in the fit 
 x_fit = linspace(rof(1),rof(2),Nf)';  % x values for the fit

 ISy = eye(Nd)./(Sy.^2);       % inverse of the measuremen error covariance, Vy

 B     = zeros(Nd,Np);         % allocate memory for the model basis vectors
 B_fit = zeros(Nf,Np);         % allocate memory for the model basis vectors

 for i=1:Np
   B (:,i) = x.^p(i);          % the matrix of model basis vectors 
   B_fit(:,i) = x_fit.^p(i);   % the matrix of model basis vectors 
 end

 condNo = cond( B'*ISy*B + b*eye(Np) );              % condition number 

 c = inv( B'*ISy*B + b*eye(Np) ) * B'*ISy*y;         % least squares parameters

 y_fit = B_fit*c;                                    % least squares fit

 Vr = sum((y-B*c).^2) / (Nd-Np-1);  % unbiased variance of unweighted residuals

 if nargin < 5
   invVy = eye(Nd) / Vr;  % measurement error covariance computed from residuals
 else 
   invVy = ISy;           % measurement error covariance provided by user
 end

 Vc = inv( B'*invVy*B + b*eye(Np) );    % estimated parameter covariance

 if b ~= 0                              % regularized least squares 
   Vc = Vc * (B'*ISy*Vy*ISy*B) * Vc;    % parameter covariance w/ regularization
 end

 Sc = sqrt(diag(Vc));                   % standard error of the parameters

 Rc = Vc ./ (Sc * Sc');                 % parameter cross-correlation matrix

 Sy_fit = sqrt(diag(B_fit*Vc*B_fit'));  % standard error of the fit 

 R2  = 1 - sum( (y-B*c).^2 ) / sum( (y-sum(y)/Nd).^2 ); % R-squared = correlation

 AIC =  log(2*pi*Np*Vr) + (B*c-y)'*invVy*(B*c-y) + 2*Np; 

 disp('     p         a            +/-   da           (percent) ')
 disp('---------------------------------------------------------')
 for i=1:Np
   if rem(p,1) == 0
      fprintf('   a[%2d] =  %11.3e;    +/- %10.3e    (%7.2f %%)\n',  ...
                                p(i), c(i), Sc(i), 100*Sc(i)/abs(c(i)) ); 
   else
      fprintf(' %8.2f :  %11.3e     +/- %10.3e    (%7.2f %%)\n',  ...
                                p(i), c(i), Sc(i), 100*Sc(i)/abs(c(i)) ); 
   end
 end

if figNo % Plots ---------------------------------------
 
  formatPlot(12,2, 5);  % font size, line width, marker size

  figure(figNo);        % plot the datc, the model, and the confidence interval
    
   CI = [ 0.90  0.99 ];    % confidence intervals 
   z  = norminv( 1 - (1 - CI) / 2 );    % z-scores corresponding to confidence intervals

    % confidence interval (CI) of the model (y_fit)
    yps95 =  y_fit + z(1)*(Sy_fit + 0*sqrt(Vr));       %  + 90 CI
    yms95 =  y_fit - z(1)*(Sy_fit + 0*sqrt(Vr));       %  - 90 CI
    yps99 =  y_fit + z(2)*(Sy_fit + 0*sqrt(Vr));       %  + 99 CI
    yms99 =  y_fit - z(2)*(Sy_fit + 0*sqrt(Vr));       %  - 99 CI
    xp  = [ x_fit ; x_fit(end:-1:1) ; x_fit(1) ];    % x coordinates for patch
    yp95  = [ yps95 ; yms95(end:-1:1) ; yps95(1) ];  % y coordinates for patch
    yp99  = [ yps99 ; yms99(end:-1:1) ; yps99(1) ];  % y coordinates for patch

    patchColor95 = [ 0.95, 0.95, 0.1 ];              % color for the 95% CI
    patchColor99 = [ 0.2,  0.95, 0.2 ];              % color for the 99% CI

    clf
    subplot(121)        % plot the datc, curvefit, and CI's
    hold on
      hc99 = patch( xp, yp99, patchColor99);         % 99% CI
      hc95 = patch( xp, yp95, patchColor95);         % 95% CI
      hd   = plot( x, y,     'ob',  'LineWidth', 3); % data points
      hf   = plot( x_fit, y_fit, '-k');              % polynomial fit
    hold off
    set(hc95,'EdgeColor', patchColor95 );
    set(hc99,'EdgeColor', patchColor99 );
    xlabel('x') 
    ylabel('y')
    legend([hd, hf , hc95, hc99 ], 'data', 'y_{fit}', sprintf('%2.0f%% c.i.',CI(1)*100), sprintf('%2.0d%% c.i.',CI(2)*100) );
    axis([ min(xp) , max(xp) , min(y)-0.1*(max(y)-min(y)) , max(y)+0.1*(max(y)-min(y)) ])

% {
    subplot(122)        % plot the data vs the model and show correlation 
    hold on 
      plot( y, y,'-k', 'LineWidth', 0.5 )             % 1-to-1 line
      plot( B*c, y, 'ob', 'LineWidth', 3)             % data and model points
      tx = 0.90*min(y) + 0.10*max(y);
      ty = 0.02*min(y) + 0.98*max(y); 
      ry  = [ max(y) ; min(y) ];
      text(tx,[1.0,0.0]*ry, sprintf('condn # = %4.1f', condNo));
      text(tx,[0.9,0.1]*ry, sprintf('\\sigma_r = %5.3f', sqrt(Vr)));
      text(tx,[0.8,0.2]*ry, sprintf('AIC = %4.1f', AIC));
      text(tx,[0.7,0.3]*ry, sprintf('R^2 = %5.3f', R2));
      text(tx,[0.6,0.4]*ry, sprintf('n = %d', length(c) ));
    hold off
    xlabel('y_{fit}')
    ylabel('y')
    axis('tight')
% }

  figure(figNo+1)          % plot the histogram (distribution) of the residuals
    clf
    nBars = max(10,round(Nd/5)); 
    [fx,xx] = hist(y-B*c, nBars, nBars/(max(x)-min(x)));
      bar(xx,fx)        % histogram of the residuals
      xlabel('residuals, r = y - y_{fit}')
      ylabel('empirical PDF, f_R(r)')

  plotCDFci(y-B*c, 95, figNo+2);  % CDF of residuals with confidence intervals

end % ------------------------------------------------------- Plots

% ------------------------------------------------------------------ poly_fit

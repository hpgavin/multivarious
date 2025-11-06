function [x_avg, x_med, x_sd, x_cov] = plotCDFci(x,ci,figNo)
% [x_avg, x_med, x_sd, x_cov] = plotCDFci(x,ci,figNo)
%
% plot a CDF with confidence intervals
% 
%    INPUT         DESCRIPTION                                DIMENSION
%    =====         ================================           ================
%     x            a vector of a random sample                n x 1  or 1 x n
%     ci           a confidence level, eg 90 or 95            1 x 1
%    figNo         figure number                              1 x 1 
%   OUTPUT         DESCRIPTION                                DIMENSION
%   -=====         ================================           ================
%    x_avg         the average value of x                     1 x 1
%    x_med         the median  value of x                     1 x 1
%    x_sd          the standard deviation of x                1 x 1
%    x_cov         the coefficient of variation of x          1 x 1

 pdfPlots = 0;

 N = length(x);                         % number of values in the sample
 x = sort(x);                           % sort the sample
 x_avg = sum(x)/N;
 x_med = x(round(N/2));
 x_sd  = sqrt(sum((x-x_avg).^2)/(N-1));
 x_cov = abs(x_sd/x_avg);
 nBins = floor(N/5);                    % number of bins in the histogram
 [fx,xx] = hist(x,nBins);               % compute the histogram
  fx = fx / N * nBins/(max(x)-min(x));  % scale the histogram to a PDF
  Fx = ([1:N])/(N+1);                   % empirical CDF
 sFx = sqrt(Fx.*(1-Fx)/(N+2));          % standard error of Fx Gumbel p.47
%if x(N) > 1, 
%  probability_X_gt_1 = sum(x>1) / (N+1) % fraction of the sample for which X>1
%end


if figNo 

 ppl = [ 68, 1, 84 ]/256;    % purple to match histogram bars
 patchColor = [ 0.80 , 0.85 , 1 ];
 figure(figNo)
  clf
  z   = -norm_inv( (1-ci/100)/2 );
  Fps = min(Fx + z*sFx,1-0.001/N);
  Fms = max(Fx - z*sFx,0+0.001/N);
  [ xsp , Fsp ] = stairs( (x), Fps);
  [ xsm , Fsm ] = stairs( (x), Fms);
  xp = [ xsp ; xsm(end:-1:1) ; xsp(1) ];
  Fp = [ Fsp ; Fsm(end:-1:1) ; Fsp(1) ];
  hold on
  hc = patch(xp, Fp,  patchColor);
  hm = plot( x_med*[1,1,0] , 0.5*[0,1,1] , '--r', 'LineWidth', 2 );
  set(hc, 'EdgeColor', patchColor );
  ee = 0.05;
  xl =  (1+ee)*x(1) - ee*x(N);
  xh = -ee*x(1)     + (1+ee)*x(N);
  hx =  0.30*xl     + 0.7*xh;
% if  x(1) < 1 && x(N) > 1 
%   plot( [1,1,0] , (1-probability_X_gt_1)*[0,1,1] , '--k', 'LineWidth', 2 );
%   text (hx,0.5, sprintf('P[X>1] = %5.3f', probability_X_gt_1 )) 
% end
  text( x_med-0.05,-0.07, 'x_{med}', 'color','r')
  text (hx,0.4, sprintf('x_{avg} = %5.3f',x_avg))
  text (hx,0.3, sprintf('x_{med} = %5.3f',x_med))
  text (hx,0.2, sprintf('x_{sd}   = %5.3f',x_sd))
  if x_cov < 100
    text (hx,0.1, sprintf('x_{cov}  = %5.3f',x_cov))
  end
% title(sprintf(' x1 =  %f ', x(1)))

  stairs( x, Fx, 'color', ppl, 'LineWidth', 3.0 ); % plot empirical CDF + c.i.
  axis( [ xl  xh 0 1.05 ]);
  xlabel('x')
  ylabel('CDF       F_X(x)  =  Prob[ X<=x ]')

 if pdfPlots, print('plotCDFci.pdf','-dpdfcrop'); end

end


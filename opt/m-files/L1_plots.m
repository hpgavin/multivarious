function L1_plots( B, c, y, cvg_hst, alfa, w, fig_no )
% L1_plots( B, c, y, cvg_hst, alfa, w, fig_no )
% plot results from the fit of a linear model via L1 regularization 
% via L1_fit.m 

 [ n, max_iter ] = size(cvg_hst);
 n = (n-2)/5;    % number of coefficients in the linear model
 m = length(y);

 c0 = B\y;       % OLS model coefficients 
 y0 = B*c0;      % OLS model
 y1 = B*c;       % L1 model
 err_norm_0 = norm(y0-y)/(m-n) 
 err_norm_1 = norm(y1-y)/(m-n) 

 format bank 
 printf(' coefficients (alpha = %7.5f, w = %3.1f)\n',  0  , w ); disp(c0')
 printf(' coefficients (alpha = %7.5f, w = %3.1f)\n', alfa, w ); disp(c')
 
%disp(' Lagrange multipliers mu'); disp(mu')
%disp(' Lagrange multipliers nu'); disp(nu')
 format

 formatPlot(22,2,7); 

 figure(fig_no+1)
   clf
   hold on
   plot( B\y,'+r','MarkerSize',20, 'LineWidth',3)
   plot(  c, 'o', 'color', [0 0.8 0], 'MarkerSize',9, 'LineWidth',4)  
   plot( [0,n+1],[0,0], '--k')
    xlabel('coefficient index, i')
    ylabel('coefficients, c_i')
    legend('\alpha = 0', sprintf('\\alpha = %7.5f w=%3.1f',alfa,w) )

  figure(fig_no+2)
  clf
  plot([1:m],y,'ok', [1:m],y0, 'or',  [1:m],B*c,'o', 'color', [0 0.8 0]  )
  xlabel('i')
  ylabel('y_i')
  legend('data','\alpha = 0', sprintf('\\alpha = %7.5f w=%2.0f',alfa,w) )

 y_labels = { 'c', 'p', 'q', '\mu', '\nu' , '\alpha and L_2 error' };
 figure(fig_no+3)
  clf
  hold on
  for ii=1:5
    subplot(6,1,ii)
      plot([1:max_iter],cvg_hst( n*(ii-1)+1:n*ii , : ))
      ylabel(y_labels{ii})
    subplot(6,1,6)
      hold on
      semilogy([1:max_iter],cvg_hst( 5*n+1 , : ), '-g')
      semilogy([1:max_iter],cvg_hst( 5*n+2 , : ), '-k')
      ylabel(y_labels{6})
  end
  xlabel('L_1 iteration number')
   

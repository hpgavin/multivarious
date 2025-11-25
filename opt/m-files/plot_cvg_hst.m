function plot_cvg_hst ( cvg_hst, v_opt, figNo, clr )
% plot_cvg_hst ( cvg_hst, v_opt, figNo )
% plot the convergence history for a solution computed by ORSopt, NMAopt, SQPopt
% 
%  INPUT
% =======
% cvg_hst   : a record of p, f, and g as returned by ORSopt, NMAopt, or SQPopt
% v_opt     : the optimal design variables computed by ORSopt, NMAopt, or SQPopt
% figNo     : the figure number used for plotting, default FigNo = 1002
% clr       : colormap for subplot 2

% HP Gavin, Duke Univ., 2013-03-01, 2018-03-08, 2020-01-15

  if nargin < 3, figNo = 1002; end
  if nargin < 4, clr   = colormap; end

   maxiter = size(cvg_hst,2);		%  number of iterations
   n = length(v_opt);			%  number of design variables

   if maxiter > 100  pltstr = '-'; else pltstr = '-o'; end

   while n > size(clr,1), clr = [clr ; clr ]; end

   fc = cvg_hst(n+3,:);                 % function count
   lw = 3;                              % line width
   ms = 6;                              % marker size
 
if figNo  % --- make Plots -------------------------------------------------
formatPlot(14,2,4);
figure(figNo+1);	% plot the design variable  convergence criterion
 clf
 subplot(211)		% plot the objective function convergence criterion
    if ( max(cvg_hst(n+5,:)) > 100*min(cvg_hst(n+5,:)) && min(cvg_hst(n+4,:))>0)
      hP = semilogy(fc,cvg_hst(n+5,:),pltstr,'LineWidth',lw, 'MarkerSize',ms);
    else
      hP = plot(fc,cvg_hst(n+5,:),pltstr, 'LineWidth',lw, 'MarkerSize',ms);
    end
     ylabel('F convergence')
     %axis('tight')
 subplot(212)
    if ( max(cvg_hst(n+4,:)) > 100*min(cvg_hst(n+4,:)) && min(cvg_hst(n+4,:))>0)
      hP = semilogy(fc,cvg_hst(n+4,:),pltstr, 'LineWidth',lw, 'MarkerSize',ms);
    else
      hP = plot(fc,cvg_hst(n+4,:),pltstr, 'LineWidth',lw, 'MarkerSize',ms);
    end
     ylabel('X convergence')
     xlabel('function evaluations')
     %axis('tight')

  print(sprintf('plot_cvg_hst-%d.png',figNo+1),'-dpng');

figure(figNo);
    clf
    subplot(311);		% plot the objective function convergence
     cmin = min(cvg_hst(n+1,:));
     cmax = max(cvg_hst(n+1,:));
     rnge = cmax-cmin;
     if (cmax > 100*cmin && cmin+0.01*rnge > 0 && cmin > 0)
       hP = semilogy(fc,(cvg_hst(n+1,:))',pltstr, 'LineWidth',lw, 'MarkerSize',ms);
     else
       hP = plot(fc,(cvg_hst(n+1,:))',pltstr, 'LineWidth',lw, 'MarkerSize',ms);
     end
     ylabel('objective   f_A')
     %axis('tight')
     title(sprintf(' f_{opt} = %11.4e         max(g_{opt}) = %11.4e', ...
                                cvg_hst(n+1,maxiter), cvg_hst(n+2,maxiter)));

    subplot(312);		% plot the design variable convergence
     pmin = min(min(cvg_hst(1:n,:)));
     pmax = max(max(cvg_hst(1:n,:)));
     rnge = pmax-pmin;
     hold on
%    for ii = 1:n
       if (pmax/pmin > 100 && pmin-0.1*rnge > 0)
        hP = plot(fc,cvg_hst(1:n,:),pltstr, 'LineWidth',lw, 'MarkerSize',ms);
       else
        hP = plot(fc,cvg_hst(1:n,:),pltstr, 'LineWidth',lw, 'MarkerSize',ms);
       end
%    end
     ylabel('variables')
     %axis('tight')

    subplot(313);		% plot the max constraint convergence
     gmin = min(cvg_hst(n+2,:));
     gmax = max(cvg_hst(n+2,:));
     rnge = gmax-gmin;
     if (gmax/(gmin+0.01) > 100 && gmin-0.1*rnge > 0)
      hP = plot(fc,cvg_hst(n+2,:)',pltstr, 'LineWidth',lw, 'MarkerSize',ms);
     else
      hP = plot(fc,cvg_hst(n+2,:)',pltstr, 'LineWidth',lw, 'MarkerSize',ms);
     end
     ylabel('max(constraints)')
     xlabel('function evaluations')
     %axis('tight')

  print(sprintf('plot_cvg_hst-%d.png',figNo),'-dpng');
end % --- make Plots --------------------------------------------------

% ------------------------------------------------------------- plot_cvg_hst.m

function [fmin,fmax] = plot_surface(func,x,v_lb,v_ub,options,consts,fig_no)
% [f_min,f_max]=plot_surface(func,x,v_lb,v_ub,options,consts,fig_no)
%
% Draw a surface plot of J(x) vs. x(i),x(j), where all other values in x
% are held constant.   
%
% INPUT
% ======
%  func    :   the name of the matlab function to be optimized in the form
%               [objective, constraints] = func(x,consts)
%  x       :   the vector of initial design variable values ... a column vector
%  v_lb    :   lower bound on permissible values of the design variables, x
%  v_ub    :   upper bound on permissible values of the design variables, x
%  options :   options(1)  = 3  ... draw a surface
%              options(4)  = tol_g  tolerance on constraint functions
%              options(6)  = penalty  on constraint violations
%              options(7)  = exponent on constraint violations
%              options(11) = 1st index for plotting cost function surface
%              options(12) = 2nd index for plotting cost function surface
%              options(13) = number of 1st index values to plot cost funct surf
%              options(14) = number of 2nd index values to plot cost funct surf
%  consts  :   an optional vector of constats to be passed to func(x,consts)
%  fig_no  :   figure number used to draw the figure
%
% OUTPUT
% ======
%  fmin,fmax :  max and min values of the meshed surface data

%  H.P. Gavin, Civil & Environ. Eng'g, Duke Univ.  

  tol_g   = options(4);
  penalty = options(6);
  q       = options(7);
  i       = options(11);
  j       = options(12);
  Ni      = options(13);
  Nj      = options(14);

  v_init = x;

  v_i = linspace( v_lb(i), v_ub(i), Ni);
  v_j = linspace( v_lb(j), v_ub(j), Nj);
  f_mesh = NaN(Ni,Nj);
  for ii=1:Ni
      for jj=1:Nj
          x([i,j]) = [ v_i(ii) ; v_j(jj) ];
          [f,g] = feval(func,x,consts);
%         if g < 0
            f_mesh(ii,jj) = f + penalty*sum(g.*(g>tol_g))^q;
%         end
	  if ( penalty <= 0 &&  max(g) > tol_g ), f_mesh(ii,jj) = NaN; end
      end
  end
  frms = sqrt(sum(sum(f_mesh.^2)));
  fmin = 0.9*min(min(f_mesh));
  fmax = min(max(max(f_mesh)),5*frms);
  if (isnan(fmin) || isnan(fmax))
     fmin = 0; fmax = 1; 
  end
% cmap = prisim(100);
% cmap = rainbow(100);
% color_index = round(100*(f_mesh-fmin)/(fmax-fmin));  

  [ii,jj] = find(min(min(f_mesh)) == f_mesh);
  ii = min(ii);   % in case there are multiple minima with same objective
  jj = min(jj);   % in case there are multiple minima with same objective

  figure(fig_no)
    clf
    mesh(v_i,v_j,f_mesh', 'LineWidth',1.5)
%   plot3(v_i'*ones(1,Nj),ones(Ni,1)*v_j,f_mesh,'ok')
      xlabel(sprintf('v_%d',i))
      ylabel(sprintf('v_%d',j))
      zlabel(sprintf('objective   f_A(v_%d,v_%d)',i,j))
      axis([ min(v_i), max(v_i), min(v_j), max(v_j), fmin, fmax]);
    hold on
    [f,g] = feval(func,v_init,consts);
    fa = f + penalty*sum(g.*(g>tol_g))^q;
    plot3(v_init(i),v_init(j),fa+(fmax-fmin)/50,'og', 'MarkerSize',12, 'LineWidth',7);
    plot3(v_i(ii),v_j(jj),f_mesh(ii,jj)+(fmax-fmin)/50,'or', 'MarkerSize',12, 'LineWidth',7);
%   ttl = sprintf('grid opt: v_%d = %f , v_%d = %f , f(%d,%d) = %f',  i, v_i(ii), j, v_j(jj), v_i(ii),v_j(jj), f_mesh(ii,jj));
%   title(ttl);
 end


 

% plot_surface ================================================================
% updated 2015-09-29  2016-03-23  2016-09-12, 2020-01-20, 2021-12-31, 2025-01-26

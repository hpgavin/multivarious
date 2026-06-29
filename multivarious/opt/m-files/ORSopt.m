function [v_opt,f_opt,g_opt,cvg_hst]=ORSopt(func,v_init,v_lb,v_ub,options,consts)
% [v_opt,f_opt,g_opt,cvg_hst]=ORSopt(func,v_init,v_lb,v_ub,options,consts)
%
% ORSopt: Optimized Step Size Randomized Search
% Nonlienar optimization with inequality constraints using Random Search
%
%       minimizes f(v) such that g(v)<0 and v_lb <= v_opt <= v_ub.
%       f is a scalar objective function, v is a vector of design variables, and
%       g is a vector of constraints.
%
% INPUT
% ======
%  func    :   the name of the matlab function to be optimized in the form
%               [objective, constraints] = func(v,consts)
%  v_init  :   the vector of initial design variable values            (n x 1)
%  v_lb    :   lower bound on permissible values of the variables, v   (n x 1)
%  v_ub    :   upper bound on permissible values of the variables, v   (n x 1)
%  options :   options(1) = 1 means display intermediate results
%                           2 means display more intermediate results
%                           3, 4, ... even more and more intermediate results
%              options(2)  = tol_v  tolerance on convergence of design variables
%              options(3)  = tol_f  tolerance on convergence of objective
%              options(4)  = tol_g  tolerance on convergence of constraints
%              options(5)  = max_evals limit on number of function evaluations
%              options(6)  = penalty  on constraint violations
%              options(7)  = exponent on constraint violations
%              options(8)  = max number of function eval's in est. of mean f(v)
%              options(9)  = desired accuracy of mean f (as a c.o.v.)
%              options(10)  = 1  means stop when solution is feasible   
%  consts  :   an optional vector of constats to be passed to func(v,consts)
%
% OUTPUT
% ======
%  v_opt   :   a set of design variables at or near the optimal value
%  f_opt   :   the objective associated with the near-optimal design variables
%  g_opt   :   the constraints associated with the near-optimal design variables
% cvg_hst  :   record of v, f, g, function_count, and convergence criteria

 
% Sheela Belur V, An Optimized Step Size Random Search ,
% Computer Methods in Applied Mechanics & Eng'g, Vol 19, 99-106, 1979
%
% S.Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984
%
% modifications by:
% H.P. Gavin , Civil & Env'ntl Eng'g, Duke Univ.

% ``Consider everything. Keep the good. Avoid evil whenever you notice it.''

BOX = 1;

% handle missing arguments
 if nargin<3                      % default lower and upper bounds 
   v_lb = -1.0e2*abs(v_init);
   v_ub =  1.0e2*abs(v_init);
 end
 if nargin<5, options = []; end   % use default option values
 options = optim_options(options);
 if nargin<6, consts = 1.0; end;

 msglev    = options(1);         % 1: display intermediate values
 tol_v     = options(2);         % tolerance for convergence in design var's
 tol_f     = options(3);         % tolerance for convergence in objective fctn 
 tol_g     = options(4);         % constraint tolerance
 max_evals = options(5);         % maximum number of function evaluations
 penalty   = options(6);         % penalty  on constraint violations
 q         = options(7);         % penalty  function exponent
 find_feas = options(10);        % stop once a feasible solution is found

 v_lb=v_lb(:); v_ub=v_ub(:); v_init=v_init(:);	% make column vectors

 n = length(v_init);                    % number of variables

% check lower bounds bounds are less than upper bounds
 if any(v_ub < v_lb)
   disp('error: v_ub can not less than or equal to v_lb for any variable');
   v_opt = v_init;
   f_opt = (sqrt(5)-1)/2;
   g_opt = 1;
   return
 end

 % initialize variables
 function_count = 0;			% number of function evaluations
 iteration = 1;
 cvg_f = 1;
 cvg_hst = NaN(n+5,max_evals);	        % convergence history
 fa = zeros(4,1);			% augmented cost function

 % scale v from [ v_lb < v < v_ub ] to x [ -1 < x < +1 ]
 s0 = (v_lb+v_ub)./(v_lb-v_ub);  s1 = 2./(v_ub-v_lb);  % scale to [-1:+1]
 x_init = s0+s1.*v_init;                % scale x to [-1:+1]
 x_init = min(max(x_init,-1.0),1.0);

 i3 = 1e-9*eye(3);               % for regularization 

 % evaluate the initial guess and the range of allowable variable variation
 [fv,gv,v_init,cJ,nAvg] = avg_cov_func(func,x_init,s0,s1,options, consts,1);
 g_max = max(gv);
 function_count = function_count + nAvg;

 m = length(gv);                 % number of constraints

 % dimensions of the objective function (1x1) and the constraints (mx1)
 if prod(size(fv)) ~= 1
  error('the objective computed by your analysis function must be a scalar')
 end
 if size(gv,1) == 1 && size(gv,2) > 1
  error('the constraints computped by your analysis function must be a column vector')
 end


 if ( msglev > 2 ) % objective landscape in terms of two selected design var's
   [f_min,f_max]=plot_surface(func,(x_init-s0)./s1,v_lb,v_ub,options,consts,103);
 end

% sigma: the standard deviation of fractional random perturbations 0 < sigma < 1
% larger  sigma ... more "broad exploring"
% smaller sigma ... more "narrow focusing"
 sigma = 0.200;  % standard deviation of fractional random perturbations
 nu    = 2.5;    % exponent for reducing sigma at each iteration 

 if msglev, ORS_tic_id = tic; end

% analyze initial guess
 x1 = x_init;
 [fa(1),g1,x1,c1,nAvg] = avg_cov_func(func,x1,s0,s1,options,consts,BOX);
 function_count = function_count + nAvg;
 
% initialize optimal values 
 f_opt = fa(1);  x_opt = x1;  g_opt = g1;

% save the initial guess to the convergence history 
 cvg_hst(:,iteration)=[(x_opt-s0)./s1;f_opt;max(g_opt);function_count;sigma;1];

 if msglev, more off; end

 x4 = x1; g4 = g1; fa(4) = fa(1);

 last_update = function_count;

 while ( function_count < max_evals )		% === main loop ===

   r = sigma*randn(n,1);		% random search perturbation

   a2 = box_constraint(x1,r,n);
   x2 = x1 + a2*r;			% 1st random perturbation +1*r

   [fa(2),g2,x2,c2,nAvg] = avg_cov_func(func,x2,s0,s1,options,consts,BOX);
   function_count = function_count + nAvg;

   if ( fa(2) < fa(1) ),  step = +2;  else  step = -1;  end
   a3 = box_constraint(x1,step*r,n);
   x3 = x1 + a3*step*r;			% 2nd random perterbation -1*r or +2*r 
   [fa(3),g3,x3,c3,nAvg] = avg_cov_func(func,x3,s0,s1,options,consts,BOX);
   function_count = function_count + nAvg;

   dx2 = norm(x2-x1)/norm(r);
   dx3 = norm(x3-x1)/norm(r);

   abc = ( [ 0        , 0   , 1 ; 
            0.5*dx2^2 , dx2 , 1 ; 
            0.5*dx3^2 , dx3 , 1 ] + i3 ) \ fa(1:3);

   a = abc(1);   b = abc(2);   c = abc(3); 

   quad_update = 0;
   if a > 0 				% curvature is positive!
      d = -b/a;                         % try to go to zero-slope point
      a4 = box_constraint(x1,d*r,n);
      x4 = x1 + a4*d*r;
      [fa(4),g4,x4,c4,nAvg] = avg_cov_func(func,x4,s0,s1,options,consts,BOX);
      function_count = function_count + nAvg;
   end

   [fa(1),i_min]=min(fa);               % find the lowest fa of 4 evaluations

   if ( i_min == 2 ), 	x1=x2; g1=g2; c1=c2; end % and the corresponding x,g,c
   if ( i_min == 3 ),	x1=x3; g1=g3; c1=c3; end
   if ( i_min == 4 ),	x1=x4; g1=g4; c1=c4; quad_update = 1; end

   % did the solution improve? 
   if ( i_min > 1 )  % slightly reduce the scope of the search and continue
%     sigma = sigma / 1.15;
      sigma = sigma * ( 1 - function_count/max_evals )^nu; 
%     sigma = sigma*(phi*exp(-function_count^2/0.2/max_evals^2) + (1-phi));
   end

   x1 = min(max(x1,-1),+1);        % ... keep x within bounds -1 < x < 1

   if ( fa(1) < f_opt )  % update the optimal solution 

      x_opt = x1;   
      f_opt = fa(1);   
      g_opt = g1;

      % convergence criteria
      cvg_v = norm( cvg_hst(1:n,iteration) - (x_opt-s0)./s1 ) ./ norm( (x_opt-s0)./s1 );
      cvg_f = norm( cvg_hst(n+1,iteration) - f_opt ) / norm(f_opt);

      last_update = function_count;
      iteration   = iteration + 1; % increment the iteration counter
      cvg_hst(:,iteration) = ...
            [ (x_opt-s0)./s1; f_opt; max(g_opt); function_count; cvg_v; cvg_f ];

      if msglev                     % display some results ----

         secs_left = round((max_evals-function_count)*toc(ORS_tic_id)/function_count);
         eta       = datestr(now+secs_left/3600/24,14);

         [max_g, idx_ub_g] = max(g_opt);
         home
         fprintf(1,' -+-+-+-+-+-+-+-+-+-+- ORSopt -+-+-+-+-+-+-+-+-+-+-+-+-+\n')
         fprintf(1,' iteration               = %5d', iteration );
         if max(g_opt) > tol_g 
            fprintf(1,'     !!! infeasible !!!\n');
         else
            fprintf(1,'     ***  feasible  ***\n');
         end
         fprintf(1,' function evaluations    = %5d  of  %5d  (%4.1f%%) \n', ...
                     function_count, max_evals, 100*function_count/max_evals );
         fprintf(1,' e.t.a.                  = %s\n', eta );
         fprintf(1,' objective               = %11.3e\n', f_opt );
         fprintf(1,' variables               = ');
         fprintf(1,'%11.3e', (x_opt-s0)./s1 );
         fprintf(1,'\n');
         fprintf(1,' max constraint          = %11.3e (%d) \n', max_g, idx_ub_g );
         fprintf(1,' Convergence Criterion F = %11.4e    tolF = %8.6f\n',cvg_f, tol_f);
         fprintf(1,' Convergence Criterion X = %11.4e    tolX = %8.6f\n',cvg_v, tol_v);
         fprintf(1,' c.o.v. of f_a           = %11.3e\n', c1 );
         fprintf(1,' step std.dev            = %5.3f\n', sigma );

         fprintf(1,' -+-+-+-+-+-+-+-+-+-+- ORSopt -+-+-+-+-+-+-+-+-+-+-+-+-+\n')
         if quad_update
             fprintf(1,' line quadratic update successful\n');
             %%pause(1);
         end
      end                          % ---- display some results 
   end                             % ----- update the optimal point

   if ( msglev > 2 )		   % plot values on the surface ----
     v1 = (x1-s0)./s1; v2 = (x2-s0)./s1; v3 = (x3-s0)./s1; v4 = (x4-s0)./s1;
     f_offset = (f_max-f_min)/100;
     figure(103)
       ii = options(11);  jj = options(12);
       if step == -1
         plot3( [v2(ii),v1(ii),v3(ii)] , [v2(jj),v1(jj),v3(jj)], ...
                        fa([2,1,3]) + f_offset, '-or', 'MarkerSize', 4); 
       else
         plot3( [v1(ii),v2(ii),v3(ii)] , [v1(jj),v2(jj),v3(jj)], ...
                        fa([1,2,3]) + f_offset, '-or', 'MarkerSize', 4); 
       end
       if quad_update
         plot3([v4(ii)],[v4(jj)],fa([4]) + f_offset, 'ob', 'MarkerSize', 9, 'LineWidth', 3 );
       end
       drawnow
   end                          % ---- plot values on the surface

   if ( msglev > 3 && a > 0 )
     figure(104);
       clf
       xx = [min([0,a3*step,a4*d])-0.5:0.1:max([a2,a3*step,a4*d])+0.5];
       ff = 0.5*a*xx.^2 + b*xx + c;
       hold on
       plot(sigma*xx,ff,'-', 'LineWidth', 6 );
       plot(sigma*[0,a2,a3*step,a4*d], fa,'o', ...
                                      'MarkerSize',19, 'LineWidth', 6 )
       plot(sigma*a4*d,fa(4),'*', 'MarkerSize',19, 'LineWidth', 6 )
       hold off
       title(sprintf(' iter:  %d  |  eval:  %d  |  f_{opt} = %11.4e  |  \\sigma = %5.3f', iteration, function_count, min(fa), sigma ));
       xlabel('change in design variables , \Delta x')
       ylabel('augmented objective function, f_{aug}(x)')
       grid on;  axis('tight');  drawnow;   
   end

   if ( msglev > 3 && a < 0 )
      figure(104);
        clf
        plot( sigma*[0,a2,a3*step], fa(1:3),'o' , 'MarkerSize',19, 'LineWidth',4 )
%          title(sprintf(' iter:  %d  |  eval:  %d  |  f_{opt} = %11.4e  |  \\sigma = %5.3f', iteration, function_count, min(fa), sigma ));
          grid on;  axis('tight');  drawnow;    
   end

   % check convergence criteria and display termination information -----

   if ( max(g_opt) < tol_g && find_feas ) % a feasible point has been found
      fprintf(1,' Woo Hoo!  Feasible solution found! ')
      fprintf(1,' ... and that is all we are asking for.\n')
      break;                       % ... and that's all we want
   end

   if ( iteration > n*n && ( (cvg_v < tol_v) || (cvg_f < tol_f)  
                           || (function_count-last_update>0.2*max_evals) ) )
      fprintf(1,' *** Woo-Hoo!  Converged solution found! \n')
      if ( cvg_v < tol_v )
        fprintf(1,' ***           convergence in design variables \n');
      end
      if ( cvg_f < tol_f )
        fprintf(1,' ***           convergence in design objective \n');
      end
%     [f_opt,g_opt] = feval(func,(x_opt-s0)./s1,consts);
      if (max(g_opt) < tol_g )
        fprintf(1,' *** Woo-Hoo!  Converged solution is feasible! \n')
      else
        fprintf(1,' *** Boo-Hoo!  Converged solution is NOT feasible! \n')
        if max(g_opt) > tol_g
           fprintf(1,'     ... Increase options(6) and try, try again ...\n')
        else
           fprintf(1,'     ... Decrease options(6) and try, try again ...\n')
        end
      end
      if msglev, format, end
      break;
   end                         % ----- check convergence criteria

 end						% === main loop ===

 if function_count >= max_evals        % time-out --  
   if msglev
      disp([' Enough!!  Maximum number of function evaluations (', ...
              int2str(max_evals),') has been exceeded']);
      disp('     ... Increase tol_v (options(2)) or max_evals (options(5)) and try try again.')
      format
      %% current_options = options
   end
 end                                   % -- time-out

 v_init = (x_init-s0)./s1;              % scale back to original units
 v_opt  = (x_opt-s0)./s1;               % scale back to original units

 fprintf(1,' *** Completion  : %s (%s)\n', ...
 datestr(now,14), datestr(toc(ORS_tic_id)/3600/24,13));
 fprintf(1,' *** Objective   : %11.3e \n', f_opt);
 fprintf(1,' *** Variables   : \n');
 fprintf(1,'               v_init         v_lb     <     v_opt    <     v_ub\n')
 fprintf(1,'------------------------------------------------------------------\n');
 for ii = 1:n
   eqlb = ' ';
   equb = ' ';
   if ( v_opt(ii) < v_lb(ii) + tol_g + 10*eps )
     eqlb = '=';
   elseif ( v_opt(ii) > v_ub(ii) - tol_g - 10*eps )
     equb = '=';
   end
   fprintf(1,'x(%3u)  %12.5f   %12.5f %s %12.5f %s  %12.5f \n',ii,v_init(ii),v_lb(ii),eqlb,v_opt(ii),equb,v_ub(ii));
 end

 fprintf(1,' *** Constraints : \n');
 for jj=1:m
    binding = ' ';
    if ( g_opt(jj) > -tol_g )
      binding = ' ** binding ** ';
    end
    if ( g_opt(jj) > tol_g )
      binding = ' ** not ok  ** ';
    end
    fprintf(1,'       g(%3u) = %12.5f  %s \n', jj, g_opt(jj), binding );
 end
 fprintf(1,'--------------------------------------------------------------\n\n')

 if msglev, format; end

 iteration = iteration + 1;            % save final iteration information 
 cvg_hst(:,iteration) = [v_opt; f_opt; max(g_opt); function_count; cvg_v; cvg_f];
 cvg_hst(n+5,[1,2]) = [1,1]*cvg_hst(n+5,3);
 cvg_hst = cvg_hst(:,1:iteration);

end
% ORSopt ======================================================================
% updated 2011-04-13, 2014-01-12, 2015-03-14, (pi day 03.14.15), 2015-03-26, 
% 2016-04-06, 2019-02-23, 2020-01-17, 2024-04-03, 2025-11-24

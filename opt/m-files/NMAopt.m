function [x_opt,f_opt,g_opt,cvg_hst,HH,ll] = NMAopt(func,x_init,x_lb,x_ub,options,consts)
% [x_opt,f_opt,g_opt,cvg_hst] = NMAopt(func,x_init,x_lb,x_ub,options,consts)
%
% NMAopt: 
% Nelder-Mead method for the nonlinear optimization with inequality constraints 
%
%       minimizes f(x) such that g(x)<0 and x_lb <= x_opt <= x_ub.  
%       f is a scalar objective function, x is a vector of design variables, and
%       g is a vector of constraints.  
%
% INPUT
% ======
%  func    :   the name of the matlab function to be optimized in the form
%               [ objective, constraints ] = func( x, consts )
%  x_init  :   the vector of initial parameter values                   (n x 1)
%  x_lb    :   lower bound on permissible values of the variables, x    (n x 1)
%  x_ub    :   upper bound on permissible values of the variables, x    (n x 1)
%  options :   options(1)  = 1  means display intermediate results
%              options(2)  = tol_x  tolerance on convergence of variables
%              options(3)  = tol_f  tolerance on convergence of objective
%              options(4)  = tol_g  tolerance on convergence of constraints
%              options(5)  = max_evals limit on number of function evaluations
%              options(6)  = penalty  on constraint violations
%              options(7)  = exponent on constraint violations
%              options(8)  = max number of function eval's in est. of mean f(x)
%              options(9)  = desired accuracy of mean f (as a c.o.v.)
%              options(10) = 1  means stop when solution is feasible      
%  consts  :   an optional vector of values that are not design variables
%
% OUTPUT
% ======
%  x_opt   :   a set of design variables at or near the optimal value
%  f_opt   :   the objective associated with the near-optimal design variables
%  g_opt   :   the constraints associated with the near-optimal design variables
% cvg_hst  :   record of x, f, g, function_count, and convergence criteria
 
% H.P. Gavin , Civil & Env'ntl Eng'g, Duke Univ. 

% The algorithm of NMAopt is a Nelder-Mead simplex search method.
% For optimization with respect to N variables, a simplex is a 
% combination of N+1 parameter sets.   It is represented by a 
% N by N+1 matrix.  Each column of the simplex matrix represents 
% a set of design variables, and each set of design variables has an associated
% value of the objective function.  By convention, the parameter 
% columns in the simplex are sorted by increasing value of the 
% objective function. 

% J.A. Nelder and R. Mead, A simplex method for function minimization,' 
% Computer Journal, 7(4)(1965):308-313.
%
% S. Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984
% S. Rao, Computer Methods in Applied Mechanics & Engg,  19(1979):99-106.
%
% J.E. Dennis, Jr. and D.J. Woods,
% New Computing Environments: Microcomputers in Large-Scale Computing,
% edited by A. Wouk, SIAM, (1987):116-122.
%
% http://en.wikipedia.org/wiki/Nelder-Mead_method

% handle missing arguments
if nargin<3                      % default values
  x_lb = -1.0e2*abs(x_init);
  x_ub =  1.0e2*abs(x_init);
end
if nargin<5, options = []; end   % use default values
options = optim_options(options);
if nargin<6, consts = 1; end
HH = 0; ll=0;

msglev    = options(1);         % 1: display intermediate values
tol_x     = options(2);         % tolerance for convergence in x
tol_f     = options(3);         % tolerance for convergence in f
tol_g     = options(4);         % tolerence for active constraints g
max_evals = options(5);         % maximum number of function evaluations
penalty   = options(6);         % penalty  on constraint violations
q         = options(7);         % exponent on constraint violatoins
find_feas = options(10);        % only find a feasible solution

optimize_contraction = 0;       % 1: try to optimize contraction; 0: don't

BX = 1;                         % 1: bound x within [x_lb,x_ub];

n = prod(size(x_init));         % number of design variables

cvg_hst = NaN(n+5,max_evals);   % init. matrix to record convergence history

function_count = 0;             % the number of function evaluations
iteration = 1;                  % iteration counter
cvg_hst_file = 1;               % 2: write convergence history to a file, 1:no
chfp = 0;                       % convergence history file pointer

x_lb = x_lb(:); x_ub = x_ub(:); x_init = x_init(:); % make column vectors

x_ub(find(x_ub == 0)) = 1e-6;  % a parameter bound should not be zero
x_lb(find(x_lb == 0)) = 1e-6;  % a parameter bound should not be zero

if ( msglev == 3 )
  [f_min,f_max]=plot_surface(func,x_init,x_lb,x_ub,options,consts,103);
end

s0 = (x_lb+x_ub)./(x_lb-x_ub);  s1 = 2./(x_ub-x_lb);  % scale to [-1:+1]
x_init = s0+s1.*x_init;                                   % scale x to [-1:+1]
x_init = min(max(x_init,-0.8),0.8);             % x_init not too close to edge
 
% standard  Nelder-Mead algorithm coefficients ... and ...
% Gao + Han, Comput Optim Appl (2012) 51:259-277
%            Gao+Han         Standard
a_reflect  = 1;  
a_extend   = 1 + 2/n;        % 2;  
a_contract = 0.75 - 0.5/n;   % 0.5;  
a_shrink   = 1 - 1/n;        % 0.5;
a_expand   = 1.3;

NMA_tic_id = tic;

% Evaluate the initial guess and the range of allowable parameter variation
if any(x_ub <= x_lb)
  disp('error: x_ub can not less than or equal to x_lb for any parameter');
  x_opt = (x_init-s0)./s1;
  f_opt = (sqrt(5)-1)/2;
  g_opt = 1;
  return
else
  [f,g,x_init,cx,nAvg] = avg_cov_func(func,x_init,s0,s1,options, consts,1);
  g_max = max(g);
  function_count = function_count + nAvg;

  if prod(size(f)) ~= 1
    error('the objective computed by your analysis function must be a scalar')
  end
  if size(g,1) == 1 && size(g,2) > 1
    error('the constraints computped by your analysis function must be a column vector')
  end
end
m = length(g);     % number of constraints 

% allocate memory for simplex, objectives, constraints, cov's, 
simplex = NaN(n,n+1);
gv      = NaN(m,n+1);
fv      = NaN(1,n+1);
cJ      = NaN(1,n+1);
n1      = [1:n+1];
vo      = NaN(n,n+1);               % opposite face points
lo      = NaN(1,n+1);               % distance to opposite face points

simplex(:,1) = x_init; 

% save the initial guess ...
cvg_hst(:,iteration) = [ (x_init-s0)./s1 ; f ; max(g) ; function_count ; 1 ; 1 ];

% Set up the initial simplex near the initial guess.
% The initial simplex should not be too small.  
% delta_x = min( 0.3*(1+abs(x_init)) , 0.2); 
% idx = find(delta_x == 0);
% delta_x(idx) = 0.2;

% Include the initial guess in the simplex! (credit L.Pfeffer at Stanford)
fv(1)   = f; 
gv(:,1) = g; 
cJ(1)   = cx;
% equilateral simplex   (Standford AA-222 - chapter 6 course notes)
cc = 0.3;         
bb = cc/(sqrt(2)*n) * ( sqrt(n+1) - 1 );
aa = bb + cc/sqrt(2);
for i = 1:n                  % set up initial simplex ----
  delta_x = bb*ones(n,1);
  delta_x(i) = aa;
  x = x_init + delta_x;;
  [f,g,x,ci,nAvg] = avg_cov_func( func, x, s0,s1, options, consts, 1);
  j = i+1;
  simplex(:,j) = x;
  cnstrts(:,j) = g;
  fv(j)    = f;
  gv(:,j)  = g;
  cJ(j)    = ci;
  g_max(j) = max(g);
  function_count = function_count + nAvg;
end                            % ---- set up initial simplex

on1 = ones(1,n+1);

% SORT the vertices in increasing order of fv
[fv,idx]=sort(fv); simplex=simplex(:,idx); gv=gv(:,idx); g_max=g_max(idx); cJ=cJ(idx);

if msglev                        % display initial information ----
   more off
   format compact
   format short e
   home
   disp('========================= NMAopt ===========================')
   function_count
   disp('initial ')
   disp(' ')
   if (n<10 && msglev == 2)
     (simplex-s0*on1)./(s1*on1) 
     fv
     g_max
   end
   disp('========================= NMAopt ============================')
end                             % ---- display initial information

f_old = fv(1);                  % the best point in the initial simplex
last_update = function_count;

% evaluate convergence criteria for the initial simplex

convergence_criteria_x = ...
 2*max(max(abs( ([simplex(:,1:n),simplex(:,1)] - [simplex(:,2:n+1),simplex(:,n+1)]) ./ ...
                ([simplex(:,1:n),simplex(:,1)] + [simplex(:,2:n+1),simplex(:,n+1)]+1e-9) )));
convergence_criteria_f = abs((fv(n+1)-fv(1))/(fv(1)+1e-9));

iteration = iteration+1;

% save the best vertex of the initial simplex ...
cvg_hst(:,iteration) = [ (simplex(:,1)-s0)./s1 ; fv(1) ; g_max(1) ; 
            function_count ; convergence_criteria_x ; convergence_criteria_f ];

% Iterate long enough or until convergence 
while ( function_count < max_evals )        % === main loop ===

    % one step of the Nelder-Mead simplex algorithm ...

    accept_point = 0;

    xo = (sum(simplex(:,1:n)')/n)';   % average of the best n vertices

% REFLECT the worst point through the average of the best n vertices
    xr = xo + a_reflect*(xo-simplex(:,n+1));
    [fr,gr,xr,cj,nAvg] = avg_cov_func(func, xr, s0,s1, options, consts, BX);
    if ( fv(1) <= fr &&  fr < fv(n) )  % fr is between best and second worst
       xw = xr;  fw = fr;  gw = gr;  cw = cj;  move_type = 'reflect';
       accept_point = 1;
    end                            % reflect
    function_count = function_count + nAvg; 

% EXTEND the worst point further past the average of the best n vertices
    if ( accept_point == 0 && fr < fv(1) )    % fr is better than best 
       xe = xo + a_extend*(xr-xo);
       [fe,ge,xe,cj,nAvg] = avg_cov_func(func, xe, s0,s1, options, consts, BX);
       if fe < fr
          xw = xe;  fw = fe;  gw = ge;  cw = cj;  move_type = 'extend';
       else
          xw = xr;  fw = fr;  gw = gr;  cw = cj;  move_type = 'reflect';
       end
       accept_point = 1;
       function_count = function_count + nAvg;
    end                            % extend

% CONTRACT the worst point to a point between the worst point and point xr
    if accept_point == 0 && fr > fv(n)   % fr is worse than next-to-worst
       xci = xo - a_contract*(xr-xo);    %  inside 
       xco = xo + a_contract*(xr-xo);    % outside 
       [fci,gci,xci,ci,nAvg] = avg_cov_func(func, xci, s0,s1, options, consts, BX);
       function_count = function_count + nAvg;
       [fco,gco,xco,co,nAvg] = avg_cov_func(func, xco, s0,s1, options, consts, BX);
       function_count = function_count + nAvg;

       if optimize_contraction     % try to optimize the contraction step
          d  = [ -norm(xo-simplex(:,n+1)) ;
                 -norm(xo-xci) ;
                  norm(xo-xco) ;
                  norm(xo-xr)  ];
          a = [ ones(4,1) d  0.5*d.^2 ] \ [ fv(n+1) ; fci ; fco ; fr ];
          dx = - a(2) / a(3);

          if abs(dx) < d(4) && a(3) > 0 
             xc_opt  = xo + dx*(xr-xo)/d(4);  % 'optimal contraction'
             [fc_opt,gc_opt,xc_opt,cj,nAvg] = ...
                          avg_cov_func(func, xc_opt, s0,s1, options,consts, BX);
             function_count = function_count + nAvg;

             if fc_opt < min(fci,fco) && fc_opt < fv(n)
                xw = xc_opt;  fw = fc_opt;  gw = gc_opt;  cw = cj;
                move_type = 'contract opt';
                accept_point = 1;
             end
             function_count = function_count + nAvg;
          end
       end                         % end optimize_contraction

       if accept_point == 0 && fci < fco && fci < fv(n)
             xw = xci;  fw = fci;  gw = gci;  cw=ci; move_type = 'contract in';
             accept_point = 1;
       end
       if accept_point == 0 && fco < fci && fco < fv(n)
             xw = xco;  fw = fco;  gw = gco;  cw=co; move_type = 'contract out';
             accept_point = 1;
       end
    end                            % contract

% If a new point is accepted, replace the worst point (point n+1) with it.

    if ( accept_point == 1 )

       simplex(:,n+1) = xw; 
       fv(n+1)        = fw; 
       gv(:,n+1)      = gw; 
       g_max(n+1)     = max(gw); 
       cJ(n+1)        = cw;

    else

% Otherwise, SHRINK into the best point (point 1).

       x1 = simplex(:,1);
       for i=2:n+1
           xs = x1 + a_shrink*(simplex(:,i)-x1);
           [fs,gs,xs,cj,nAvg] = avg_cov_func(func, xs, s0,s1, options,consts, BX);
           simplex(:,i) = xs;
           fv(i)        = fs;
           gv(:,i)      = gs;
           cJ(i)        = cj;
           function_count = function_count + nAvg;
       end                         % shrink
       move_type = 'shrink';

    end

% EXPAND elongated (degenerate) simplex
% {
    for i=1:n+1
      idx = n1(find(n1 ~= i));
      vo(:,i) = (sum(simplex(:,idx)')/n)';    % centroid of opposite n vertices
      lo(i) = norm( simplex(:,i) - vo(:,i) ); % vertex to opposite face distnc
    end
    idx =  find ( lo/max(lo) < (a_expand-1) );   % elongated simplex
    for i=1:length(idx)
      j = idx(i);
      xx = vo(:,j) + a_expand*(simplex(:,j) - vo(:,j)); % expand 
      [fx,gx,xx,cj,nAvg] = avg_cov_func(func, xx, s0,s1, options,consts, BX);
      simplex(:,j) = xx;
      fv(j)   = fx;
      gv(:,j) = gx; 
      cJ(j)   = cj;
      function_count = function_count + nAvg;
    end                            % expand
    if length(idx) > 0
      for i=1:n+1                  % distances from vertices to opposite face
        fprintf(1,'vertex %d distance = %f ... ratio = %f  \n', i, lo(i), lo(i)/max(lo))
      end  
%     pause(0.5)
    end
% }

% SORT the vertices in increasing order of fv
    [fv,idx]=sort(fv); simplex=simplex(:,idx); gv=gv(:,idx); g_max=g_max(idx); cJ=cJ(idx);
    x_opt = simplex(:,1);
    f_opt = fv(1);
    g_opt = gv(:,1);
    if (f_opt < f_old)
       last_update = function_count;
       f_old = f_opt;
    end                            % sort

% evaluate convergence criteria

    convergence_criteria_x = ...
               2*max(max(abs( (simplex(:,1:n) - simplex(:,2:n+1)) ./ ...
                              (simplex(:,1:n) + simplex(:,2:n+1)+1e-2) )));
    convergence_criteria_f = abs((fv(n+1)-fv(1))/(fv(1)+1e-2));

% update record of convergence history

    iteration = iteration + 1;     % increment the iteration counter

    cvg_hst(:,iteration) = [ (x_opt-s0)./s1 ; f_opt ; max(g_opt) ; function_count ;
                            convergence_criteria_x ; convergence_criteria_f ];

    secs_left = round((max_evals-function_count)*toc(NMA_tic_id)/function_count);
    eta       = datestr(now+secs_left/3600/24,14);

    if ( msglev == 4 )   % write convergence history to file
       secs_left
       chfp_filename = sprintf('NMA_opt_cvg_hst_%s.txt', datestr(now,30) );
       chfp = fopen(chfp_filename, 'w' ) 
       cvg_hst_file = 2
%      pause(5)
       fprintf(chfp,'%%  =========================================== \n');
       fprintf(chfp,'%%  %s\n', chfp_filename);
       fprintf(chfp,'%%  Nelder Mead Algorithm converence history ETA: %s \n', eta );
       fprintf(chfp,'%%  =========================================== \n');
       fprintf(chfp,'%%  x_opt f_opt g_opt function_count convergence_criterial_x convergence_criteria_f \n');
       fprintf(chfp,'%%  =========================================== \n');
       for ii=1:3
          for jj=1:n+5 
             fprintf(chfp,'%15.7e\t',cvg_hst(jj,ii));
          end
          fprintf(chfp,'\n');
       end
    end
    if ( cvg_hst_file==2 && iteration > 3 )   % write convertence history to file
       for jj=1:n+5 
          fprintf(chfp,'%15.7e\t',cvg_hst(jj,iteration));
       end
       fprintf(chfp,'\n');
    end

    if ( msglev )                            % display some results -----
        
       home
       disp('========================= NMAopt ===========================')

       fprintf(1,'iteration            = %5d', iteration);
       if max(g_opt) > tol_g
           fprintf(1,'           !!! infeasible !!!\n');
       else
           fprintf(1,'           ***  feasible  ***\n');
       end
       fprintf(1,'                                          ');
       fprintf(1,'  %s\n', move_type );
       fprintf(1,'function evaluations = %6d  of  %6d  (%4.1f%%) \n', ...
                     function_count, max_evals, 100*function_count/max_evals );
       fprintf(1,'e.t.a.               = %s\n', eta );
       disp(' ')
       if (n<10 && msglev > 0)
           Simplex = (simplex-s0*on1)./(s1*on1)
           F = fv 
           G_max = g_max
           COV = cJ
%          fprintf(1,'G_max = %11.4e\n', g_max(1) );
       end
       if (n>9 && msglev == 1)
           fprintf(1,'F_opt = %11.4e\n', f_opt );
           fprintf(1,'G_max = %11.4e\n', max(g_opt) );
       end
       if (n>9 && msglev == 2)
           Simplex = ((simplex(:,1)-s0)./s1)'
           fprintf(1,'F_opt = %11.4e\n', f_opt );
           fprintf(1,'G_max = %11.4e\n', max(g_opt) );
       end
       fprintf(1,'Convergence Criterion X = %11.4e    tolX = %10.6f\n',convergence_criteria_x, tol_x);
       fprintf(1,'Convergence Criterion F = %11.4e    tolF = %10.6f\n',convergence_criteria_f,tol_f);
       fprintf(1,'c.o.v. of J_a           = %11.4e\n', cJ(1) );
       disp('========================= NMAopt ============================')
    end                                     % ----- display some results

    if ( msglev == 3 )          %  plot the simplex on the surface ---
     ii = options(11);  jj = options(12);
     figure(103)
      simplex = (simplex-s0*on1)./(s1*on1);
      plot3( simplex(ii,[1:3,1]), simplex(jj,[1:3,1]), ...
                                   fv([1:3,1])+(f_max-f_min)/100,'-or', 'MarkerSize',6 )
      drawnow
      simplex = s0*ones(1,n+1) + (s1*ones(1,n+1)).*simplex;
      %pause(1)
    end                        % --- plot the simplex on the surface

% check convergence criteria and display termination information

    if ( max(g_opt) < tol_g && find_feas ) % a feasible point has been found
      for fp=1:2
        if ( fp==2 ) , fp = chfp; end
        fprintf(fp,'%%  Woo-Hoo!  Feasible solution found!\n')
        fprintf(fp,'%% ... and that is all we are asking for.\n')
      end
      break;                       % ... and that's all we want.  
    end

    if ( convergence_criteria_x < tol_x || convergence_criteria_f < tol_f )
      for fp=1:cvg_hst_file
        if ( fp==2 ) , fp = chfp; end
        fprintf(fp,'%% *** Woo-Hoo!  Converged solution found! \n')
        if ( convergence_criteria_x < tol_x )
          fprintf(fp,'%% *** convergence in design variables \n');
        end
        if ( convergence_criteria_f < tol_f )
          fprintf(fp,'%% *** convergence in design objective \n');
        end
        if (max(g_opt) < tol_g )
          fprintf(fp,'%% *** Woo-Hoo!  Converged solution is feasible! \n')
        else
          fprintf(fp,'%% *** Boo-Hoo!  Converged solution is NOT feasible! \n')
          fprintf(fp,'%%     ... Increase the Penalty (options(6)) and try, try again ...\n')
        end
      end
      break;
    end
    if ( function_count - last_update > 0.20*max_evals )
      for fp=1:cvg_hst_file
        if ( fp==2 ) , fp = chfp; end
        fprintf(fp,'%% ***  Hmmm ... Best solution not improved in the last %d evaluations\n', function_count - last_update)
        fprintf(fp,'%% *** Increase tolX (options(2), tolF (options(3), or MaxEvals, options(5) and try, try, try again!\n')
        if (max(g_opt) < tol_g )
          fprintf(fp,'%% *** Woo-Hoo!  Best solution is feasible! \n')
        else
          fprintf(fp,'%% *** Boo-Hoo!  Best solution is NOT feasible! \n')
          fprintf(fp,'%%     ... Increase the Penalty (options(6)) and try, try again ...\n')
        end
      end
      break;
    end

end                                         % === main loop ===

if function_count >= max_evals              % time-out -----
   for fp=1:cvg_hst_file
      if ( fp==2 ) , fp = chfp; end
      fprintf(fp,'%% *** Enough!  Maximum allowable number of function evaluations %d has been exceeded!\n', max_evals );
      fprintf(fp,'%% *** Increase tolX (options(2), tolF (options(3), or MaxEvals, options(5) and try, try, try again!\n')
   end
end                                         % ----- time-out

x_opt  = (x_opt-s0)./s1;                    % scale back to original units
x_init = (x_init-s0)./s1;                   % scale back to original units

for fp=1:cvg_hst_file                       % display conclusion data
   if ( fp==2 ) , fp = chfp; end
   fprintf(fp,'%% *** Completion  : %s (%s)\n', ...
                                     datestr(now,14), datestr(toc/3600/24,13));
   fprintf(fp,'%% *** Objective   : %12.5f \n', f_opt);
   fprintf(fp,'%% *** Variables   : \n');
   fprintf(fp,'               x_init         x_lb     <     x_opt    <     x_ub\n')
   fprintf(fp,'------------------------------------------------------------------\n');
   for ii = 1:n
     binding = ' '; 
     eqlb = ' '; 
     equb = ' '; 
     if ( x_opt(ii) < x_lb(ii) + tol_g + 10*eps )
       eqlb = '=';
       binding = '** binding **'; 
     elseif ( x_opt(ii) > x_ub(ii) - tol_g - 10*eps )
       equb = '=';
       binding = '** binding **'; 
     end
     fprintf(fp,'x(%3u)  %12.5f   %12.5f %s %12.5f %s  %12.5f %s\n',ii,x_init(ii),x_lb(ii),eqlb,x_opt(ii),equb,x_ub(ii),binding);
   end

   fprintf(fp,'%% *** Constraints : \n');
   for jj=1:m
      binding = ' '; 
      if ( g_opt(jj) > -tol_g ) 
        binding = '    ** binding **';
      end
      if ( g_opt(jj) > tol_g )
        binding = '    ** not ok  **';
      end
      fprintf(fp,'       g(%3u) = %12.5f  %s \n', jj, g_opt(jj), binding );
   end
   fprintf(fp,'\n');
end

cvg_hst = cvg_hst(:,1:iteration);
if ( cvg_hst_file == 2 )
  fclose(chfp);
end

if msglev, format; end

return;

% NMAopt ======================================================================
% updated ...
% 2005-1-22, 2006-1-26, 2011-1-31, 2011-4-13, 2016-03-24, 2016-04-06, 2019-02-23, 
% 2019-03-21, 2019-11-22, 2020-01-17, 2021-01-19, 2024-04-03

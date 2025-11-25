function [x_opt,f_opt,g_opt,cvg_hst,lambda,Hess]=SQPopt(func,x_init,x_lb,x_ub,options,c)
% [x_opt,f_opt,g_opt,cvg_hst,lambda,Hess]=SQPopt(func,x_init,x_lb,x_ub,options,c)
%
% SQPopt: Nonlinear optimization with inequality constraints using 
%          Sequential Quadratid Programming
%
%       minimizes f(x) such that g(x)<0 and x_lb <= x_opt <= x_ub.  
%       f is a scalar objective function, p is a vector of design variables,
%       and g is a vector of constraints.  
%
% INPUT
% ======
%  func    :   the name of the matlab function to be optimized in the form
%               [objective, constraints] = func(x,c)
%  x_init  :   the vector of initial parameter values                  (n x 1)
%  x_lb    :   lower bound on permissible values of the variables, x  (n x 1)
%  x_ub    :   upper bound on permissible values of the variables, x  (n x 1)
%  options :   options(1) = 1  means display intermediate results
%              options(2) = tol_p  tolerance on convergence of variables 
%              options(3) = tol_f  tolerance on convergence of objective
%              options(4) = tol_g  tolerance on constraint functions
%              options(5)  = max_evals limit on number of function evaluations
%  c       :   an optional vector of contsants used by func(x,c)
%
% OUTPUT
% ======
%  x_opt   :   a set of design variables at or near the optimal value
%  f_opt   :   the objective associated with the optimal design variables
%  g_opt   :   the constraints associated with the optimal design variables 
% cvg_hst  :   record of x, f, g, function_count, and convergence criteria
% lambda   :   the set of Lagrange multipliers at the active constraints
%  Hess    :   the Hessian of the objective function at the optimal point

%	X=SQPopt('FUN',X,x_lb,x_ub,OPTIONS,'gradfunc') allows a function 
%	'gradfunc' to be entered which returns the partial derivatives of the 
%	function and the  constraints at X:  [grad_f,grad_g] = gradfunc(X).

%	Copyright (c) 1990 by the MathWorks, Inc.
%	Andy Grace 7-9-90.

% --------------------- READ INPUTS -----------------------------

 if nargin<2 		% need at least 'func' and x_init to get started
    help SQPopt;
    return
 end

 x_init = x_init(:);              % Set up design variables as a column vector

% Handle missing arguments and do error checking

 if nargin<3                      % default values for parameter ranges
   x_lb = -1.0e2*abs(x_init);
   x_ub =  1.0e2*abs(x_init);
 else
   x_lb = x_lb(:);
   x_ub = x_ub(:);
 end

% Put the initial X within the specified parameter ranges.
 if any(x_lb>x_ub)
    error('Parameter bounds are infeasible ... x_lb > x_ub')
 else
    x_init = min(max(0.9*x_lb,x_init),0.9*x_ub);   % apply constraints
 end

 if nargin<5, options = []; end   % use default option values
 options = optim_options(options);
 if nargin<6, c = 1.0; end	% optional constants passed to func(x,c)

 msglev   = options(1);         % 1: display intermediate values
 tol_x    = options(2);         % tolerance for convergence in x
 tol_f    = options(3);         % tolerance for convergence in f
 tol_g    = options(4);         % tolerence for active constraints g
 max_evals= options(5);         % maximum number of function evaluations
 penalty  = options(6);         % penalty  factor  for unsatisfied constraints
 q        = options(7);         % penalty exponent for unsatisfied constraints
 Navg     = options(8);         % penalty exponent for unsatisfied constraints
 grad_chk = options(15);        % gradient step size
 del_min  = options(17);        % minimum param. change for finite difference
 del_max  = options(18);        % maximum param. change for finite difference
 neqcstr  = options(19);        % number of equality constraints

 options(6) = 0;
 if nargin < 7, gradfunc=[]; end

 if msglev > 0
    more off
 end
 if ( msglev > 2 )
    [f_min,f_max]=plot_surface(func,x_init,x_lb,x_ub,options,c,103);
 end

% --------------- INITIALIZE VECTORS AND MATRICES -------------------------

 nvars = length(x_init);	% the number of design variables 

 s0 = (x_lb+x_ub)./(x_lb-x_ub);  s1 = 2./(x_ub-x_lb);  % scale to [-1:1]
 x_init = s0+s1.*x_init;                               % scale x to [-1:+1]

 if any(x_ub <= x_lb)
   disp('error: x_ub can not less than or equal to x_lb for any parameter');
   x_opt = (x_init-s0)./s1;
   f_opt = (sqrt(5)-1)/2; 
   g_opt = 1;
   return
 end

 x_lb = -1.0*ones(nvars,1);  x_ub = +1*ones(nvars,1);    

 XOUT = x_init;
 x    = x_init; 

 SQP_tic_id = tic;
 f_opt = Inf;             % best value of objective function starts at infinity
 [f,g] = feval(func,(x-s0)./s1,c); % first function evaluation
%[f,g] = avg_cov_func(func, (x-s0)./s1, s0,s1, options, c, 0);

 if prod(size(f)) ~= 1
   error('the objective computed by your analysis function must be a scalar')
 end
 if size(g,1) == 1 && size(g,2) > 1
   error('the constraints computped by your analysis function must be a column vector')
 end

 g_max = max(g);

 ncstr = length(g);		% the number of constraint equations

 OLDX     = x_init;
 OLDG     = g;
 gradf    = zeros(1,nvars);
 OLDgradf = zeros(1,nvars);
 gradg    = zeros(ncstr,nvars);
 OLDgradg = zeros(ncstr,nvars);
 LAMBDA   = zeros(ncstr,1);   % initialize Lagrange multipliers for constraints
 HESS     = eye(nvars,nvars); % initialize Hessian matrix
 SD       = zeros(nvars,1);   % search direction

 cvg_hst  = NaN(nvars+5,max_evals);

 CHG  = 1e-7 * ( ones(nvars,1) + abs(x_init) );  % fractional paramter changes
 GNEW = 1e8*CHG;

%if length(GRADfunc)		% gradients supplied 
%	if ~any(gradfunc<48) % Check alphanumeric
%		gtype = 1;
%		evalstr2 = [gradfunc,'(x'];
%		for i=1:nargin - 6
%			gtype = 2;
%			evalstr2 = [evalstr2,',P',int2str(i)];
%		end
%		evalstr2 = [evalstr2, ')'];
%	else
%		gtype = 3;
%		evalstr2=[gradfunc,';'];
%	end
%end


%---------------------------- MAIN LOOP ----------------------------

 function_count = Navg;
 gradient_count = 0;
 iteration      = 1;
 StepLength     = 1;
 end_iterations = 0; 

 cvg_hst(:,iteration) = [ (x-s0)./s1 ; f ; max(g) ; function_count ; 1 ; 1 ];

while ~end_iterations

% ----------- COMPUTE GRADIENTS OF COST AND CONSTRAINTS -------------

%	if ~length(gradfunc) || grad_chk         % Finite Difference gradients
		oldf = f;
		oldg = g;
		ncstr = length(g);
		gradg = zeros(ncstr,nvars);
		% Try to make the finite differences equal to 1.0e-8.
                CHG = -1.0e-8./(GNEW+eps);
		CHG = sign(CHG+eps).*min(max(abs(CHG),del_min),del_max);
		for gcnt=1:nvars
			temp = XOUT(gcnt);
			XOUT(gcnt) = temp + CHG(gcnt);
			x(:) = XOUT; 
			[f,g] = feval(func,(x-s0)./s1,c);
                        %[f,g]=avg_cov_func(func,(x-s0)./s1,s0,s1,options,c,0);

			if ( g_max < tol_g && f < f_opt )
				if msglev > 1
                                   fprintf(1,' update optimum point \n');
                                end
				f_opt = f;
 				g_opt = g;
				x_opt = x;
 			end

% Next line is to be used for problems with varying number of constraints
%			if ncstr~=length(g), diff=length(g); g=v2sort(oldg,g); end
			gradf(1,gcnt) = (f - oldf) / CHG(gcnt);
			gradg(:,gcnt) = (g - oldg) / CHG(gcnt); 
			XOUT(gcnt) = temp;
		end
 		function_count = function_count + nvars;
 		f = oldf;
 		g = oldg;

%               if grad_chk == 1                         % Gradient check
%                       gradfFD = gradf;
%			gradgFD = gradg; 
%                       x(:)=XOUT; 
%			if gtype == 1
%				[gradf, gradg] = feval(gradfunc, x);
%			elseif gtype == 2
%				[gradf, gradg] = eval(evalstr2);
%			else
%				eval(evalstr2);
%			end
%			disp('Function derivative')
%                        graderr(gradfFD, gradf, evalstr2);
%			disp('Constraint derivative')
%                       graderr(gradgFD, gradg, evalstr2);
%                       grad_chk = 0;
%               end
%
%	else                                        % User-supplied gradients
%
%		if gtype == 1
%			[gradf, gradg] = feval(gradfunc, x);
%		elseif gtype == 2
%			[gradf, gradg] = eval(evalstr2);
%		else
%			eval(evalstr2);
%		end
%	end
%	gradient_count = gradient_count + 1;

        if iteration == 1  % this is within the first iteration
           PENALTY = (eps+gradf*gradf')*ones(ncstr,1)./(sum(gradg'.^2)'+eps);
        end

% Change constraint gradients to account for equality constraints ...
% Make the gradient of the equality constraints face in 
% the opposite direction to the function gradient.
%
%	for i=1:neqstr 	
%		schg=gradg(i,:)*gradf;
%		if schg>0
%			gradg(:,i)=-gradg(:,i);
%			g(i)=-g(i);
%		end
%	end

	GOLD = OLDgradf + LAMBDA'*OLDgradg; % old gradient of augmented cost
	GNEW = gradf    + LAMBDA'*gradg;    % new gradient of augmented cost
	q    = GNEW - GOLD;                 % change in the augmented gradients
	p    = XOUT - OLDX;                 % change in the design variables 
% ------------ Finished Gradients ---------------

%------------- UPDATE THE HESSIAN ---------------

% Ensure the Hessian is positive definite in the update.

	if q*p < StepLength^2*1e-3          % if (qp) is small or negative
		how='modify gradients to ensure Hessian > 0,';
		while q*p < -1e-5
			[qp_min,qp_idx]=min(q'.*p);  % find most negative qp
			q(qp_idx)=q(qp_idx)/2; % cut corresponding q in half
		end
		if q*p < (eps*norm(HESS,'fro')); % qp still small or negative
			FACTOR = gradg' * g  -  OLDgradg' * OLDG;
			FACTOR = FACTOR.*(p.*FACTOR>0).*(q'.*p<=eps);
			WT=1e-2;
			if max(abs(FACTOR))==0;
				FACTOR = 1e-5*sign(p); 
				how = 'small gradients,';
                        end
			while q*p < (eps*norm(HESS,'fro')) && WT < 1/eps
				q = q + WT*FACTOR';
				WT = WT*2;
			end
		end
        else
		how='regular';
 	end

%--------- perform a BFGS Hessian update only if q*p is positive --------
	if q*p > eps
		HESS = HESS + (q'*q)/(q*p) - (HESS*p*p'*HESS')/(p'*HESS*p);

  		how=[how,' Hessian update.'];

% BFGS update using Cholesky factorization  of Gill, Murray and Wright.
% In practice this was less robust than above method, and slower.
%	R=chol(HESS); 
%	s2=R*S; y=R'\q; 
%	W=eye(nvars,nvars)-(s2'*s2)\(s2*s2') + (y'*s2)\(y*y');
%	HESS=R'*W*R;

	else
  		how=[how,' no Hessian update.'];
	end
% --------- Finished Hessian Update -------------

% --------- DETERMINE THE SEARCH DIRECTION ------------------

% ----------- set up the quadratic programming problem --------
	OLDX      = XOUT;
	OLDF      = f;
	OLDG      = g;
	OLDgradf  = gradf;
	OLDgradg  = gradg;
	OLDLAMBDA = LAMBDA;
	SDi       = zeros(nvars,1);	% initial guess for search direction

        % append parameter constraints to nonlinear constraints and gradg
        GT    = [ g ; -XOUT+x_lb ;  XOUT-x_ub ];
        gradg = [ gradg ; -eye(nvars) ; eye(nvars) ];

	% solve the quadratic programming problem to get search direction, SD.
        % min 0.5* SD' * HESS * SD + gradf * SD  subject to  gradg * SD < -GT
	[SD, lambda, howqp] = mwQP(HESS,gradf',gradg,-GT,[],[],SD,neqcstr);

%	lambda(1:neqcstr) = abs(lambda(1:neqcstr)); % equality cnstrnts
%	ga=[ abs(g(1:neqcstr))' ; g(neqcstr+1:ncstr)' ]; % equality cstrts
% 	ga=g;					% no equality constraints
	g_max = max(g);

	LAMBDA=lambda(1:ncstr);	 % exclude parameter constraints from LAMBDA

        % don't change PENALTY too quickly ...
 	PENALTY = max([ LAMBDA  ; 0.5*(LAMBDA  + PENALTY ) ]);

%	PENALTY = max([ LAMBDA' ; 0.5*(LAMBDA' + PENALTY(:)') ]);
	% PENALTY is used only as a scalar penalty factors for
	% unsatisfied constraints in the merit function for the line search
% --------- Finished Search Direction -------------


% ---------------- LINE SEARCH TO REDUCE f OR max(g) ----------------

% This line search looks for improvement in either the maximum constraint
% or the objective function unless the sub-problem is infeasible in which
% case only a reduction in the maximum constraint is tolerated.
% This less "stringent" merit function has produced faster convergence in
% a large number of problems.

	infeas = (howqp(1) == 'i');

	GOAL_1 = f + sum(PENALTY*(g>tol_g).*g ) + 1e-30;  % penalized cost

	if g_max > tol_g		% constraints are being violated
		GOAL_2 = g_max;         % GOAL_2 is max constraint
	elseif f >=0 
		GOAL_2 = -1/(f+1);      % GOAL_2 is -1/(f+1) (negative)
	else 
		GOAL_2 = 0;             % GOAL_2 is zero
	end
	if ~infeas && f < 0
		GOAL_2 = GOAL_2 + f - 1;
	end

	COST_1  = GOAL_1 + 1;  % initialize penalized cost
	COST_2  = GOAL_2 + 1;  % initialize max constraint (if it is > 0)

        if msglev > 1
           fprintf(1,'   alpha      max{g}          COST_1           COST_2\n');
        end

	StepLength = 2;	       % initialize step length along search direction

	while ((COST_1>GOAL_1) && (COST_2>GOAL_2) && (function_count<max_evals))
		StepLength = StepLength / 2;		% reduce step length
		if StepLength < 1e-4  
			StepLength = -StepLength; 	% change directions
		end
		XOUT = OLDX + StepLength*SD;            % new design variables
		x(:) = XOUT; 
		 [f,g] = feval(func,(x-s0)./s1,c);	% evaluate system
                %[f,g] = avg_cov_func(func, (x-s0)./s1, s0,s1, options, c, 0);

		if ( g_max < tol_g && f < f_opt )
			if msglev  1 ,  fprintf(1,' update \n'); end
			f_opt = f;
 			g_opt = g;
			x_opt = x;
 		end

		function_count = function_count + 1; 	% function call counter

% 		ga=[abs(g(1:neqcstr));g(neqcstr+1:length(g))];  % eqlty cstrts
%		ga=g;					% no equality constrts
		g_max=max(g);
		COST_1 = f+sum(PENALTY*(g>0).*g); % penalized cost
		if g_max > tol_g	   % constraints are being violated
			COST_2 = g_max;    %  COST_2 is max constraint
		elseif f >=0 
			COST_2 = -1/(f+1); %  COST_2 is -1/(f+1) (negative)
		else 
			COST_2 = 0;        %  COST_2 is zero
		end
		if ~infeas && f < 0
			COST_2 = COST_2 + f - 1;
		end
                if msglev > 1
                  fprintf(1,'  %9.2e   %9.2e  %9.2e %6.3f %9.2e %6.3f\n', ...
                    StepLength,g_max,COST_1,COST_1/GOAL_1,COST_2,COST_2/GOAL_2);
                end
	end
        if msglev == 3	% plot surface and convergence point
		%% pause(1.0);
	end
%------------ Finished Line Search -------------

	absSL  = abs(StepLength);               % absolute value of step length
	LAMBDA = absSL*LAMBDA + (1-absSL)*OLDLAMBDA; % new Lagrange multipliers
        g_ok = find(g < -tol_g);
	LAMBDA(g_ok) = 0;
	lambda(g_ok) = 0;

	iteration = iteration + 1;

% ----------- DISPLAY RESULTS -------------
        secs_left = round((max_evals-function_count)*toc(SQP_tic_id)/function_count);
        eta       = datestr(now+secs_left/3600/24,14);
        if msglev > 0
         home
         fprintf(1,' ************************* SQPopt ********************\n');
         fprintf(1,' iteration            : %5d', iteration );
         if max(g) > tol_g || any(x<-1) || any(x>1)
            fprintf(1,'       !!! infeasible !!!\n');
         else
            fprintf(1,'       ***  feasible  ***\n');
         end
         fprintf(1,' function evaluations : %5d \n', function_count );
         fprintf(1,' e.t.a.               : %s\n', eta );
         fprintf(1,' objective = %11.3e\n', f );
         fprintf(1,' variables = ');
         fprintf(1,'%11.3e', (x-s0)./s1 );
         fprintf(1,'\n');
         [g_max,idxmaxg] = max(g);
         fprintf(1,' max constraint = %11.3e  (%d)\n', g_max, idxmaxg );
         fprintf(1,' Step Size      = %11.3e\n', StepLength );
         fprintf(1,' BFGS method    : %12s\n', how );
         fprintf(1,' QP   method    : %12s\n', howqp );
         fprintf(1,' F convergence  : %11.3e     tolF : %8.6f\n', abs(absSL*gradf*SD/f), tol_f );
         fprintf(1,' X convergence  : %11.3e     tolX : %8.6f\n', max(abs(absSL*SD./x)), tol_x );
         fprintf(1,' ************************* SQPopt ********************\n');
        end

        cvg_hst(:,iteration) = [ (x-s0)./s1 ; f ; max(g) ; function_count ; 
                              max(abs(absSL*SD./x)) ;  abs(absSL*gradf*SD/f) ];

        if ( msglev > 2 )    % plot surface and convergence point
	   x = (x-s0)./s1;
           figure(103)
            plot3(x(options(11)),x(options(12)),f+(f_max-f_min)/100,'ro','MarkerSize',8, 'LineWidth', 4)
            drawnow
           x = s0+s1.*x;
        end

%	if ( g_max < tol_g && f < f_opt )	% update the optimal solution
%		if msglev > 1 , fprintf(1,' update \n'); end;
%		f_opt = f;
%		g_opt = g;
%		x_opt = x;
%	end

% ----------- CHECK CONVERGENCE CRITERIA ------------

	if ( ( max(abs(absSL*SD./x)) < tol_x   || ...   % design var converged
               abs(absSL*gradf*SD/f) < tol_f ) && ...   % design obj converged
              ((g_max < tol_g) || ((howqp(1) == 'i') && (g_max > 0) ) ) )
%   	     (iteration > nvars) ) %  && ...
                                                              % constraints ok

	    end_iterations = 1;

	    if howqp(1) ~= 'i'
 	     fprintf(1,' *** Woo-Hoo!  Converged solution found in %5d iterations! \n',iteration)
             if ( max(abs(absSL*SD./x)) < tol_x )
               fprintf(1,' *** convergence in design variables\n');
             end
             if ( abs(absSL*gradf*SD/f) < tol_f )
               fprintf(1,' *** convergence in design objective \n');
             end
             if (g_max < tol_g )
              fprintf(1,' *** Woo-Hoo!  Converged solution is feasible! \n')
             else
              fprintf(1,' *** Boo-Hoo!  Converged solution is NOT feasible!\n')
             end
	    end
	    if (howqp(1) == 'i' && g_max > tol_g)
	       disp(' *** Boo-Hoo-Hoo!  No feasible solution found.')
	    end

	elseif function_count >= max_evals % NOT converged but give up

	    x_opt = OLDX;  % iterations exeeded during line search, reset XOUT
	    f_opt = OLDF;  % iterations exeeded during line search, reset f 
	    g_opt = OLDG;  % iterations exeeded during line search, reset g 
            disp([' *** Drats!  Maximum number of function evaluations (', ...
                   int2str(max_evals),') has been exceeded']);
            disp(' *** Increase tolX (options(2), tolF (options(3), or MaxEvals, options(5) and')
            disp(' *** try, try, try again!')
	    end_iterations = 1;

	end  
%------------ Finished Convergence Criteria -------------


end %------------------------- Finished Main Loop ----------------------------

% if a better feasible solution was found, then use it!
if ( (f_opt < f) && (max(g_opt) < tol_g) )
 	x = x_opt;
 	f = f_opt;
 	g = g_opt;
end

x_opt  = (x-s0)./s1;
x_init = (x_init-s0)./s1;
x_lb   = (x_lb-s0)./s1;
x_ub   = (x_ub-s0)./s1;
f_opt = f;
g_opt = g;

fprintf(1,' *** Objective : %12.5f \n', f_opt );
fprintf(1,' *** Variables : \n')
fprintf(1,'               x_init         x_lb     <     x_opt     <    x_ub         lambda\n')
fprintf(1,'--------------------------------------------------------------------------------\n');
for ii = 1:nvars
  eqlb = ' ';
  equb = ' ';
  lulb = ' ';
  if ( x_opt(ii) < x_lb(ii) + tol_g + 10*eps )
    eqlb = '=';
    lulb = sprintf('%12.5f', lambda(ncstr+ii));
  elseif ( x_opt(ii) > x_ub(ii) - tol_g - 10*eps )
    equb = '=';
    lulb = sprintf('%12.5f', lambda(ncstr+nvars+ii));
  end
  fprintf(1,'x(%3u)  %12.5f   %12.5f %s %12.5f %s  %12.5f %s\n',ii,x_init(ii),x_lb(ii),eqlb,x_opt(ii),equb,x_ub(ii),lulb);
end
fprintf(' *** Constraints : \n');
for ii=1:ncstr
  if ( lambda(ii) == 0.0 ) 
     binding = ' ';
  end
  if ( lambda(ii) >  0.0 ) 
     binding = '    ** binding **';
  end
  if ( g_opt(ii) >= tol_g ) 
     binding = '    ** not ok  **';
  end
  fprintf(1,'       g(%3u) = %12.5f      lambda(%3u) = %12.5f   %s\n', ii, g_opt(ii), ii, lambda(ii), binding );
end
actCstr = find(LAMBDA>0);
fprintf(' *** Active Constraints : ');  
for j = 1:length(actCstr)
  fprintf(1,'  %2u', actCstr(j) );
end
fprintf('\n\n')

cvg_hst = cvg_hst(:,1:iteration);

% endfunction # --------------------------------------------------- SQPopt


function [X,lambda,how]=mwQP(H,f,A,B,x_lb,x_ub,X,neqcstr,verbosity,negdef,normalize)
% [X,lambda,how]=mwQP(H,f,A,B,x_lb,x_ub,X0,neqcstr,verbosity,negdef,normalize)
%
%       [x,lambda] = mwQP(H,f,A,b,x_lb,x_ub,X0,neqcstr,verbosity))
%
%        solves the quadratic programming problem:
%
%            min 0.5*x'Hx + f'x   subject to:  Ax <= b 
%             x    
%
%       lambda is the set of Lagrangian multipliers at the optimal point
%
%       x_lb and x_ub define a set of lower and upper bounds on the
%       design variables, X, so that the solution  
%       is always in the range x_lb < X < x_ub.
%
%       X0 is the initial starting point
%
%       neqcstr indicates that the first neqcstr constraints are equality 
%       constraints defined by A and b are equality constraints.
%
%       verbosity indicates the level of warning messages displayed during
%       the solution.  A value of -1 results in no warning messages. 
%
%  Copyright (c) 1990-94 by The MathWorks, Inc.
%  Andy Grace 7-9-90.


% Handle missing arguments
if nargin < 11, normalize = 1;
  if nargin < 10, negdef = 0; 
    if nargin< 9, verbosity = []; 
      if nargin< 8, neqcstr=[]; 
        if nargin < 7, X=[]; 
          if nargin<6, x_ub=[]; 
            if nargin<5, x_lb=[];
end, end, end, end, end, end, end
[ncstr,nvars]=size(A);
nvars = length(f); % In case A is empty
if ~length(verbosity), verbosity = 0; end
if ~length(neqcstr), neqcstr = 0; end
if ~length(X), X=zeros(nvars,1); end

f=f(:);
B=B(:);

simplex_iter = 0;
if ( norm(H,'inf')==0 || prod(size(H))==0 )
    H=0;
    is_qp=0;
else
    is_qp=~negdef;
end

how = 'ok'; 

normf = 1;
if normalize > 0
  if ~is_qp
    normf = norm(f);
    f = f./normf;
  end
end


% Handle the parameter bounds as linear constraints
  lenXmin=length(x_lb);
  if lenXmin > 0     
    A=[A;-eye(lenXmin,nvars)];
    B=[B;-x_lb(:)];
  end
  lenXmax=length(x_ub);
  if lenXmax>0
    A=[A;eye(lenXmax,nvars)];
    B=[B;x_ub(:)];
  end 
  ncstr=ncstr+lenXmin+lenXmax;

  errcstr = 100*sqrt(eps)*norm(A); 
  % Used for determining threshold for whether a direction will violate
  % a constraint.
  normA = ones(ncstr,1);
  if normalize > 0 
    for i=1:ncstr
      n = norm(A(i,:));
      if (n ~= 0)
        A(i,:) = A(i,:)/n;
        B(i) = B(i)/n;
      normA(i,1) = n;
    end
  end
else 
  normA = ones(ncstr,1);
end
errnorm = 0.01*sqrt(eps); 

lambda=zeros(ncstr,1);
aix=lambda;
ACTCNT=0;    % number of active constraints
ACTSET=[];    % the set of active constraints
ACTIND=0;    % the indices of the active constraints
CIND=1;      % a constraint index
eqix = 1:neqcstr; 
%------------EQUALITY CONSTRAINTS---------------------------
Q = zeros(nvars,nvars);
R = [];
if neqcstr>0
  aix(eqix)=ones(neqcstr,1);
  ACTSET=A(eqix,:);
  ACTIND=eqix;
  ACTCNT=neqcstr;
  if ACTCNT >= nvars - 1, simplex_iter = 1; end
  CIND=neqcstr+1;
  [Q,R] = qr(ACTSET');
  if max(abs(A(eqix,:)*X-B(eqix)))>1e-10 
    X = ACTSET\B(eqix);
    % X2 = Q*(R'\B(eqix)); does not work here !
  end
  %  Z=null(ACTSET);
  [m,n]=size(ACTSET);
  Z = Q(:,m+1:n);
  err = 0; 
  if neqcstr > nvars 
    err = max(abs(A(eqix,:)*X-B(eqix)));
    if (err > 1e-8) 
      how='infeasible quadratic program'; 
      if verbosity > -1
        disp('Warning: The equality constraints are overly stringent;')
        disp('         there is no feasible solution.') 
      end
    end
    actlambda = -R\(Q'*(H*X+f)); 
    lambda(eqix) = normf * (actlambda ./normA(eqix));
    return
  end
  if ~length(Z) 
    actlambda = -R\(Q'*(H*X+f)); 
    lambda(eqix) = normf * (actlambda./normA(eqix));
    if (max(A*X-B) > 1e-8)
      how = 'infeasible quadratic program';
      disp('Warning: The constraints or bounds are overly stringent;')
      disp('         there is no feasible solution.') 
      disp('         Equality constraints have been met.')
    end
    return
  end
% Check whether in Phase 1 of feasibility point finding. 
  if (verbosity == -2)
    cstr = A*X-B; 
    mc=max(cstr(neqcstr+1:ncstr));
    if (mc > 0)
      X(nvars) = mc + 1;
    end
  end
else
  Z=1;
end

% === Find Initial Feasible Solution ====
cstr = A*X-B;
mc=max(cstr(neqcstr+1:ncstr));
if mc>eps
  A2=[[A;zeros(1,nvars)],[zeros(neqcstr,1);-ones(ncstr+1-neqcstr,1)]];

  [XS,lambdas]=mwQP([],[zeros(nvars,1);1],A2,[B;1e-5],[],[],[X;mc+1],neqcstr,-2,0,-1);

  X=XS(1:nvars);
  cstr=A*X-B;
  if XS(nvars+1)>eps 
    if XS(nvars+1)>1e-8 
      how='infeasible quadratic program';
      if verbosity > -1
        disp('Warning: The constraints are overly stringent;')
        disp('         there is no feasible solution.')
      end
    else
      how = 'overly constrained quadratic program';
    end
    lambda = normf * (lambdas(1:ncstr)./normA);
    return
  end
end

if (is_qp && prod(size(H)) )
  gf=H*X+f;
  SD=-Z*((Z'*H*Z)\(Z'*gf));
% Check for -ve definite problems:
%  if SD'*gf>0, is_qp = 0; SD=-SD; end
else
  gf = f;
  SD=-Z*Z'*gf;
  if ( (norm(SD) < 1e-10) && neqcstr )
    % This happens when equality constraint is perpendicular
    % to objective function f(x).
    actlambda = -R\(Q'*(H*X+f)); 
    lambda(eqix) = normf * (actlambda ./ normA(eqix));
    return;
  end
end
% Sometimes the search direction goes to zero in negative
% definite problems when the initial feasible point rests on
% the top of the quadratic function. In this case we can move in
% any direction to get an improvement in the function so try 
% a random direction.
if negdef
  if norm(SD) < sqrt(eps)
    SD = -Z*Z'*(rand(nvars,1) - 0.5);
  end
end
oldind = 0; 


t=zeros(10,2);
tt = zeros(10,1);

% The maximum number of iterations for a simplex type method is:
% maxiters = prod(1:ncstr)/(prod(1:nvars)*prod(1:max(1,ncstr-nvars)));

% --------- START MAIN QUADRATIC PROGRAMMING ROUTINE ----------
while 1
% Find distance we can move in search direction SD before a 
% constraint is violated.
  % Gradient with respect to search direction.
  GSD=A*SD;

  % Note: we consider only constraints whose gradients are greater
  % than some threshold. If we considered all gradients greater than 
  % zero then it might be possible to add a constraint which would lead to
  % a singular (rank deficient) working set. The gradient (GSD) of such
  % a constraint in the direction of search would be very close to zero.
  indf = find((GSD > errnorm * norm(SD))  &  ~aix);

  if ~length(indf)
    STEPMIN=1e16;
  else
    dist = abs(cstr(indf)./GSD(indf));
    [STEPMIN,ind2] =  min(dist);
    ind2 = find(dist == STEPMIN);
% Bland's rule for anti-cycling: if there is more than one blocking constraint
% then add the one with the smallest index.
    ind=indf(min(ind2));
% Non-cycling rule:
    % ind = indf(ind2(1));
  end
%----------------- QP ... QUADRATIC PROGRAMMING ------------
  if ( is_qp && (prod(size(H)) > 0) ) 
% If STEPMIN is 1 then this is the exact distance to the solution.
    if STEPMIN>=1
      X=X+SD;
      if ACTCNT>0  % if number of active constraints > 0

        if ACTCNT>=nvars-1
                                   if size(ACTSET,1) >= CIND
                                       ACTSET(CIND,:)=[];
                                   end
                                   if length(ACTIND) >= CIND
                                      ACTIND(CIND)=[];
                                   end
                                end
        
        rlambda = -R\(Q'*(H*X+f));
        actlambda = rlambda;
        actlambda(eqix) = abs(rlambda(eqix));
        indlam = find(actlambda < 0);
        if (~length(indlam)) 
          lambda(ACTIND) = normf * (rlambda./normA(ACTIND));
          return
        end
% Remove constraint
        lind = find(ACTIND == min(ACTIND(indlam)));
        lind=lind(1);
        ACTSET(lind,:) = [];
        aix(ACTIND(lind)) = 0;
        [Q,R]=qrdelete(Q,R,lind);
        ACTIND(lind) = [];
        ACTCNT = ACTCNT - 2;
        simplex_iter = 0;
        ind = 0;
      else
        return
      end
    else
      X=X+STEPMIN*SD;
    end
    % Calculate gradient w.r.t objective at this point
    gf=H*X+f;
  else 
    % Unbounded Solution
    if ~length(indf) || ~isfinite(STEPMIN)
      if norm(SD) > errnorm
        if normalize < 0
          STEPMIN=abs((X(nvars)+1e-5)/(SD(nvars)+eps));
        else 
          STEPMIN = 1e16;
        end
        X=X+STEPMIN*SD;
        how='unbounded quadratic program'; 
      else
        how = 'ill posed quadratic program';
      end
      if verbosity > -1
        if norm(SD) > errnorm
          disp('Warning: The solution is unbounded and at infinity;')
          disp('         the constraints are not restrictive enough.') 
        else
          disp('Warning: The search direction is close to zero; the problem is ill posed.')
          disp('         The gradient of the objective function may be zero')
          disp('         or the problem may be badly conditioned.')
        end
      end
      return
    else 
      X=X+STEPMIN*SD;
    end
  end %if (qp)

% Update X and calculate constraints
  cstr = A*X-B;
  cstr(eqix) = abs(cstr(eqix));
% Check no constraint is violated
  if normalize < 0 
    if X(nvars,1) < eps
      return;
    end
  end
      
  if max(cstr) > 1e5 * errnorm
    if max(cstr) > norm(X) * errnorm 
      if verbosity > -1
        disp('Warning: The problem is badly conditioned;')
        disp('         the solution is not reliable') 
        verbosity = -1;
      end
      how='unreliable quadratic program'; 
      X=X-STEPMIN*SD;
      return
    end
  end


% Sometimes the search direction goes to zero in negative
% definite problems when the current point rests on
% the top of the quadratic function. In this case we can move in
% any direction to get an improvement in the function so 
% foil search direction by giving a random gradient.
  if negdef
    if norm(gf) < sqrt(eps)
      gf = randn(nvars,1);
    end
  end
  if ind
    aix(ind)=1;
    ACTSET(CIND,:)=A(ind,:);
    ACTIND(CIND)=ind;
    [m,n]=size(ACTSET);
    [Q,R] = qr_insert(Q,R,CIND,A(ind,:)');
  end
  if oldind 
    aix(oldind) = 0; 
  end
  if ~simplex_iter
    % Z = null(ACTSET);
    [m,n]=size(ACTSET);
    Z = Q(:,m+1:n);
    ACTCNT=ACTCNT+1;
    if ACTCNT == nvars - 1, simplex_iter = 1; end
    CIND=ACTCNT+1;
    oldind = 0; 
  else
    rlambda = -R\(Q'*gf);
    if rlambda(1) == -Inf
      fprintf('         Working set is singular; results may still be reliable.\n');
      [m,n] = size(ACTSET);
      rlambda = -(ACTSET + sqrt(eps)*randn(m,n))'\gf;
    end
    actlambda = rlambda;
    actlambda(eqix)=abs(actlambda(eqix));
    indlam = find(actlambda<0);
    if length(indlam)
      if STEPMIN > errnorm
% If there is no chance of cycling then pick the constraint which causes
% the biggest reduction in the cost function. i.e the constraint with
% the most negative Lagrangian multiplier. Since the constraints
% are normalized this may result in less iterations.
        [minl,CIND] = min(actlambda);
      else
% Bland's rule for anti-cycling: if there is more than one 
% negative Lagrangian multiplier then delete the constraint
% with the smallest index in the active set.
        CIND = find(ACTIND == min(ACTIND(indlam)));
      end

      [Q,R]=qrdelete(Q,R,CIND);
      Z = Q(:,nvars);
      oldind = ACTIND(CIND);
    else
      lambda(ACTIND)= normf * (rlambda./normA(ACTIND));
      return
    end
  end %if ACTCNT<nvars
  if ( is_qp && (prod(size(H)) > 0) )
    Zgf = Z'*gf; 
    if (norm(Zgf) < 1e-15)
      SD = zeros(nvars,1); 
    elseif ~length(Zgf) 
      % Only happens in -ve semi-definite problems
      disp('Warning: QP problem is -ve semi-definite.')
      SD = zeros(nvars,1);
    else
      SD = -Z * ( (Z'*H*Z)\(Zgf) );
    end
    % Check for -ve definite problems
    % if SD'*gf>0, is_qp = 0; SD=-SD; end
  else
    if ~simplex_iter
      SD = -Z*Z'*gf;
      gradsd = norm(SD);
    else
      gradsd = Z'*gf;
      if  gradsd > 0
        SD = -Z;
      else
        SD = Z;
      end
    end
    if abs(gradsd) < 1e-10  % Search direction null
      % Check whether any constraints can be deleted from active set.
      % rlambda = -ACTSET'\gf;
      if ~oldind
        rlambda = -R\(Q'*gf);
      end
      actlambda = rlambda;
      actlambda(1:neqcstr) = abs(actlambda(1:neqcstr));
      indlam = find(actlambda < errnorm);
      lambda(ACTIND) = normf * (rlambda./normA(ACTIND));
      if ~length(indlam)
        return
      end
      cindmax = length(indlam);
      cindcnt = 0;
      newactcnt = 0;
      while ( (abs(gradsd) < 1e-10) && (cindcnt < cindmax) )
        
        cindcnt = cindcnt + 1;
        if oldind
          % Put back constraint which we deleted
          [Q,R] = qr_insert(Q,R,CIND,A(oldind,:)');
        else
          simplex_iter = 0;
          if ~newactcnt
            newactcnt = ACTCNT - 1;
          end
        end
        CIND = indlam(cindcnt);
        oldind = ACTIND(CIND);

        [Q,R]=qrdelete(Q,R,CIND);
        [m,n]=size(ACTSET);
        Z = Q(:,m:n);

        if m ~= nvars
          SD = -Z*Z'*gf;
          gradsd = norm(SD);
        else
          gradsd = Z'*gf;
          if  gradsd > 0
            SD = -Z;
          else
            SD = Z;
          end
        end
      end
      if abs(gradsd) < 1e-10  % Search direction still null
        return;
      end
      lambda = zeros(ncstr,1);
      if newactcnt 
        ACTCNT = newactcnt;
      end
    end
  end

  if ( simplex_iter && oldind )
    ACTIND(CIND)=[];
    ACTSET(CIND,:)=[];
    CIND = nvars;
  end 
end % while 1

% endfunction # ------------------------------------------------------- mwQP
%


function [Q,R] = qr_insert(Q,R,j,x)
% [Q,R] = qr_insert(Q,R,j,x)
% Insert a column in the QR factorization.
% If [Q,R] = qr(A) is the original QR factorization of A, then 
% [Q,R] = qr_insert(Q,R,j,x) changes Q and R to be the factorization
% of the matrix obtained by inserting an extra column, x, before A(:,j).  
% If A has n columns and j=n+1, then x is inserted after the last column of A.
%
% See also QR, QRDELETE, PLANEROT.

% C.B.Moler 5/1/92.  Revised ACWG 6/15/92, CBM 9/11/92.
% Copyright (c) 1984-94 by The MathWorks, Inc.

[m,n] = size(R);
if n == 0
  [Q,R] = qr(x);
  return;
end
% Make room and insert x before j-th column.
R(:,j+1:n+1) = R(:,j:n);
R(:,j) = Q'*x;
n = n+1;

% Now R has nonzeros below the diagonal in the j-th column,
% and "extra" zeros on the diagonal in later columns.
%    R = [x x x x x
%         0 x x x x
%         0 0 + x x
%         0 0 + 0 x
%         0 0 + 0 0]
% Use Givens rotations to zero the +'s, one at a time, from bottom to top.

for k = m-1:-1:j
   p = k:k+1;
   [G,R(p,j)] = planerot(R(p,j));
   if k < n
      R(p,k+1:n) = G*R(p,k+1:n);
   end
   Q(:,p) = Q(:,p)*G';
end

% endfunction # --------------------------------------------------- qr_insert

% updated ...
% 2010 - 2023  2024-02-03   2025-01-26

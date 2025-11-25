function [F_risk, g_avg, x, cov_F, N ] = avg_cov_func(func,x,s0,s1,options,consts,BOX)
% [F_risk, g_avg, x, n ] = avg_cov_func(func,x,s0,s1,options,consts,BOX) 
%
% avg_cov_func - compute the 84th percentile of the sampling distribution
% of the mean of a penalized cost function described by function func()
%
% INPUT
% ======
%  func    :   the name of the matlab function to be optimized in the form
%               [ objective, constraints ] = func( x, consts )
%  x       :   the vector of scaled design variables [-1 < x < +1]     ( n x 1 )
%  s0,s1   :   linear scaling from [v_lb,v_ub] to [-1,+1]
%  options :   options(4)  = tol_g  tolerance on convergence of constraints
%              options(6)  = penalty  on constraint violations
%              options(7)  = exponent on constraint violations
%              options(8)  = max number of function evaluations for stats
%              options(9)  = desired coefficient of variation for mean estimate
%  consts  :   an optional vector of values that are not design variables
%   BOX    :   1: bound x to within [-1:1];  0: don't
%
% OUTPUT
% ======
%  F_risk  :   average value of the penalized objective function
%  g_avg   :   average value of the constraints 
%  x       :   possibly bounded design variables, depending on "BOX"
%  cov_F   :   coefficient of variation of the augmented objective function
%  N       :   number of function evaluations used to compute the average
% 
% (1) If the input variable Bounded_X is set to 1: 
%   (a) set any value of x that exceeds the specified [x_lb, x_ub]
%       limits equal to the limits
%   (b) evaluate the specified cost function and associated constraints using 
%       the function  [f,g] = func(x).
%   (c) add a penalty fucntion to the cost ... 
%                 f = f + penalty * sum(g.*(g>tol_g))^q
% (2) If the input variable Bounded_X is set to 0 (zero):
%   (a) evaluate the specified cost function and associated constraints using 
%       the function  [f,g] = func(x).
%   (b) augment the constraint vector with parameter limit violations
%   (c) add a penalty fucntion to the cost ... 
%                 f = f + penalty * sum(ga.*(ga>tol_g))^q;
%
% Donald E. Knuth (1998). 
% The Art of Computer Programming, volume 2: 
% Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.
% 
% K.Deb, ``An efficient constraint handling method for genetic algorithms,''
% Comput. Methods Appl. Mech. Engrg, 186(2000) 311-338.
%
% H.P. Gavin, Dept. Civil & Environ. Eng'g,  Duke Univ. 

    tol_g   = options(4);
    penalty = options(6);
    q       = options(7);
    maxN    = options(8); % maximum number of evaluations for risk measure calc
    err_F   = options(9); % desired estimation error for the the mean of F

    N       = 1;          % number of objective function evaluations 
    Za2     = 1.645;     % standard normal variate for 90% confidence interval 

    avg_F = 0;  ssq_F = 0;  g_avg = 0;   cov_F = 0;   max_F = 0;
    if BOX                   % ... keep x within limits [v_lb, v_ub] ...
      x = min(max(x,-1.0),+1.0);
    end
    for N = 1:maxN

      [f,g] = feval(func,(x-s0)./s1,consts);
      F_aug = f + penalty*sum(g.*(g>tol_g))^q;

%     if ~BOX            % ... include v_lb, v_ub as constraints ...
%       gx = [ x - 1 ;  1 - x ];
%       F_aug = F_aug + penalty*sum(gx.*(gx>tol_g))^q;
%     end

      dF    = F_aug - avg_F;
      avg_F = avg_F + dF/N;                   % update average cost
      ssq_F = ssq_F + dF * (F_aug-avg_F);     % update sum-of-squares
      g_avg = g_avg + ( g - g_avg ) / N;
      max_F = max( max_F , F_aug ); 
      if N > 1
        cov_F = sqrt( ssq_F/(N-1) ) / avg_F;
        if N > 2 && N > (Za2 * cov_F / err_F )^2   % mean estimate : good enough
          break
        end
      end
    end
%{
    fprintf('n=%3d   cov_F=%5.3f   err_F=%5.3f\n', N, cov_F, err_F );
    pause(0.01)
%}
    F_risk = avg_F;    % average value
    if N > 1
%     CHOOSE ONE OF THE FOLLOWING RISK-BASED PERFORMANCE MEASURES ...
%     F_risk = max_F;                         % worst-of-N values
%     F_risk = avg_F;                         % average-of-N values
%     F_risk = avg_F * ( 1 + cov_F );         % 84th percentile of F
      F_risk = avg_F * ( 1 + cov_F/sqrt(N) ); % 84th percentile of the average of F
    end

% avg_cov_func ================================================================
% updated 2015-03-10, 2015-04-03, 2016-03-24, 2025-11-24

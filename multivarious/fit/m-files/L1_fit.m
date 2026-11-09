function [a, mu, nu, cvg_hst] = L1_fit ( X, y, alpha )
% [a, mu, nu, cvg_hst] = L1_fit(X,y,alpha)
%
% fit model parameters, a, in the model ^y = X*a  to data, y, with
% L1 regularizaion of the parameters.
%   J  = || y - X*a ||_2^2    +   alpha * ||a||_1
%  <=>
%   J = || y - X(p-q) ||_2^2  +   alpha * sum(p+q)  such that  p>=0, q>=0
%   (a = p-q; |a| = p+q)
%
% INPUT:  X    ...  the design matrix (basis of the model),      m x n 
%         y    ...  the vector of data to be fit to the model,   m x 1
%        alpha ...  l_1 regularization factor for Prony series coefficients
%
% OUTPUT: a       ...  the model coefficients
%         mu,nu    ... Lagrange multipliers for p and q
%         cvg_hst ...  convergence history
%

%{
   The main idea behind casting L1 as a QP is that the parameter 
vector {a} is replaced by the difference of two vectors {a}={p}-{q}   
that are constrained to be non-negative: p_i >=0 for all i and q_i >=0
for all i.  
  If a_i > 0, then p_i=a_i and q_i=0;  
  If a_i < 0, then p_i= 0  and q_i = -a_i.  
 With this constrained re-parameterization, |a_i| = p_i + q_i.   
 Note that the dimension of the parameter space doubles, but the KTT equations
 for the QP are simple and have analytical Hessians and gradients.
%}


 [m,n] = size(X);

 t = [1:m];

 XtX = 2*X'*X;  Xty = 2*X'*y;

% _good_ initial guess for p and q from non-regularized linear least squares 
 a  = XtX \ Xty;
 p  = zeros(n,1);	p (find(a >  2*eps)) =  a(find(a >  2*eps));
 q  = zeros(n,1);	q (find(a < -2*eps)) = -a(find(a < -2*eps));

 MaxIter = 19;

 cvg_hst = zeros ( 5*n, MaxIter );		% convergence history

 for iter = 1:MaxIter

   Au = find(p <= 2*eps);	lp = length(Au);   % active set for update u
   Av = find(q <= 2*eps);	lq = length(Av);   % active set for update v

   Ip = zeros(lp,n); for i=1:lp, Ip(i,Au(i)) = 1; end   % constraint gradient u
   Iq = zeros(lq,n); for i=1:lq, Iq(i,Av(i)) = 1; end   % constraint gradient v

   mu = zeros(n,1); nu = zeros(n,1);
   % KKT equations
   XTX = [  XtX       -XtX     Ip'        zeros(n,lq) ;		% dL/du   = 0
           -XtX        XtX   zeros(n,lp)  Iq'         ;		% dL/dv   = 0
             Ip        zeros(lp,n+lp+lq)              ;		% dL/d mu = 0
          zeros(lq,n)   Iq   zeros(lq,lp+lq)          ];	% dL/d nu = 0

   % right-hand-side vector
   XTY = [ Xty - XtX*p + XtX*q - alpha ;			% dL/du   = 0
          -Xty + XtX*p - XtX*q - alpha ;			% dL/dv   = 0
           -p(Au)                      ;			% dL/d mu = 0
           -q(Av)                      ];			% dL/d nu = 0

   u_v_mu_nu = XTX \ XTY;		% solve the system

%
   u      = u_v_mu_nu(1:n);			% update for p
   v      = u_v_mu_nu(n+1:2*n);			% update for q
   mu(Au) = u_v_mu_nu(2*n+1:2*n+lp);		% what to use mu for?
   nu(Av) = u_v_mu_nu(2*n+lp+1:2*n+lp+lq);	% what to use nu for?

   % ratio test: one common step length for u and v
   d  = 1;
   iu = find( u < -1e-14 & p > 2*eps );    % inactive p components moving toward zero
   iv = find( v < -1e-14 & q > 2*eps );    % inactive q components moving toward zero
   if ~isempty(iu), d = min( d, min( -p(iu)./u(iu) ) ); end
   if ~isempty(iv), d = min( d, min( -q(iv)./v(iv) ) ); end
   p = p + d*u;   q = q + d*v;
   p( p < 2*eps ) = 0;   q( q < 2*eps ) = 0;   % remove round-off at the bounds
   a = p - q;

   cvg_hst(:,iter) = [ a ; p ; q ; mu ; nu ];	% convergence history

   % convergence check
   if ( norm(u) <= norm(p)/1e3  &&  min(p) > -5*eps  && ...
        norm(v) <= norm(q)/1e3  &&  min(q) > -5*eps )
	break;
   end

%  figure(101); clf; plot(t,y,'o', t,X*a); drawnow; 

 end

 cvg_hst = cvg_hst(:,1:iter);

% figure(102); clf; plot([1:iter], cvg_hst(1:n,:)' ,'-o'); drawnow; %pause(1);

% ------------------------------------------------------------------ L1_fit
% H.P. Gavin, 2013-10-04

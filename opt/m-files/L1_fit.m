function [c, mu, nu, cvg_hst] = L1_fit ( B, y, alfa, w )
% [c, mu, nu, cvg_hst] = L1_fit( B, y, alfa, w )
%
% fit model coefficients, c, in the model ^y = B*c  to data, y, with
% L1 regularizaion of the coefficients.
%   J  = || y - B*c ||_2^2    +   alfa * ||c||_1
%  <=>
%   J = || y - B(p-q) ||_2^2  +   alfa * sum(p+q)  such that  p>=0, q>=0
%   (c = p-q; |c| = p+q)
%
% INPUT       DESCRIPTION                                          DIMENSION
% --------    ---------------------------------------------------  ---------
%    B        basis of the model (the design matrix                 m x n 
%    y        vector of data to be fit to the model                 m x 1
%  alfa      L_1 regularization factor for sum(abs(c))             1 x 1
%    w        0: without weighting  ...  >0: with weighting         1 x 1
%
% OUTPUT      DESCRIPTION                                          DIMENSION
% --------    ---------------------------------------------------  ---------
%    c        the model coefficients                                 n x 1 
%  mu,nu      Lagrange multipliers for p and q                       n x 1
%  cvg_hst    convergence history for c, p, q, mu, nu, alfa, err 5n+2 x # iter
%

%{
   The main idea behind casting L1 as a QP is that the coefficient 
vector {c} is replaced by the difference of two vectors {c}={p}-{q}   
that are constrained to be non-negative: p_i >=0 for all i and q_i >=0
for all i.  
  If c_i > 0, then p_i=c_i and q_i=0;  
  If c_i < 0, then p_i= 0  and q_i = -c_i.  
 With this constrained re-parameterization, |c_i| = p_i + q_i.   
 Note that the dimension of the parameter space doubles, but the KTT equations
 for the QP are simple and have analytical Hessians and gradients.
%}

 [m,n] = size(B);

 w = abs(w(1));         % ensure weight is a positive scalar

 t = [1:m];

 BtB = 2*B'*B;  Bty = 2*B'*y;

% _good_ initial guess for p and q from non-regularized linear least squares 
 c  = BtB \ Bty;         % ordinary least squares 
 p  = zeros(n,1);        p (find(c >  2*eps)) =  c(find(c >  2*eps));
 q  = zeros(n,1);        q (find(c < -2*eps)) = -c(find(c < -2*eps));

 err_norm = norm(B*c-y)/(m-n);
 err_norm_old = 100;

 max_iter = 500;
 cvg_hst = zeros ( 5*n+2, max_iter ); % convergence history
 cvg_hst(:,1) = [ c ; p ; q ; zeros(2*n,1) ; alfa ; err_norm ];  % convergence history

 for iter = 2:max_iter

   Au = find(p <= 2*eps);        lp = length(Au);   % active set for update u
   Av = find(q <= 2*eps);        lq = length(Av);   % active set for update v

   Ip = zeros(lp,n); for i=1:lp, Ip(i,Au(i)) = 1; end   % constraint gradient u
   Iq = zeros(lq,n); for i=1:lq, Iq(i,Av(i)) = 1; end   % constraint gradient v

   mu = zeros(n,1); nu = zeros(n,1);
   % KKT equations
   BTB = [  BtB       -BtB     Ip'        zeros(n,lq) ;   % dL/du   = 0
           -BtB        BtB   zeros(n,lp)  Iq'         ;   % dL/dv   = 0
             Ip        zeros(lp,n+lp+lq)              ;   % dL/d mu = 0
          zeros(lq,n)   Iq   zeros(lq,lp+lq)          ];  % dL/d nu = 0

   weight_p = abs(c(Au)).^w + w*1e-5;
   weight_q = abs(c(Av)).^w + w*1e-5;
 
   % right-hand-side vector -- standard L1
   BTY = [ Bty - BtB*p + BtB*q - alfa   ;                % dL/du   = 0
          -Bty + BtB*p - BtB*q - alfa   ;                % dL/dv   = 0
           -p(Au) ./ weight_p            ;                % dL/d mu = 0
           -q(Av) ./ weight_q            ];               % dL/d nu = 0
  
   u_v_mu_nu = BTB \ BTY;                % solve the system

%
   u      = u_v_mu_nu(1:n);                        % update for p
   v      = u_v_mu_nu(n+1:2*n);                        % update for q
   mu(Au) = u_v_mu_nu(2*n+1:2*n+lp);                % what to use mu for?
   nu(Av) = u_v_mu_nu(2*n+lp+1:2*n+lp+lq);        % what to use nu for?

   % if an element of p+u becomes negetive, reduce the step length of u
   du = 1;
   [p_min, j] = min(p+u);
   if p_min < 0 
     du = -p(j)/u(j);
   end

   % if an element of q+v becomes negetive, reduce the step length of v
   dv = 1;
   [q_min, j] = min(q+v);
   if q_min < 0 
     dv = -q(j)/v(j);
   end

   err_norm = norm(B*(p+du*u-q-dv*v)-y)/(m-n);

   if err_norm < err_norm_old
     err_norm_old = err_norm;
     p = p + du*u;
     q = q + dv*v;
     c = p - q;
     alfa = alfa * 1.2;
   else
     alfa = alfa / 1.1;
   end

   c(find(abs(c) < 1e-6)) = 0;    % zero-out small coefficients

   cvg_hst(:,iter) = [ c ; p ; q ; mu ; nu ; alfa ; err_norm ];  % convergence history

   % convergence check 
   % coefficients change by less than 1 percent, p>=0 and q>=0
   if ( norm(u) <= norm(p)/1e2  &&  min(p) > -1e-4  && ...
        norm(v) <= norm(q)/1e2  &&  min(q) > -1e-4 )
     break;
   end
%  if alfa < 1e-4 || ( err_norm > 1e3 && err_norm_old > 1e3 )
%    break
%  end

%  figure(101); clf; plot(t,y,'o', t,B*a); drawnow; 

 end

 cvg_hst = cvg_hst(:,1:iter);

% figure(102); clf; plot([1:iter], cvg_hst(1:n,:)' ,'-o'); drawnow; %pause(1);

% ------------------------------------------------------------------ L1_fit
% H.P. Gavin, 2013-10-04

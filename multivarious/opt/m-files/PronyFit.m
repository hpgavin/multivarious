function [ko,k,cvg_hst] = PronyFit(G_dat,f_dat,tau,alpha)
% [ko,k,cvg_hst] = PronyFit(G_dat,f_dat,tau,alpha)
% 
%  Fit a prony series to frequency domain complex modulus data
%
% INPUT:  G_dat  ...  complex modulus data                    Mx1 complex vector
%         f_dat  ...  frequencies where the measured frequency response data is 
%                     evaluated  (Hz),   (f>0)                Mx1 real vector
%          tau   ...  specified set of relaxation times,      Nx1 real vector
%        alpha   ...  l_1 regularization factor for Prony series coefficients
%
% OUTPUT: ko, k      ...  Prony series coefficients
%         cvg_hst    ...  convergence history


 i = sqrt(-1.0);

 m  = length(f_dat);					% number of data points
 w  = 2*pi * f_dat;					% one-sided frequency

 T = [ ones(m,1) ,  i*w*tau' ./ (i*w*tau' + 1.0) ];	% design matrix

% plot the basis functions (columns of T) with single-sided log-scale frequencies
figure(101)
 semilogx(f_dat, real(T), '-r', f_dat, imag(T),'-b')
  ylabel('basis functions   T_k(\omega) = i \omega \tau_k / (i \omega \tau_k + 1)')
  xlabel('frequency, f, Hz')
  axis( [ 0.5*min(f_dat) 1.5*max(f_dat) -0.05 1.05 ]  ) 

 [m,n] = size(T);	% m = number of data points; n = number of parameters

 TtT = 2*real(T'*T);	% taking real part is like adding complex conjugate
 TtG = 2*real(T'*G_dat);

 k = TtT \ TtG;	% O.L.S. fit is a better initial guess, even though infeasible, k<0

 MaxIter = 20;			% usually enough
 cvg_hst = zeros(2*n,MaxIter);	% convergence of coefficients and multipliers

 for iter = 1:MaxIter		% --- Start Main Loop

    A  = find(k < 2*eps); l = length(A);  % active set for update constraints
    % Ia is the constraint gradient w.r.t. the step vector h
    Ia = zeros(l,n); for i=1:l, Ia(i,A(i)) = 1;	end

    % KKT equations
    lambda = zeros(n,1);			% Lagrange multiplier
    XTX = [ TtT    Ia'      ;			% dL / dh
            Ia   zeros(l,l) ];			% dL / d lambda

    XTY = [ -TtT*k + TtG - alpha ; 		% dL / dh
            -k(A)                ];		% dL / d lambda

    h_lambda = XTX \ XTY;			% solve the system

%
    h = h_lambda(1:n);				% the step
    lambda(A) = h_lambda(n+1:end);		% the non-zero multipliers

   % if an element of k+h becomes negetive, reduce the length of step h to dh*h
   dh = 1;
   [h_test_min, idx] = min(k+h);
   if h_test_min < 0 
                   dh = -k(idx)/h(idx);
   end

   k = (k  +  dh * h);				% update the Prony coefficients

   figure(102)					% plot msmnt points and fit
    G_hat = T*k;
    clf
    subplot(211)
    semilogx ( w,real(G_dat),'or', w,real(G_hat),'-k')
     ylabel('storage modulus,  G''(\omega) ')
     legend('G''(\omega) meas.','G''(\omega) fit','location','northwest')
    subplot(212)
    semilogx ( w,imag(G_dat),'ob', w,imag(G_hat),'-k')
     xlabel('\omega, rad/s')
     ylabel('loss modulus, G"(\omega) ')
     legend('G"(\omega) meas.','G"(\omega) fit','location','northwest')
    drawnow

   cvg_hst(:,iter) = [ k ; lambda ]; 			% convergence history
   if ( norm(h) < norm(k)/1e3 && min(k) > -5*eps )	% convergence check
        break;
   end

 end				% --- End Main Loop

  cvg_hst = cvg_hst(:,1:iter);

% asymptotic standard errors of the Prony coefficients ???
% k_std_err = sqrt(norm(G_dat-T*k)/(m-n+1) * diag(real(inv(TtTT))));

 ko = k(1);
 k = k(2:n);

% ------------------------------------------------------------ PRONY_FIT
%                                                   HP Gavin, 2013-10-04

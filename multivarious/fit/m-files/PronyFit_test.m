% --- PronyFit_test.m
% test prony series fitting with L1 regularization

clear all

% simulated noisy data

% a relatively small number of 'true' time lags
tau = 1e-2*[ 0.01 0.1 1 10 100 1000 ]';
ko  = 1;				% static stiffness
k   = 0.5 ./ tau.^0.2;			% 'true' Prony series coefficients

points = 100;
f_dat = logspace(-3,3,points)';
iw = 2*pi*f_dat * sqrt(-1.0);

% generate complex modulus data with some additive noise and friction effects
T = [ ones(points,1) ,  iw*tau' ./ (iw*tau' + 1.0) ];
G_dat = T * [ ko ; k ] + 5e-2*(randn(points,1) + i*randn(points,1));

friction = 0.1; 		% definately NOT linear viscoelasticity!
G_dat = real(G_dat) + sqrt(-1)*(friction + imag(G_dat));

% plot the simulated data
figure(1)
 clf
% semilogx(f,real(G),'or',f,imag(G),'ob')
  subplot(311)
   semilogx(f_dat,real(G_dat),'or')
    ylabel('storage modulus')
  subplot(312)
   semilogx(f_dat,imag(G_dat),'ob')
    ylabel('loss modulus')
  subplot(313)
   semilogx(f_dat,imag(G_dat)./real(G_dat),'ok')
    ylabel('tan \delta')
    xlabel('frequency, f_{dat}, Hz')

% number of terms in the Prony series
nID = 97;

tau_id = logspace(-4.5,2,nID)';		% specified set of time lags, tau

alpha = 0.50;				% l_1 regularization parameter

[ko_id,k_id,cvg_hst] = PronyFit(G_dat,f_dat,tau_id,alpha); % do it!

f_fit  = logspace(-4,4,100);
iw_fit = 2*pi*f_fit * sqrt(-1.0); 
G_fit  = ko_id + sum( (k_id.*tau_id * iw_fit) ./ (tau_id * iw_fit + 1) ); 


 figure(1)
  clf
  subplot(311)
   semilogx(f_dat,real(G_dat),'or', f_fit, real(G_fit),'-k')
    ylabel('storage modulus')
  subplot(312)
   semilogx(f_dat,imag(G_dat),'ob', f_fit, imag(G_fit),'-k')
    ylabel('loss modulus')
  subplot(313)
   semilogx(f_dat,imag(G_dat)./real(G_dat),'ok', ...
            f_fit, imag(G_fit)./real(G_fit),'-k')
    ylabel('tan \delta')
    xlabel('frequency, f, Hz')


 figure(2)
  clf
  semilogx(tau,k,'kd', tau_id,k_id,'o');
   title('relaxation spectrum')
   legend('true','fit')
    xlabel('relaxation time, \tau_k')
    ylabel('coefficient, k_k')


 figure(3)
  MaxIter = size(cvg_hst,2);
  iter = [1:MaxIter];
  clf
    plot(iter,cvg_hst(1:nID+1,:),'-oo', iter,cvg_hst(nID+2:2*nID+2,:),'-xx');
     xlabel('iteration')
     ylabel('parameters: ''o'';  lagrange multipliers: ''x''')
     title('convergence history')


% mypolyfit_test.m
% test the mypolyfit function for least squares data fitting and error analysis

% ----------------------------------------------------------
% first generate some (x,y) data with measrument errors in y

 x_l =  -1;                    % low  value of the indpendent variables
 x_h =   1;                    % high value of the indpendent variables
 Nd  =  40;                    % number of data 

x = linspace( x_l, x_h ,Nd )'; % precise values of the independent variable

measurement_error = 0.20;  % root-mean-square of the simulated measurement error

y = -cos(4*x) + 1.0*x.^3 .*exp(-x/3) +  measurement_error*randn(Nd,1);

% ----------------------------------------------------------


% select a model by specifying the set of polynomial term exponents 

p = [ 0 1 2 3 4 ];              % powers involved in the fit

figNoA = 10;                    % figure number for plotting

[c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, condNo] = mypolyfit(x,y,p,figNoA);

p = [ 0   2 3 4 ];              % powers involved in the fit

figNoB = 20;                    % figure number for plotting

[c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, condNo] = mypolyfit(x,y,p,figNoB);

% ----------------------------------------------------------- mypolyfit_test.m 

pdfPlots = 1;
if pdfPlots
figure(figNoA)
  subplot(121)
  legend('location', 'south')
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-1a.pdf','-dpdfcrop')
figure(figNoA+1)
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-2a.pdf','-dpdfcrop')
figure(figNoA+2)
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-3a.pdf','-dpdfcrop')


figure(figNoB)
  subplot(121)
  legend('location', 'south')
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-1b.pdf','-dpdfcrop')
figure(figNoB+1)
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-2b.pdf','-dpdfcrop')
figure(figNoB+2)
  print('/home/hpgavin/Teaching/SystemID/CourseNotes/Figures/mypolyfit-3b.pdf','-dpdfcrop')
end % pdfPlots

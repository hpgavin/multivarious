% test_uniform.m
% Generate MATLAB reference outputs for uniform distribution

a = 2;
b = 8;

x_vals = linspace(a, b, 100)';
p_vals = linspace(0.001, 0.999, 100)';

pdf_vals = unifpdf(x_vals, a, b);
cdf_vals = unifcdf(x_vals, a, b);
inv_vals = unifinv(p_vals, a, b);

dlmwrite('x_vals_uniform.txt', x_vals, 'delimiter', '\t');
dlmwrite('pdf_vals_uniform.txt', pdf_vals, 'delimiter', '\t');
dlmwrite('cdf_vals_uniform.txt', cdf_vals, 'delimiter', '\t');
dlmwrite('p_vals_uniform.txt', p_vals, 'delimiter', '\t');
dlmwrite('inv_vals_uniform.txt', inv_vals, 'delimiter', '\t');

disp("✅ Saved uniform test outputs for Python.");

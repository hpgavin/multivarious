% test_normal.m
% Generate MATLAB reference outputs for normal distribution

mu = 0;       % Mean
sigma = 1.5;  % Standard deviation

x_vals = linspace(mu - 4*sigma, mu + 4*sigma, 100)';
p_vals = linspace(0.001, 0.999, 100)';

pdf_vals = normpdf(x_vals, mu, sigma);
cdf_vals = normcdf(x_vals, mu, sigma);
inv_vals = norminv(p_vals, mu, sigma);

dlmwrite('x_vals_normal.txt', x_vals, 'delimiter', '\t');
dlmwrite('pdf_vals_normal.txt', pdf_vals, 'delimiter', '\t');
dlmwrite('cdf_vals_normal.txt', cdf_vals, 'delimiter', '\t');
dlmwrite('p_vals_normal.txt', p_vals, 'delimiter', '\t');
dlmwrite('inv_vals_normal.txt', inv_vals, 'delimiter', '\t');

disp("✅ Normal distribution reference outputs saved.");

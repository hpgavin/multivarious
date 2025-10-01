% Parameters (same as in Python test)
a = 1;
b = 5;
c = 3;

% x-values split around mode (c), staying inside support
x_vals = linspace(a + 0.01, c - 0.01, 50)';
x_vals = [x_vals; linspace(c + 0.01, b - 0.01, 50)'];

% Probabilities for inverse CDF
p_vals = linspace(0.001, 0.999, 100)';

% Evaluate
pdf_vals = triangular_pdf(x_vals, a, b, c);
cdf_vals = triangular_cdf(x_vals, [a, b, c]);
inv_vals = triangular_inv(p_vals, a, b, c);

% Save
dlmwrite('x_vals.txt', x_vals, 'delimiter', '\t');
dlmwrite('pdf_vals.txt', pdf_vals, 'delimiter', '\t');
dlmwrite('cdf_vals.txt', cdf_vals, 'delimiter', '\t');
dlmwrite('p_vals.txt', p_vals, 'delimiter', '\t');
dlmwrite('inv_vals.txt', inv_vals, 'delimiter', '\t');

disp('Saved all reference outputs for Python comparison.');

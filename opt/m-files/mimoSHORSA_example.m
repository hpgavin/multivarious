% example_usage.m
% Example usage of mimoSHORSA for polynomial response surface fitting
%
% This script demonstrates:
% 1. Creating synthetic nonlinear data
% 2. Fitting a high-order polynomial model
% 3. Evaluating model performance
% 4. Visualizing results
%
% This MATLAB script mirrors example_usage.py for cross-validation


function example_usage()
% Run all examples

  fprintf('\n');
  fprintf('######################################################################\n');
  fprintf('# mimoSHORSA Example Usage (MATLAB)\n');
  fprintf('######################################################################\n');
  
  fprintf('\n\nRunning examples (this may take a few minutes)...\n\n');
  
  % Run examples
  [order1, coeff1, testModelY1, testX1, testY1] = example_1_simple_polynomial();
  
% [order2, coeff2, testModelY2, testX2, testY2] = example_2_multi_output();
  
% [order3, coeff3, testModelY3, testX3, testY3] = example_3_high_dimensional();
  
% example_4_with_scaling();
  
  % Visualize one example
  visualize_model_performance(testY1, testModelY1);
  
% fprintf('\n');
% fprintf('######################################################################\n');
% fprintf('# All examples completed successfully!\n');
% fprintf('######################################################################\n');
% fprintf('\nKey Takeaways:\n');
% fprintf('  1. mimoSHORSA automatically identifies important polynomial terms\n');
% fprintf('  2. Model reduction removes uncertain coefficients iteratively\n');
% fprintf('  3. Scaling is important for numerical stability\n');
% fprintf('  4. The method works for single and multiple outputs\n');
% fprintf('  5. Higher dimensions require careful choice of maxOrder\n\n');

end % ====================================================== function example_usage


function [order, coeff, testModelY, testX, testY] = example_1_simple_polynomial()
% Example 1: Fit a simple 2D polynomial function

  fprintf('\n');
  fprintf('======================================================================\n');
  fprintf('Example 1: Simple 2D Polynomial\n');
  fprintf('======================================================================\n');
  
  rand('seed', 42);
  randn('seed', 42);
  
  nInp = 2;    % 2 input variables
  nOut = 1;    % 1 output variable
  mData = 950; % 150 data points
  
  % Generate input data
  dataX = 2 * randn(nInp, mData);
  
  % Generate output: y = 1 + 2*x1 + 0.5*x2^2 + 0.3*x1*x2 + noise
  dataY = zeros(nOut, mData);
  dataY(1, :) = 0.0 + ...
                0.5 * dataX(1, :) + ...
                0.5 * dataX(1, :) .^ 2 + ...
               -0.5 * dataX(2, :) + ...
               -0.5 * dataX(2, :) .^ 2 + ...
                2.0 * dataX(1, :) .* dataX(2, :) + ...
                2.00 * randn(1, mData);
  
  % Fit model
  fprintf('\nFitting model...\n');
  [order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
      mimoSHORSA(dataX, dataY, 2, 70, 40, 0.10, 2, 5e1, 'L');
  %      maxOrder=3, 
  %        pTrain=70, 
  %         pCull=40, 
  %           tol=0.10, 
  %       scaling=2, 
  %      L1_pnlty=50
  %    basis_fctn='P'

% plot the model (for nInp == 2) 
  for ii = 1:nOut
    figure(2000+ii)
    clf
    hold on
    plot3(testX(1,:), testX(2,:),  testY(ii,:),'ok')
    plot3(testX(1,:), testX(2,:), testModelY(ii,:),'og')
    xlabel('X_1')
    ylabel('X_2')
    zlabel('Y')
  end

%{
  fprintf('\n');
  fprintf('----------------------------------------------------------------------\n');
  fprintf('Final Model Summary:\n');
  fprintf('----------------------------------------------------------------------\n');
  fprintf('Number of terms: %d\n', size(order{1}, 1));
  fprintf('\nTop 5 coefficients:\n');
  
  [~, top_indices] = sort(abs(coeff{1}), 'descend');
  top_indices = top_indices(1:min(5, length(coeff{1})));
  
  for i = 1:length(top_indices)
    idx = top_indices(i);
    fprintf('  Term %d: powers=[%s], coeff=%8.4f\n', idx, ...
            sprintf('%d ', order{1}(idx, :)), coeff{1}(idx));
  end
%}

end % =================================== function example_1_simple_polynomial


function [order, coeff, testModelY, testX, testY] = example_2_multi_output()
% Example 2: Multi-output system with coupling

  fprintf('\n');
  fprintf('======================================================================\n');
  fprintf('Example 2: Multi-Output System\n');
  fprintf('======================================================================\n');
  
  rand('seed', 123);
  randn('seed', 123);
  
  nInp = 3;    % 3 input variables
  nOut = 2;    % 2 output variables
  mData = 200; % 200 data points
  
  % Generate input data
  dataX = randn(nInp, mData);
  
  % Generate coupled outputs
  dataY = zeros(nOut, mData);
  
  % Output 1: y1 = 1 + x1 + 0.5*x2^2 + 0.2*x1*x3
  dataY(1, :) = 1.0 + ...
                1.0 * dataX(1, :) + ...
                0.5 * dataX(2, :) .^ 2 + ...
                0.2 * dataX(1, :) .* dataX(3, :) + ...
                0.15 * randn(1, mData);
  
  % Output 2: y2 = 0.5 + 1.5*x2 - 0.8*x3^2 + 0.3*x1*x2
  dataY(2, :) = 0.5 + ...
                1.5 * dataX(2, :) - ...
                0.8 * dataX(3, :) .^ 2 + ...
                0.3 * dataX(1, :) .* dataX(2, :) + ...
                0.15 * randn(1, mData);
  
  % Fit model
  fprintf('\nFitting multi-output model...\n');
  [order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
      mimoSHORSA(dataX, dataY, 3, 75, 35, 0.18, 1);
  %             maxOrder=3, pTrain=75, pCull=35, tol=0.18, scaling=1
  
  fprintf('\n');
  fprintf('----------------------------------------------------------------------\n');
  fprintf('Final Model Summary:\n');
  fprintf('----------------------------------------------------------------------\n');
  
  for io = 1:nOut
    fprintf('\nOutput %d:\n', io);
    fprintf('  Number of terms: %d\n', size(order{io}, 1));
    fprintf('  Top 3 coefficients:\n');
    
    [~, top_indices] = sort(abs(coeff{io}), 'descend');
    top_indices = top_indices(1:min(3, length(coeff{io})));
    
    for i = 1:length(top_indices)
      idx = top_indices(i);
      fprintf('    Term %d: powers=[%s], coeff=%8.4f\n', idx, ...
              sprintf('%d ', order{io}(idx, :)), coeff{io}(idx));
    end
  end

end % ================================================ function example_2_multi_output


function [order, coeff, testModelY, testX, testY] = example_3_high_dimensional()
% Example 3: Higher dimensional problem

  fprintf('\n');
  fprintf('======================================================================\n');
  fprintf('Example 3: High-Dimensional Problem\n');
  fprintf('======================================================================\n');
  
  rand('seed', 456);
  randn('seed', 456);
  
  nInp = 5;    % 5 input variables
  nOut = 1;    % 1 output variable
  mData = 300; % 300 data points
  
  % Generate input data
  dataX = randn(nInp, mData);
  
  % Generate output: complex nonlinear function
  dataY = zeros(nOut, mData);
  dataY(1, :) = 2.0 + ...
                1.5 * dataX(1, :) + ...
                0.8 * dataX(2, :) .^ 2 - ...
                0.6 * dataX(3, :) .^ 2 + ...
                0.4 * dataX(4, :) + ...
                0.3 * dataX(1, :) .* dataX(2, :) + ...
                0.2 * dataX(3, :) .* dataX(5, :) + ...
                0.25 * randn(1, mData);
  
  % Fit model with lower maximum order due to curse of dimensionality
  fprintf('\nFitting high-dimensional model...\n');
  [order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
      mimoSHORSA(dataX, dataY, 2, 80, 40, 0.25, 1);
  %             maxOrder=2, pTrain=80, pCull=40, tol=0.25, scaling=1
  
  fprintf('\n');
  fprintf('----------------------------------------------------------------------\n');
  fprintf('Final Model Summary:\n');
  fprintf('----------------------------------------------------------------------\n');
  fprintf('Number of terms: %d\n', size(order{1}, 1));
  fprintf('\nMost significant terms:\n');
  
  [~, top_indices] = sort(abs(coeff{1}), 'descend');
  top_indices = top_indices(1:min(6, length(coeff{1})));
  
  for i = 1:length(top_indices)
    idx = top_indices(i);
    powers = order{1}(idx, :);
    
    % Build term string
    term_parts = {};
    for j = 1:length(powers)
      if powers(j) > 0
        term_parts{end+1} = sprintf('x%d^%d', j, powers(j));
      end
    end
    
    if isempty(term_parts)
      term_str = 'constant';
    else
      term_str = strjoin(term_parts, ' ');
    end
    
    fprintf('  %s: coeff=%8.4f\n', term_str, coeff{1}(idx));
  end

end % =========================================== function example_3_high_dimensional


function example_4_with_scaling()
% Example 4: Demonstrate different scaling options

  fprintf('\n');
  fprintf('================================================================\n');
  fprintf('Example 4: Scaling Options Comparison\n');
  fprintf('================================================================\n');
  
  rand('seed', 789);
  randn('seed', 789);
  
  nInp = 2;
  nOut = 1;
  mData = 100;
  
  % Generate data with different scales
  dataX = zeros(nInp, mData);
  dataX(1, :) = 100 * randn(1, mData);   % Large scale
  dataX(2, :) = 0.01 * randn(1, mData);  % Small scale
  
  % Generate output
  dataY = zeros(nOut, mData);
  dataY(1, :) = 0.01 * dataX(1, :) + 100 * dataX(2, :) .^ 2 + randn(1, mData);
  
  scaling_names = {'No scaling', ...
                   'Standardization (mean=0, std=1)', ...
                   'Decorrelation (whitening)'};
  
  for scaling_option = 0:2
    fprintf('\n--- Scaling Option %d: %s ---\n', ...
            scaling_option, scaling_names{scaling_option + 1});
    
    try
      [order, coeff, ~, ~, ~, ~, ~, ~, ~] = ...
          mimoSHORSA(dataX, dataY, 2, 70, 30, 0.25, scaling_option);
      %             maxOrder=2, pTrain=70, pCull=30, tol=0.25
      
      fprintf('Successfully fitted with %d terms\n', size(order{1}, 1));
    catch ME
      fprintf('Error: %s\n', ME.message);
    end
  end

end % ======================================== function example_4_with_scaling


function visualize_model_performance(testY, testModelY)
% Create visualization of model performance

  nOut = size(testY, 1);
  
  figure('Position', [100, 100, 600*nOut, 500]);
  
  for io = 1:nOut
    subplot(1, nOut, io);
    
    % Scatter plot
    scatter(testModelY(io, :), testY(io, :), 50, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    
    % Perfect prediction line
    min_val = min([min(testY(io, :)), min(testModelY(io, :))]);
    max_val = max([max(testY(io, :)), max(testModelY(io, :))]);
    plot([min_val, max_val], [min_val, max_val], '--g', 'LineWidth', 2);
    
    % Correlation
    corrMatrix = corrcoef(testY(io, :), testModelY(io, :));
    corr_val = corrMatrix(1, 2);
    
    xlabel('Model Prediction', 'FontSize', 12);
    ylabel('True Value', 'FontSize', 12);
    title(sprintf('Output %d: \\rho = %.3f', io, corr_val), 'FontSize', 14);
    legend('Data', 'Perfect prediction', 'Location', 'NorthWest');
    grid on;
    axis equal;
    hold off;
  end
  
  % Save figure
  saveas(gcf, 'model_performance_matlab.png');
  fprintf('\nPerformance plot saved to ''model_performance_matlab.png''\n');

end % ======================================= function visualize_model_performance


% Helper function for strjoin (if not available in older MATLAB versions)
function str = strjoin(parts, delimiter)
  if isempty(parts)
    str = '';
    return;
  end
  
  str = parts{1};
  for i = 2:length(parts)
    str = [str, delimiter, parts{i}];
  end
end

% ========================================================== script example_usage.m
% 
% To run this script, simply type:
%   example_usage
% 
% Or to run from command line:
%   matlab -batch "example_usage"
%
% Make sure mimoSHORSA.m and all its subfunctions are in your MATLAB path
%
% Updated: 2025-10-23

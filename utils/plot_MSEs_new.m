function plot_MSEs_new(total_mse, parameters, varargin)
    % PLOT avg_mse and node_mse for each one method
    %
    % Inputs:
    %   total_mse   - Struct array (parameter sweep) or struct (single run):
    %                 .avg_mse: 1xT vector of average MSE
    %                 .node_mse: NxT matrix of node-wise MSE
    %   parameters  - Parameter values for sweep ([] for single run)
    %
    % Optional Parameters:
    %   'Method'      - DRO method name (e.g., 'KL Divergence')
    %   'ParamName'   - Parameter name (e.g., 'c')
    %   'PlotOptimal' - Plot time-series for optimal parameter (true/false)
    %   'FigureSize'  - Figure dimensions [width height] (default: [1200 600])
    %   'Language'    - Display language ('en' for English, 'cn' for Chinese)

    % Input parsing
    p = inputParser;
    addRequired(p, 'total_mse', @(x) isstruct(x) || all(isstruct([x(:)])));
    addRequired(p, 'parameters', @isnumeric);
    addParameter(p, 'Method', 'DRO', @ischar);
    addParameter(p, 'ParamName', 'Parameter', @ischar);
    addParameter(p, 'PlotOptimal', false, @islogical);
    addParameter(p, 'FigureSize', [1200 600], @isnumeric);
    addParameter(p, 'Language', 'en', @(x) any(validatestring(x, {'en','cn'}))); 
    
    parse(p, total_mse, parameters, varargin{:});
    is_sweep = numel(total_mse) > 1;

    % Main plotting logic
    if is_sweep
        plot_parameter_sweep(total_mse, parameters, p);
        if p.Results.PlotOptimal
            plot_optimal_time_series(total_mse, parameters, p);
        end
    else
        plot_time_series(total_mse, p);
    end
end

%% Helper functions
function plot_parameter_sweep(total_mse, params, p)
    % Language settings
    lang = struct(...
        'title', struct('en', '%s Parameter Sweep\nOptimal %s = %.3f', ...
                        'cn', '%s 参数扫描分析\n最优参数 %s = %.3f'),...
        'xlabel', struct('en', p.Results.ParamName, 'cn', p.Results.ParamName),...
        'ylabel', struct('en', 'Average MSE', 'cn', '平均均方误差'),...
        'legend', struct('en', {'Average MSE', 'Optimal Value'}, ...
                         'cn', {'平均MSE', '最优参数值'}),...
        'figureName', struct('en', 'Parameter Sweep Analysis', ...
                             'cn', '参数扫描分析')...
    );

    % Parameter sweep visualization
    mean_mse = arrayfun(@(x) mean(x.avg_mse), total_mse);
    [min_mse, idx] = min(mean_mse);
    optimal_param = params(idx);

    fig = figure('Position', [100 100 p.Results.FigureSize], ...
        'Name', lang.figureName.(p.Results.Language), ...
        'NumberTitle', 'off', ...
        'DefaultAxesFontName', 'Microsoft YaHei');
    
    plot(params, mean_mse, 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    hold on;
    plot(optimal_param, min_mse, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
    xlabel(lang.xlabel.(p.Results.Language));
    ylabel(lang.ylabel.(p.Results.Language));
    title(sprintf(lang.title.(p.Results.Language), ...
        p.Results.Method, p.Results.ParamName, optimal_param));
    
    grid on;
    legend(lang.legend.(p.Results.Language), 'Location', 'best');
end

function plot_optimal_time_series(total_mse, params, p)
    % Language settings
    lang = struct(...
        'mainTitle', struct('en', '%s Optimal Performance (%s = %.3f)', ...
                            'cn', '%s 最优参数性能 (%s = %.3f)'),...
        'networkMSE', struct('en', 'Network MSE', 'cn', '网络平均MSE'),...
        'nodeMSE', struct('en', 'Node MSE', 'cn', '节点MSE'),...
        'timeStep', struct('en', 'Time Step', 'cn', '时间步长'),...
        'figureName', struct('en', 'Optimal Parameter Performance', ...
                             'cn', '最优参数性能分析')...
    );
    % Time-series visualization for optimal parameter
    [~, idx] = min(arrayfun(@(x) mean(x.avg_mse), total_mse));
    optimal_data = total_mse(idx);
    
    fig = figure('Position', [100 100 p.Results.FigureSize], ...
        'Name', lang.figureName.(p.Results.Language), ...
        'NumberTitle', 'off', ...
        'DefaultAxesFontName', 'Microsoft YaHei');
    
    % Plot network average MSE
    subplot(2,1,1);
    plot(optimal_data.avg_mse, 'LineWidth', 2);
    title(sprintf(lang.mainTitle.(p.Results.Language), ...
        p.Results.Method, p.Results.ParamName, params(idx)));
    ylabel(lang.networkMSE.(p.Results.Language));
    grid on;
    
    % Plot all nodes' MSE
    subplot(2,1,2);
    hold on;
    colors = lines(size(optimal_data.node_mse, 1));  % Get colors for all nodes
    for node_idx = 1:size(optimal_data.node_mse, 1)
        plot(optimal_data.node_mse(node_idx, :), ...
            'Color', colors(node_idx, :), ...
            'LineWidth', 1.5);
    end
    ylim([0,40]);
    xlabel(lang.timeStep.(p.Results.Language));
    ylabel(lang.nodeMSE.(p.Results.Language));
    legend(arrayfun(@(k) sprintf('Node %d', k), 1:size(optimal_data.node_mse, 1), ...
        'UniformOutput', false), ...
        'Location', 'best');
    grid on;
end

function plot_time_series(total_mse, p)
    % Language settings
    lang = struct(...
        'mainTitle', struct('en', '%s Performance Analysis', ...
                            'cn', '%s 性能分析'),...
        'networkMSE', struct('en', 'Network MSE', 'cn', '网络平均MSE'),...
        'nodeMSE', struct('en', 'Node MSE', 'cn', '节点MSE'),...
        'timeStep', struct('en', 'Time Step', 'cn', '时间步长'),...
        'figureName', struct('en', 'Time-Series Performance', ...
                             'cn', '时间序列性能分析')...
    );
    % Single run time-series visualization
    fig = figure('Position', [100 100 p.Results.FigureSize], ...
        'Name', lang.figureName.(p.Results.Language), ...
        'NumberTitle', 'off', ...
        'DefaultAxesFontName', 'Microsoft YaHei');
    
    % Plot network average MSE
    subplot(2,1,1);
    plot(total_mse.avg_mse, 'LineWidth', 2);
    title(sprintf(lang.mainTitle.(p.Results.Language), p.Results.Method));
    ylabel(lang.networkMSE.(p.Results.Language));
    grid on;
    
    % Plot node-wise MSE with proper hold state management
    subplot(2,1,2);
    hold on;
    colors = lines(size(total_mse.node_mse,1));
    for k = 1:size(total_mse.node_mse,1)
        plot(total_mse.node_mse(k,:), ...
            'Color', colors(k,:), ...
            'LineWidth', 1.5);
    end
    hold off;
    % ylim([0,40]);
    
    xlabel(lang.timeStep.(p.Results.Language));
    ylabel(lang.nodeMSE.(p.Results.Language));
    legend(arrayfun(@(k) sprintf('Node %d', k), ...
        1:size(total_mse.node_mse,1), ...
        'UniformOutput', false), ...
        'Location', 'best');
    grid on;
end
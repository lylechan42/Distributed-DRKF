function plot_avg_mse(data)
    % Function to plot network average MSE for DKF and DRKF methods.
    %
    % Input:
    %   data - A struct array containing 12 structs:
    %          - 3 structs for DKF (no, average, CI)
    %          - 9 structs for DRKF (KL, MC, W with no, average, CI for each)
    %          Each struct should have:
    %          - name: Name of the method (e.g., "DKF-no", "DRKF-KL-average")
    %          - avg_mse: A vector of average MSE values over time
    %
    % Example usage:
    %   plot_avg_mse(data)
    
    % Validate input
    if length(data) ~= 12
        error('Input must be a struct array containing 12 elements.');
    end
    
    % Separate DKF and DRKF data
    DKF = data(1:3); % First 3 structs are DKF
    DRKF = data(4:end); % Remaining 9 structs are DRKF
    
    % Set colors and line styles
    colors = lines(12); % Color palette
    line_styles = {'-', '--', ':'}; % Line styles
    
    % Create figure
    figure;
    hold on;
    title('Network Average MSE Comparison');
    xlabel('Time Step');
    ylabel('Average MSE');
    grid on;
    
    % Plot DKF methods directly
    for i = 1:length(DKF)
        plot(DKF(i).avg_mse, 'Color', colors(i, :), 'LineStyle', line_styles{1}, ...
            'LineWidth', 1.5, 'DisplayName', DKF(i).name);
    end
    
    % Process and plot DRKF methods
    DRKF_methods = ["DRKF-KL", "DRKF-MC", "DRKF-W"];
    for i = 1:length(DRKF_methods)
        % Extract branches for current DRKF method
        method_idx = (i-1)*3 + (1:3); % Indices for current method (no, average, CI)
        branches = DRKF(method_idx);
        
        % Find the branch with the minimum mean(avg_mse)
        min_mean_mse = inf;
        best_branch_idx = 0;
        for j = 1:length(branches)
            current_mean = mean(branches(j).avg_mse);
            if current_mean < min_mean_mse
                min_mean_mse = current_mean;
                best_branch_idx = j;
            end
        end
        
        % Plot the best branch
        best_branch = branches(best_branch_idx);
        plot(best_branch.avg_mse, 'Color', colors(i + 3, :), ...
            'LineStyle', line_styles{2}, 'LineWidth', 1.5, ...
            'DisplayName', best_branch.name);
    end
    
    % Add legend
    legend('show', 'Location', 'best');
    hold off;
end


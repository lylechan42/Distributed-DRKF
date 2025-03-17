close all; clear all; clc
addpath('utils');
dt = 1;               % Time step
T = 200;

%% Task Initialization

vars.b1 = .5;
vars.b2 = .8;
vars.b3 = .4;
vars.q1 = 1;
vars.q11 = 5;
vars.q12 = 8;
vars.q13 = 10;

[sys,x0,P0] = getSys(dt,3,vars);

% Fetch data
[x,y] = getCorrData(x0,T,sys,vars);

n = size(x,1); m = size(y,1); 
mi = 1; p = 3;    

% functions
x_hat_Z = @(Z, n, mu, y) -Z(1:n,n+1:end)*y + mu(1:n) + Z(1:n,n+1:end)*mu(n+1:end);
x_hat_S = @(mu, S, n, y) mu(1:n) + S(1:n, n+1:end) * (S(n+1:end, n+1:end) \ (y - mu(n+1:end)));
P_hat_S = @(S, n) S(1:n, 1:n) - S(1:n, n+1:end) * (S(n+1:end, n+1:end) \ S(n+1:end, 1:n));

% Network Initialization
num_nodes = 4;  % Square network
adj_matrix = [0 1 0 1;   % Node 1 connected to 2 & 4
              1 0 1 0;   % Node 2 connected to 1 & 3
              0 1 0 1;   % Node 3 connected to 2 & 4
              1 0 1 0];  % Node 4 connected to 1 & 3

% % Visualize network
% figure;
% g = graph(adj_matrix);
% plot(g, 'Layout', 'force', 'NodeLabel', {'1','2','3','4'});
% title('4-Node Square Network Topology');

% Node Configuration
node_config = struct(...
    'sensors', { {1,2}, {2,3}, {1}, {3} }, ...  % Sensor indices
    'color',   { 'r',   'g',   'b', 'm' } ...   % Visualization colors
);
nodes = struct('sys', cell(num_nodes,1), ...
               'neighbors', cell(num_nodes,1), ...
               'color', cell(num_nodes,1), ...
               'estimates', cell(num_nodes,1));

%%
num_workers = 8; % Number of workers (adjust based on your CPU)
pool = parpool('Processes', num_workers);


%% KL divergence
num_param = 1;
num_runs = 16;  % Number of Monte Carlo runs
method = 'kl_divergence';

% KL_no
all_c =  3e-1*linspace(0,1,num_param);

diffusion_enabled = false;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_KL_no(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_c(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_KL_no(i).avg_mse = result.avg_mse;
    MSE_DRKF_KL_no(i).node_mse = result.node_mse;
end
time_KL_no = toc;

% KL_average
all_c =  3e-1*linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_KL_average(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_c(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_KL_average(i).avg_mse = result.avg_mse;
    MSE_DRKF_KL_average(i).node_mse = result.node_mse;
end
time_KL_average = toc;

% KL_CI
all_c =  3e-1*linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'CI';   
estimation_method = 'DRO';

MSE_DRKF_KL_CI(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_c(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_KL_CI(i).avg_mse = result.avg_mse;
    MSE_DRKF_KL_CI(i).node_mse = result.node_mse;
end
time_KL_CI = toc;

% output time
result = (time_KL_no / num_runs / T) * 1000 * num_workers;
fprintf('KL_no result: %.6f\n', result);
result = (time_KL_average / num_runs / T) * 1000 * num_workers;
fprintf('KL_average result: %.6f\n', result);
result = (time_KL_CI / num_runs / T) * 1000 * num_workers;
fprintf('KL_CI result: %.6f\n', result);


%% Wasserstein
num_param = 1;
num_runs = 16;  % Number of Monte Carlo runs
method = 'wasserstein';

% ======================
% W_no
all_rho = 5e-1*linspace(0,1,num_param);

diffusion_enabled = false;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_W_no(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_rho(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_W_no(i).avg_mse = result.avg_mse;
    MSE_DRKF_W_no(i).node_mse = result.node_mse;
end
time_W_no = toc;

% W_average
all_rho = 5e-1*linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_W_average(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_rho(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_W_average(i).avg_mse = result.avg_mse;
    MSE_DRKF_W_average(i).node_mse = result.node_mse;
end
time_W_average = toc;

% W_CI
all_rho = 5e-1*linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'CI';   
estimation_method = 'DRO';

MSE_DRKF_W_CI(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_rho(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_W_CI(i).avg_mse = result.avg_mse;
    MSE_DRKF_W_CI(i).node_mse = result.node_mse;
end
time_W_CI = toc;

% output time
result = (time_W_no / num_runs / T) * 1000 * num_workers;
fprintf('W_no result: %.6f\n', result);
result = (time_W_average / num_runs / T) * 1000 * num_workers;
fprintf('W_average result: %.6f\n', result);
result = (time_W_CI / num_runs / T) * 1000 * num_workers;
fprintf('W_CI result: %.6f\n', result);


%% Moment based
num_param = 1;
num_runs = 16;  % Number of Monte Carlo runs
method = 'moment_based';

% ======================
% M_no
all_gamma = linspace(0,1,num_param);

diffusion_enabled = false;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_M_no(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_gamma(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_M_no(i).avg_mse = result.avg_mse;
    MSE_DRKF_M_no(i).node_mse = result.node_mse;
end
time_M_no = toc;

% M_average
all_gamma = linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'average';   
estimation_method = 'DRO';

MSE_DRKF_M_average(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_gamma(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_M_average(i).avg_mse = result.avg_mse;
    MSE_DRKF_M_average(i).node_mse = result.node_mse;
end
time_M_average = toc;

% M_CI
all_gamma = linspace(0,1,num_param);

diffusion_enabled = true;  
diffusion_method = 'CI';   
estimation_method = 'DRO';

MSE_DRKF_M_CI(num_param) = struct('avg_mse', [], 'node_mse', []);

tic;
for i = 1:num_param
    fprintf('\n=== Running parameter %d/%d ===\n', i, num_param);
    result = run_monte_carlo_simulation(...
        sys, node_config, adj_matrix, x0, P0, T, vars, ...
        num_runs, method, all_gamma(i), diffusion_enabled, diffusion_method, estimation_method...
    );
    
    % Store results
    MSE_DRKF_M_CI(i).avg_mse = result.avg_mse;
    MSE_DRKF_M_CI(i).node_mse = result.node_mse;
end
time_M_CI = toc;

% output time
result = (time_M_no / num_runs / T) * 1000 * num_workers;
fprintf('M_no result: %.6f\n', result);
result = (time_M_average / num_runs / T) * 1000 * num_workers;
fprintf('M_average result: %.6f\n', result);
result = (time_M_CI / num_runs / T) * 1000 * num_workers;
fprintf('M_CI result: %.6f\n', result);


%% funcs
function [mu_t, Sigma_t] = predict(x_prev, V_prev, sys)
    A_aug = [sys.F; sys.H * sys.F];
    B_aug = [
        sys.G*(sys.Q)*sys.G'                sys.G*(sys.Q)*sys.G'*sys.H'
        sys.H*sys.G*(sys.Q)*sys.G'          sys.H*sys.G*(sys.Q)*sys.G'*sys.H' + sys.R
    ];
    mu_t = A_aug * x_prev;
    Sigma_t = A_aug * V_prev * A_aug' + B_aug;
end


function total_mse = run_monte_carlo_simulation(...
    sys, node_config, adj_matrix, x0, P0, T, vars, ...
    num_runs, method, parameter, diffusion_enabled, diffusion_method, estimation_method...
)
    % RUN_MONTE_CARLO_SIMULATION Perform Monte Carlo analysis of distributed estimation
    % Inputs:
    %   sys - System model
    %   node_config - Node sensor configuration
    %   adj_matrix - Network adjacency matrix
    %   x0, P0 - Initial state and covariance
    %   T - Number of time steps
    %   vars - Process/measurement noise parameters
    %   num_runs - Number of Monte Carlo runs
    %   method - DRO method ('kl_divergence', 'wasserstein', 'moment_based')
    %   parameter - DRO parameter value
    %   diffusion_enabled - Boolean for diffusion
    %   diffusion_method - 'CI' or 'average'
    %   estimation_method - 'DRO' or 'DKF'
    %
    % Output:
    %   total_mse - Struct with node-wise and average MSE results
    mc_mse = cell(num_runs, 1);  % Store MSE for each run
    num_nodes = size(adj_matrix, 1);
  
    
    parfor run_idx = 1:num_runs
        
        % Generate new data for each run
        [x_mc, y_mc] = getCorrData(x0, T, sys, vars);
        
        % Initialize nodes for this run
        nodes_mc = initialize_nodes(sys, node_config, adj_matrix, x0, T);
        
        % Run estimation
        nodes_mc = run_distributed_estimation(...
            nodes_mc, node_config, y_mc, x0, P0, T, ...
            method, parameter, diffusion_enabled, diffusion_method, estimation_method...
        );
        
        % Calculate MSE for this run
        mc_mse{run_idx} = calculate_mse(nodes_mc, x_mc);
    
        fprintf('Complete run %d/%d\n', run_idx, num_runs);
    end
    
    % Aggregate Results
    total_mse = struct();
    total_mse.node_mse = zeros(num_nodes, T);
    total_mse.avg_mse = zeros(1, T);
    
    for run_idx = 1:num_runs
        total_mse.node_mse = total_mse.node_mse + mc_mse{run_idx}.node_mse;
        total_mse.avg_mse = total_mse.avg_mse + mc_mse{run_idx}.avg_mse;
    end
    
    total_mse.node_mse = total_mse.node_mse / num_runs;
    total_mse.avg_mse = total_mse.avg_mse / num_runs;

end

function nodes = run_distributed_estimation(...
    nodes, node_config, y, x0, P0, T, ...
    method, parameter, diffusion_enabled, diffusion_method, estimation_method...
)
    % Distributed estimation with DRO and optional diffusion
    % Inputs:
    %   nodes - preconfigured node structures
    %   node_config - sensor configuration
    %   y - measurement data [m x T]
    %   x0, P0 - initial state and covariance
    %   T - time steps
    %   method - DRO method ('kl_divergence', 'wasserstein', etc.)
    %   parameter - DRO parameter (c, rho, gamma)
    %   diffusion_enabled - boolean for diffusion
    %   diffusion_method - 'CI' or 'average'
    %   estimation_method - 'DRO' or 'DKF'
    n = size(x0,1);
    num_nodes = size(nodes,1);
    x_hat_S = @(mu, S, n, y) mu(1:n) + S(1:n, n+1:end) * (S(n+1:end, n+1:end) \ (y - mu(n+1:end)));
    P_hat_S = @(S, n) S(1:n, 1:n) - S(1:n, n+1:end) * (S(n+1:end, n+1:end) \ S(n+1:end, 1:n));
    for t = 1:T
        % Phase 1: Distributed DRO
        temp_estimates = cell(num_nodes,1);
        for k = 1:num_nodes
            node = nodes(k);
            neighbors = node.neighbors;
            all_nodes = [k, neighbors];
            % 1. Get aug y and sys
            y_list = cell(1, length(all_nodes));
            for i = 1:length(all_nodes)
                node_id = all_nodes(i);
                sensors = node_config(node_id).sensors; 
                y_list{i} = y([sensors{:}], t);
            end
            y_aug = vertcat(y_list{:});
            sys_aug = node.sys_aug;
            % 2. Prediction with augmented model
            if t == 1
                prior_x = x0;
                prior_P = P0;
            else
                prior_x = node.estimates.x(:,t-1);
                prior_P = node.estimates.P(:,:,t-1);
            end
            [mu, Sigma] = predict(prior_x, prior_P, sys_aug);
            
            switch estimation_method
                case 'DRO'
                % 3. DRO Update with full measurement set
                    [S_opt, U_opt] = DRO(method, parameter, n, size(y_aug,1), Sigma, true);
                    x_hat = x_hat_S(mu, S_opt, n, y_aug);
                    V_hat = U_opt;
                case 'DKF'
                    x_hat = x_hat_S(mu, Sigma, n, y_aug);
                    V_hat = P_hat_S(Sigma, n);
                otherwise
                    error('Unknown estimation method: %s', estimation_method);
            end

            % Store temporary estimate
            temp_estimates{k} = struct(...
                'x', x_hat, ...
                'P', V_hat ...
            );
        end
    
        % Phase 2: Conditional Diffusion
        if diffusion_enabled
            for k = 1:num_nodes
                node = nodes(k);
                neighbors = node.neighbors;
                
                % Collect neighbor estimates
                if ~isempty(neighbors)
                    % Preallocate
                    num_estimates = length(neighbors) + 1;
                    x_estimates = zeros(n, num_estimates);
                    P_estimates = zeros(n, n, num_estimates);
                    
                    % Populate
                    x_estimates(:,1) = temp_estimates{k}.x;
                    P_estimates(:,:,1) = temp_estimates{k}.P;
                    
                    for nbr_idx = 1:length(neighbors)
                        current_nbr = neighbors(nbr_idx);
                        x_estimates(:,nbr_idx+1) = temp_estimates{current_nbr}.x;
                        P_estimates(:,:,nbr_idx+1) = temp_estimates{current_nbr}.P;
                    end
    
                    switch diffusion_method
                        case 'CI'
                            [x_fused, P_fused] = fusecovint(x_estimates, P_estimates);
                        case 'average'
                            x_fused = mean(x_estimates, 2);
                            P_fused = mean(P_estimates, 3);
                        otherwise
                            error('Unknown fusion method: %s', diffusion_enabled);
                    end
                else
                    % No neighbors, use local estimate
                    x_fused = temp_estimates{k}.x;
                    P_fused = temp_estimates{k}.P;
                end
                
                % Store final estimate
                nodes(k).estimates.x(:,t) = x_fused;
                nodes(k).estimates.P(:,:,t) = P_fused;
            end
        else
            % No diffusion, use local estimates directly
            for k = 1:num_nodes
                nodes(k).estimates.x(:,t) = temp_estimates{k}.x;
                nodes(k).estimates.P(:,:,t) = temp_estimates{k}.P;
            end
        end

        % fprintf('Complete timestep %d/%d\n', t, T);
    end

end



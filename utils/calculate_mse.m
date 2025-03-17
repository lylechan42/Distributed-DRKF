function mse = calculate_mse(nodes, x)
    num_nodes = length(nodes);
    T = size(x, 2);
    mse.node_mse = zeros(num_nodes, T);
    
    for t = 1:T
        xt = x(:,t);
        for k = 1:num_nodes
            x_hat = nodes(k).estimates.x(:,t);
            mse.node_mse(k,t) = sum((xt - x_hat).^2);
        end
    end
    
    mse.avg_mse = mean(mse.node_mse, 1);
end
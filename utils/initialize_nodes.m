function nodes = initialize_nodes(sys, node_config, adj_matrix, x0, T)
    num_nodes = length(node_config);
    nodes = struct('sys', cell(num_nodes,1), ...
                   'neighbors', cell(num_nodes,1), ...
                   'estimates', cell(num_nodes,1));
    
    for k = 1:num_nodes % get node sub sys
        sensors = node_config(k).sensors;
        nodes(k).sys.H = sys.H(cell2mat(sensors), :);
        nodes(k).sys.R = sys.R(cell2mat(sensors), cell2mat(sensors));
    
        nodes(k).sys.F = sys.F;
        nodes(k).sys.G = sys.G;
        nodes(k).sys.Q = sys.Q;
    
        nodes(k).neighbors = find(adj_matrix(k,:));
        nodes(k).color = node_config(k).color;
    
        nodes(k).estimates.x = zeros(size(x0,1), T);
        nodes(k).estimates.P = zeros(size(x0,1), size(x0,1), T);
    end
    
    
    for k = 1:num_nodes % get node concatenated sys
        node = nodes(k);
        neighbors = node.neighbors;
        all_nodes = [k, neighbors];
    
        H_list = cell(1, length(all_nodes));
        R_list = cell(1, length(all_nodes));
        for i = 1:length(all_nodes)
            node_id = all_nodes(i);
            H_list{i} = nodes(node_id).sys.H;
            R_list{i} = nodes(node_id).sys.R;
        end
        nodes(k).sys_aug.H = vertcat(H_list{:});
        nodes(k).sys_aug.R = blkdiag(R_list{:});
        
        nodes(k).sys_aug.F = sys.F;  % Shared dynamics
        nodes(k).sys_aug.G = sys.G;
        nodes(k).sys_aug.Q = sys.Q;
    end
end
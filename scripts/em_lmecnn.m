function [lf_history, b_hat_history, gll_history] = em_lmecnn(X, Z, clusters, y, tids, max_iterations)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
n_obs = size(y, 1);
n_clusters = length(clusters);

q = size(Z, 2);
b_hat_df = zeros(n_clusters, q);
sigma2_hat = 1;
D_hat = eye(q);

cntclusters = zeros(n_clusters, 1);
maps2ind = false(n_obs, n_clusters);

for cluster_id = 1 : n_clusters
    indices_i = (tids == clusters(cluster_id));
    cntclusters(clusters(cluster_id)) = sum(indices_i == true);
    maps2ind(:, clusters(cluster_id)) = indices_i;
end

gll_history = zeros(max_iterations, 1);
b_hat_history = cell(max_iterations,1);
lf_history = cell(max_iterations, 1);
for iteration = 1 : max_iterations
    y_star = zeros(size(y));
    for cluster_id = 1 : n_clusters
        indices_i = maps2ind(:, clusters(cluster_id));
        b_hat_i = b_hat_df(clusters(cluster_id), :)';
        y_star_i = y(indices_i, :) - Z(indices_i, :) * b_hat_i;
        y_star(indices_i) = y_star_i;
    end
    lf = fitrlinear(X, y_star, 'Learner', 'leastsquares');
    lf_history{iteration} = lf;
    f_hat = lf.predict(X);
    
    sigma2_hat_sum = 0;
    D_hat_sum = 0;
    for cluster_id = 1 : n_clusters
        indices_i = maps2ind(:, clusters(cluster_id));
        n_i = cntclusters(clusters(cluster_id));
        f_hat_i = f_hat(indices_i);
        y_i = y(indices_i, :);
        Z_i = Z(indices_i, :);
        V_hat_i = Z_i * D_hat * Z_i' + sigma2_hat * eye(n_i);
        V_hat_inv_i =  V_hat_i \  eye(n_i);
        b_hat_i = D_hat * Z_i' * V_hat_inv_i * (y_i - f_hat_i);
        eps_hat_i = y_i - f_hat_i - Z_i * b_hat_i;
        b_hat_df(clusters(cluster_id), :) = b_hat_i';
        sigma2_hat_sum = sigma2_hat_sum + eps_hat_i' * eps_hat_i + sigma2_hat * (n_i - sigma2_hat * trace(V_hat_inv_i));
        D_hat_sum = D_hat_sum + b_hat_i * b_hat_i' + (D_hat - D_hat * Z_i' * V_hat_inv_i * Z_i * D_hat);
    end
    sigma2_hat = 1 ./ n_obs * sigma2_hat_sum;
    D_hat = 1 ./ n_clusters * D_hat_sum;
    b_hat_history{iteration} = b_hat_df;
    gll = 0;
    for cluster_id = 1 : n_clusters
        indices_i = maps2ind(:, clusters(cluster_id));
        n_i = cntclusters(clusters(cluster_id));
        y_i = y(indices_i, :);
        Z_i = Z(indices_i, :);
        I_i = eye(n_i);
        f_hat_i = f_hat(indices_i);
        R_hat_i = sigma2_hat * I_i;
        b_hat_i = b_hat_df(clusters(cluster_id), :)';
        gll = gll + (y_i - f_hat_i - Z_i * b_hat_i)' / R_hat_i * (y_i - f_hat_i - Z_i * b_hat_i) + b_hat_i' / D_hat * b_hat_i + det(D_hat) + det(R_hat_i);
    end
    gll_history(iteration) = gll;
end


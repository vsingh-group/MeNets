function [y_hat_fix, y_hat] = em_lmecnn_predict(lf, b_hat, X, Z, clusters, tids)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
y_hat_fix = lf.predict(X);
y_hat = y_hat_fix;
nclusters = length(clusters);

for cluster_id = 1 : nclusters
    indices_i = (tids == clusters(cluster_id));
    if (sum(indices_i == true) == 0)
        continue
    end
    b_i = b_hat(clusters(cluster_id), :)';
    Z_i = Z(indices_i, :);
    y_hat(indices_i) = y_hat(indices_i) + Z_i * b_i;
end

end


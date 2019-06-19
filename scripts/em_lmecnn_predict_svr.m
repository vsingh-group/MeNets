function [y_hat_fix, y_hat] = em_lmecnn_predict_svr(lf, b_hat, X, Z, clusters, X_train, trainIDs)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
y_hat_fix = lf.predict(X);
y_hat = y_hat_fix;
nclusters = length(clusters);

nsamples = size(X_train, 1);
b_svr = zeros(nsamples, 1);
for i = 1 : nsamples
    b_svr(i) = b_hat(trainIDs(i), :)';
end

blf = fitrlinear(X_train, b_svr, 'Learner', 'leastsquares');

testnsamples = size(X, 1);
for i = 1 : testnsamples
    b_i = blf.predict(X(i, :));
    y_hat(i) = y_hat(i) + Z(i, :) * b_i;
end

end


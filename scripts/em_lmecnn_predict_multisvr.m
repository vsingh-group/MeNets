function [y_hat_fix, y_hat] = em_lmecnn_predict_multisvr(lf, b_hat, X, Z, X_train, trainIDs)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
y_hat_fix = lf.predict(X);
y_hat = y_hat_fix;

nsamples = size(X_train, 1);
msamples = size(Z, 2);
b_svr = zeros(nsamples, msamples);
for i = 1 : nsamples
    b_svr(i, :) = b_hat(trainIDs(i), :);
end

blfs = cell(msamples);
for i = 1 : msamples
    blfs{i} = fitrlinear(X_train, b_svr(:, i), 'Learner', 'leastsquares');
end

testnsamples = size(X, 1);
b_i = zeros(msamples, 1);
for i = 1 : testnsamples
    for j = 1 : msamples
        b_i(j) = blfs{j}.predict(X(i, :));
    end
    y_hat(i) = y_hat(i) + Z(i, :) * b_i;
end

end


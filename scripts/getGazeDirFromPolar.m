function [gazeDir] = getGazeDirFromPolar(leftTheta)
%   predicted Gaze direction for test dataset
%   theta, phi to gaze direction
    ntemp = size(leftTheta, 1);
    gazeDir = zeros(ntemp, 3);
    for i = 1 : ntemp
        gazeDir(i, 1) = (-1) * cos(leftTheta(i, 1)) * sin(leftTheta(i, 2));
        gazeDir(i, 2) = (-1) * sin(leftTheta(i, 1));
        gazeDir(i, 3) = (-1) * cos(leftTheta(i, 1)) * cos(leftTheta(i, 2));
        gazeDir(i, :) = gazeDir(i, :) ./ norm(gazeDir(i, :));
    end
end
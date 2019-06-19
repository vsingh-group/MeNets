function [accu] = GetMeanAccuracy(predictedGaze, groundtruthGaze)
%   Mean accuracy for test dataset
%   angle between two unit vector
    accu = 0;
    [ntemp, mtemp] = size(predictedGaze);
    for i = 1 : ntemp
        gtgaze = groundtruthGaze(i, :) ./ norm(groundtruthGaze(i, :));
        angleVal = sum(predictedGaze(i, :) .* gtgaze);
        if (angleVal > 1)
            angleVal = 1;
        else if (angleVal < -1)
                angleVal = -1;
            end
        end
        accu = accu + (acos(angleVal) * 180)/pi;
    end
    accu = accu / ntemp;
end


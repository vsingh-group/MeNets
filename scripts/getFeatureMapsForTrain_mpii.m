function [featureMat, trainIDs, trainDir, meanAccu] = getFeatureMapsForTrain_mpii(net, iperson, npersons, trainDataset, trainLabels)
%   Get feature matrix from train dataset
%   
    meanAccu = [];
    featureMat = [];
    trainIDs = [];
    icnt = 0;
    nsamples = size(trainLabels, 2);
    %nlcnt = ceil(9 * nsamples / 10);
    nlcnt = nsamples;
    startInd = 0;
    trainDir = trainLabels(1:2, :)';
    if iperson == 13
        for i = 0 : npersons - 1
            if i == iperson 
                icnt = icnt + 1;
                continue;
            end
            for j = 0 : 2
                net.blobs('data').set_data(trainDataset(:, :, 1, startInd + 1:startInd + 1000));
                net.forward_prefilled();
                features = net.blobs('cat').get_data();
                accu = net.blobs('loss').get_data();
                meanAccu = [meanAccu; accu];
                featureMat = [featureMat; features'];
                startInd = startInd + 1000;
                if startInd > nlcnt
                    return;
                end
            end
            tempIDs = zeros(3000, 1);
            tempIDs(1:2:end) = ones(1500, 1) * icnt;
            tempIDs(2:2:end) = ones(1500, 1) * (icnt + npersons);
            icnt = icnt + 1;
            trainIDs = [trainIDs; tempIDs];
        end
    else
        for i = 0 : 12
            if i == iperson 
                icnt = icnt + 1;
                continue;
            end
            for j = 0 : 2
                net.blobs('data').set_data(trainDataset(:, :, 1, startInd + 1:startInd + 1000));
                net.forward_prefilled();
                features = net.blobs('cat').get_data();
                accu = net.blobs('loss').get_data();
                meanAccu = [meanAccu; accu];
                featureMat = [featureMat; features'];
                startInd = startInd + 1000;
                if startInd > nlcnt
                    return;
                end
            end
            tempIDs = zeros(3000, 1);
            tempIDs(1:2:end) = ones(1500, 1) * icnt;
            tempIDs(2:2:end) = ones(1500, 1) * (icnt + npersons);
            icnt = icnt + 1;
            trainIDs = [trainIDs; tempIDs];
        end
        for j = 0 : 1
            net.blobs('data').set_data(trainDataset(:, :, 1, startInd + 1:startInd + 1000));
            net.forward_prefilled();
            features = net.blobs('cat').get_data();
            accu = net.blobs('loss').get_data();
            meanAccu = [meanAccu; accu];
            featureMat = [featureMat; features'];
            startInd = startInd + 1000;
            if startInd > nlcnt
               return;
            end
        end
        net.blobs('data').set_data(trainDataset(:, :, 1, startInd -3 :startInd + 996));
        net.forward_prefilled();
        features = net.blobs('cat').get_data();
        accu = net.blobs('loss').get_data();
        meanAccu = [meanAccu; accu];
        featureMat = [featureMat; features(:, 1:996)']; %
        tempIDs = zeros(2996, 1);
        tempIDs(1:2:end) = ones(1498, 1) * icnt;
        tempIDs(2:2:end) = ones(1498, 1) * (icnt + npersons);
        icnt = icnt + 1;
        trainIDs = [trainIDs; tempIDs];
        startInd = startInd + 996;
        if (iperson ~= 14)
            for j = 0 : 2
                net.blobs('data').set_data(trainDataset(:, :, 1, startInd + 1:startInd + 1000));
                net.forward_prefilled();
                features = net.blobs('cat').get_data();
                accu = net.blobs('loss').get_data();
                meanAccu = [meanAccu; accu];
                featureMat = [featureMat; features'];
                startInd = startInd + 1000;
                if startInd > nlcnt
                    return;
                end
            end
            tempIDs = zeros(3000, 1);
            tempIDs(1:2:end) = ones(1500, 1) * icnt;
            tempIDs(2:2:end) = ones(1500, 1) * (icnt + npersons);
            trainIDs = [trainIDs; tempIDs];
        end
    end
end


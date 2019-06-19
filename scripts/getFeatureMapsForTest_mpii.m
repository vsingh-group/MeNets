function [featureMat, testIDs, testDir, meanAccu] = getFeatureMapsForTest_mpii(net, iperson, npersons, testDataset, testLabels)
%   Get feature matrix from train dataset
%   
    tic;
    meanAccu = [];
    featureMat = [];
    nsamples = size(testLabels, 2);
    testIDs = zeros(nsamples, 1);
    startInd = 0;
    if (iperson ~= 13)
        for j = 0 : 2
            net.blobs('data').set_data(testDataset(:, :, 1, startInd + 1:startInd + 1000));
            net.forward_prefilled();
            features = net.blobs('cat').get_data();
            accu = net.blobs('accuracy').get_data();
            meanAccu = [meanAccu; accu];
            featureMat = [featureMat; features'];
            startInd = startInd + 1000;
        end
    else
        for j = 0 : 1
            net.blobs('data').set_data(testDataset(:, :, 1, startInd + 1:startInd + 1000));
            net.forward_prefilled();
            features = net.blobs('cat').get_data();
            accu = net.blobs('accuracy').get_data();
            meanAccu = [meanAccu; accu];
            featureMat = [featureMat; features'];
            startInd = startInd + 1000;
        end
        net.blobs('data').set_data(testDataset(:, :, 1, startInd -3 :end));
        net.forward_prefilled();
        features = net.blobs('cat').get_data();
        accu = net.blobs('accuracy').get_data();
        meanAccu = [meanAccu; accu];
        featureMat = [featureMat; features(:, 5:1000)'];
    end
    testIDs(1:2:end) = ones(nsamples / 2, 1) * iperson;
    testIDs(2:2:end) = ones(nsamples / 2, 1) * (iperson + npersons);
    testDir = testLabels(1:2, :)';
    toc;
end


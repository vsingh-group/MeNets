clc;

pid = 1;
npersons = 7;
load(sprintf('%s%02dc.mat', 'Real_person', pid));
featureTTMat = [featureTrainMat; featureTestMat];
trainTTDir = [trainDir; testDir];
trainTTIDs = [trainIDs; testIDs];

max_iterations = 10;
for looid = 0 : 6
    mark = (trainTTIDs == looid) | (trainTTIDs == looid + npersons);
    testIDs = trainTTIDs(mark) + 1;
    trainIDs = trainTTIDs(~mark) + 1;
    featureTestMat = featureTTMat(mark, :);
    featureTrainMat = featureTTMat(~mark, :);
    testDir = trainTTDir(mark, :);
    trainDir = trainTTDir(~mark, :);
    
    clusters = npersons;
    uids = 1:clusters;
    Z_train = ones(size(trainIDs, 1), 1);
    Z_test = ones(size(testIDs, 1), 1);
    %Z_train = featureTrainMat;
    %Z_test = featureTestMat;


    [phi_lf, phi_b_hat_history, phi_gll_history] = em_lmecnn(featureTrainMat, Z_train, uids, trainDir(:, 1), trainIDs, max_iterations);
    %[phi_fix, phi_lme] = em_lmecnn_predict(phi_lf{max_iterations}, phi_b_hat_history{max_iterations}, featureTestMat, Z_test, uids, testIDs);
    [phi_fix, phi_lme] = em_lmecnn_predict_multisvr(phi_lf{max_iterations}, phi_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);

    [theta_lf, theta_b_hat_history, theta_gll_history] = em_lmecnn(featureTrainMat, Z_train, uids, trainDir(:, 2), trainIDs, max_iterations);
    %[theta_fix, theta_lme] = em_lmecnn_predict(theta_lf{max_iterations}, theta_b_hat_history{max_iterations}, featureTestMat, Z_test, uids, testIDs);
    [theta_fix, theta_lme] = em_lmecnn_predict_multisvr(theta_lf{max_iterations}, theta_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);

    pred_fix = [phi_fix, theta_fix];
    pred_lme = [phi_lme, theta_lme];

    pred_fix_gaze_dir = getGazeDirFromPolar(pred_fix);
    pred_lme_gaze_dir = getGazeDirFromPolar(pred_lme);
    gth_gaze_dir = getGazeDirFromPolar(testDir);

    fixAccu = GetMeanAccuracy(pred_fix_gaze_dir, gth_gaze_dir);
    lmeAccu = GetMeanAccuracy(pred_lme_gaze_dir, gth_gaze_dir);
    
    %get initial one
    max_iterations = 1;
    [phi_fix, phi_lme] = em_lmecnn_predict_multisvr(phi_lf{max_iterations}, phi_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);
    [theta_fix, theta_lme] = em_lmecnn_predict_multisvr(theta_lf{max_iterations}, theta_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);
    pred_fix = [phi_fix, theta_fix];
    pred_lme = [phi_lme, theta_lme];

    pred_fix_gaze_dir = getGazeDirFromPolar(pred_fix);
    pred_lme_gaze_dir = getGazeDirFromPolar(pred_lme);
    
    fixAccu_1 = GetMeanAccuracy(pred_fix_gaze_dir, gth_gaze_dir);
    lmeAccu_1 = GetMeanAccuracy(pred_lme_gaze_dir, gth_gaze_dir);
    
    foldpre = sprintf('%s%d%s', 'power_real_person_loo', looid, '_pred_test.mat');
    save(foldpre, 'phi_lf', 'phi_b_hat_history', 'phi_gll_history','theta_lf', 'theta_b_hat_history', 'theta_gll_history', 'fixAccu', 'lmeAccu', 'fixAccu_1', 'lmeAccu_1');
end
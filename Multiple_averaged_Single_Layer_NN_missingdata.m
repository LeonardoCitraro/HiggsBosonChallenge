%%
%%=====================================================
%                HIGGS BOSON CHALLENGE 
%======================================================
%   University of Southampton
%   Msc Systems and Signal Processing
%   COMP6208 - Advanced Machine Learning
%   
%   Citraro L., Perodou A., Roullier B., Iyengar A.
%   Start: 08.03.2015 
%   End: 22.04.2015
%======================================================

clear all
clc

%% ========================================================================
% Constants
% =========================================================================
EVENTID     = 1;
FEATURES    = 2:31;
WEIGHTS     = 32;
LABELS      = 33;
SIGNAL      = 1;
BACKGROUND  = 0;

%% ========================================================================
% Toolbox
% =========================================================================
addpath(genpath('rasmusbergpalm-DeepLearnToolbox_modified'));


%% ========================================================================
% Load dataset, split, normalization and missing values
% =========================================================================
load higgs_training.mat
D = higgs_training;

D(D == -999) = NaN; % avoid missing data from normalization

mean_all = nanmean(D);
std_all = nanstd(D);

D(isnan(D)) = 0; % put missing data to lower importance

k           = 5; % number of network to create
n_train     = 152000; % gross training size
n_val       = 95000; % gross validation size

[train_sets, val_sets] = stratified_bootstrapping_k_subset(D, k, n_train, n_val);
[Nv, ~] = size(val_sets);
[Nt, ~] = size(train_sets);

% normalization
for j=1:k
    Dt = train_sets(:,:,j);
    Dv = val_sets(:,:,j);
    for i=FEATURES
        Dt(:, i)=bsxfun(@minus,Dt(:, i),mean_all(i));
        Dt(:, i)=bsxfun(@rdivide,Dt(:, i), std_all(i));
        Dv(:, i)=bsxfun(@minus,Dv(:, i),mean_all(i));
        Dv(:, i)=bsxfun(@rdivide,Dv(:, i), std_all(i));    
    end   
    train_sets(:,:,j) = Dt(:,:);
    val_sets(:,:,j) = Dv(:,:);
end

% get integer mini-batch sizes
K=1:Nt;
dd = K(rem(Nt,K)==0) % possibles batchsizes, multiple of Nt

%% ========================================================================
% Training process Neural Network
% =========================================================================
clc
clear nn opts nn_val nn_train
close all
tic

verbose     = 0; % plot infos
batch       = 20; % 154 mini batch size in samples

AMS_th_k_t  = [];
PERF_th_k_t = [];

AMS_th_k_v  = [];
PERF_th_k_v = [];

nn_bests = []; % store best models

for i=1:k % for every Network

    % ---------------------------------------------
    % setup data, labels and weights
    % ---------------------------------------------
    labels      = train_sets(:, LABELS, i);  % 0/1 
    labels_v    = val_sets(:, LABELS, i);  % 0/1 
    data        = train_sets(:, FEATURES, i);   
    data_v      = val_sets(:, FEATURES, i);
    weights     = train_sets(:, WEIGHTS, i);
    weights_v   = val_sets(:, WEIGHTS, i);
    
    
    % ---------------------------------------------
	% Train NN
    % ---------------------------------------------
    %rand('state',0)
    nn = nnsetup([30 80 1]);
    nn.activation_function  = 'sigm';
    nn.output               = 'sigm';
    nn.weightPenaltyL2      = 0;
    nn.normalize_input      = 0;
    %nn.learningRate         = 0.5; % low batch size
    nn.learningRate         = 1.5; %high batch size
    nn.dropoutFraction      = 0;
    nn.momentum             = 0.5;
    opts.plot               = 1; 
    opts.numepochs          = 50;
    opts.batchsize          = dd(batch);
    opts.validation         = 1;
    [nn, L] = nntrain(nn, data, labels, opts, data_v, labels_v, weights, weights_v);
    
    % inside nn there are all the wights that maximise the validation AMS stored during training process
    nn_bests{i} = nn; % store best models
end



%% ========================================================================
% Averaging process and AMS calculation on the test set
% =========================================================================

% ---------------------------------------------
% Load and normalize test set
% ---------------------------------------------
clear data data_v Dt Dv higgs_training labels labels_v weights weights_v test_data test_weights_labels
clear float_prediction_constrained_result
load higgs_private_leaderboard.mat

higgs_private_leaderboard(higgs_private_leaderboard==-999)=0;

test_data = bsxfun(@minus,higgs_private_leaderboard(:, FEATURES), mean_all(FEATURES));
test_data = bsxfun(@rdivide, test_data, std_all(FEATURES));  
test_weights_labels = higgs_private_leaderboard(:, [WEIGHTS, LABELS]);

clear higgs_private_leaderboard

% ---------------------------------------------
% Get prediction for each NN
% ---------------------------------------------
float_prediction_constrained_result = [];
ths = [];

for i=1:k % for every NN
   
    % get prediction
    nn_bests{i}.nn_best.testing = 1;    
    nn_result = nnff(nn_bests{i}.nn_best, test_data, zeros(size(test_data,1), nn.size(end)));  
    nn_bests{i}.nn_best.testing = 0;
    
    float_prediction_result = nn_result.a{end};  
    
    clear nn_result
    
    % normalize output NN
    mean_f_p_v = mean(float_prediction_result);
    std_f_p_v = std(float_prediction_result);
    temp = (float_prediction_result-mean_f_p_v)/std_f_p_v;
    float_prediction_constrained_result = [float_prediction_constrained_result, temp];
    
    % get prediction and AMS
    prediction_test = temp > (nn_bests{i}.nn_best.TH_val);
    [AMS, ~, ~, ~] = AMS_metric(prediction_test, test_weights_labels, 0);
    disp(['AMS val NN-' num2str(i) ' : ' num2str(AMS)])
    
    ths = [ths, nn_bests{i}.nn_best.TH_val]; % store thresholds
end

clear test_data

% ---------------------------------------------
% Get averaged prediction and final AMS
% ---------------------------------------------
prediction_test = mean(float_prediction_constrained_result, 2) > mean(ths);
[AMS, ~, ~, ~] = AMS_metric(prediction_test, test_weights_labels, 0);


disp(['================================================================'])
disp(['Final      AMS on private leaderboard: ' num2str(AMS) ' !!!!!!!!!!!!!!!!!!'])
disp(['================================================================'])
beep

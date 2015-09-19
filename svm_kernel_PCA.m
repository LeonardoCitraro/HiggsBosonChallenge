%%=====================================================
%                HIGGS BOSON CHALLENGE 
%======================================================
%   University of Southampton
%   Msc Systems and Signal Processing
%   COMP6208 - Advanced Machine Learning
%   
%   Citraro L., Perodou A., Roullier B., Iyengar A.
%   Start: 13.02.2015 
%   End: 21.04.2015
%======================================================

clear all
clc
%% =========================================================================
% Setup toolbox
% =========================================================================
addpath(genpath('vlfeat-0.9.19-bin'));
vl_setup

%% =========================================================================
% Read dataset and apply PCA
% =========================================================================
load training_set.mat

% Constants
FEATURES    = 2:31;
WEIGHTS     = 32;
LABELS      = 33;
[pc,score,latent,tsquare] = princomp(training_set(:, FEATURES)); % exctract principal components

format long
score_PCA = cumsum(latent)./sum(latent) % get most relevant principal components
format short

n_eigenvectors = 30; % choosen most relevant principal components
retained_eigenvectors = pc(:, 1:n_eigenvectors);

% projection to lower dimension (training set)
D = training_set(:, FEATURES)*retained_eigenvectors;
% integrate weights and labels not modified, eventid excluded
training_set = [D, training_set(:, [WEIGHTS, LABELS])];

% projection to lower dimension (test set)
D = test_set(:, FEATURES)*retained_eigenvectors;
% integrate weights and labels not modified, eventid excluded
test_set = [D, test_set(:, [WEIGHTS, LABELS])];

%% =========================================================================
% Reduced features by pca
% =========================================================================
FEATURES    = 1:n_eigenvectors;
WEIGHTS     = n_eigenvectors+1;
LABELS      = n_eigenvectors+2;
SIGNAL      = 1;
BACKGROUND  = 0;

%% ========================================================================
% Split k-folds and normalization
% =========================================================================

mean_all = nanmean(D);
std_all = nanstd(D);

k           = 4; % number of fold for k-folds cross-validation
verbose     = 0; % display infos
[cv_train_sets, cv_val_sets] = split_dataset_k_folds(D, k, verbose);
[Nv, ~] = size(cv_val_sets);
[Nt, ~] = size(cv_train_sets);

% normalize every feature
for j=1:k
    Dt = cv_train_sets(:,:,j);
    Dv = cv_val_sets(:,:,j);
    for i=FEATURES
        Dt(:, i)=bsxfun(@minus,Dt(:, i),mean_all(i));
        Dt(:, i)=bsxfun(@rdivide,Dt(:, i), std_all(i));
        Dv(:, i)=bsxfun(@minus,Dv(:, i),mean_all(i));
        Dv(:, i)=bsxfun(@rdivide,Dv(:, i), std_all(i));    
    end   
    cv_train_sets(:,:,j) = Dt(:,:);
    cv_val_sets(:,:,j) = Dv(:,:);
end


%% ========================================================================
% Training process
% =========================================================================

clc
tic
%--------------------------------------------------
% General config
lambda      = [0.000001 0.00001 0.0001 0.0005 0.001 0.002 0.005 0.01 0.1 1 10];
k           = 4;
verbose     = 0;
kernel_en   = 1;

%--------------------------------------------------
% Kernel config
%hom.kernel  = 'kchi2';
hom.kernel  = 'kinters';
%hom.kernel  = 'kjs';
hom.order   = 1; 
%hom.window  = 'uniform';
hom.window  = 'rectangular';
hom.gamma   = 1;
%--------------------------------------------------
% SVM config
struct_values = {        
    'Epsilon';              1e-3;
                            'Verbose';              
	'MaxNumIterations';     100000000;
    'BiasMultiplier';       1;
	'BiasLearningRate';     0.5; %sgd only
	'Loss';                 'HINGE';
	'Solver';               'SGD';
	'Bias';                 0;
	};
svm = struct('config',struct_values);
%--------------------------------------------------    

AMS_lambda          = []; % store best AMS vs. regularization
optimal_th_lambda   = []; % store best threshold vs regularization
w                   = []; % weights hyperplane for k-folds
b                   = []; % k biasses
infos               = []; % store infos

for l=1:length(lambda) % for each regularizer
    AMS_th_k = [];
    PERF_th_k = [];

    for i=1:k % for each fold

        %----------------------------------
        % Apply kernel training subset
        %----------------------------------
        if kernel_en
            training_subset = vl_svmdataset(cv_train_sets(:, FEATURES, i)', 'homkermap', hom);
        else
            training_subset = cv_train_sets(:, FEATURES, i)';
        end

        % using label +1 -1 produce much better result than +1 0 !!!!!!
        labels = (cv_train_sets(:, LABELS, i)*2)-ones(Nt, 1);

        %----------------------------------
        % Train SVM
        %----------------------------------
        [w(:, i, l) b(i, l) info] = vl_svmtrain(training_subset, labels, lambda(l), svm.config);
        infos = [infos, info];

        %----------------------------------
        % Apply kernel on validation set
        %----------------------------------
        if kernel_en
            val_event_kernel = vl_homkermap(cv_val_sets(:, FEATURES, i)', hom.order, hom.kernel);
        else
            val_event_kernel = cv_val_sets(:, FEATURES, i)';
        end
        
        %----------------------------------
        % Normalize output SVM
        %----------------------------------
        float_prediction = val_event_kernel'*w(:, i, l) + b(i, l);  
        mean_f_p = mean(float_prediction);
        std_f_p = std(float_prediction);
        float_prediction_constrained = (float_prediction-mean_f_p)/std_f_p;

        %----------------------------------
        % Sweep output threshold and calualte AMS plus accuracy
        %----------------------------------
        AMS_th = [];
        PERF_th = [];
        th = linspace(-5, 4, 200);
        for j=th
            % Compute AMS with specific threshold
            prediction = float_prediction_constrained > j;
            [AMS, ~, ~, ~] = AMS_metric(prediction, cv_val_sets(:, [WEIGHTS, LABELS], i), verbose);
            AMS_th = [AMS_th, AMS];

            % Compute performance of classification 0-1
            PERF_th = [PERF_th, sum(prediction==cv_val_sets(:, LABELS, i))/length(prediction)];
        end

        % Storage for k-folds
        AMS_th_k = [AMS_th_k; AMS_th];
        PERF_th_k = [PERF_th_k; PERF_th];
    end

    AMS_th_k_mean = mean(AMS_th_k);
    AMS_th_k_std = std(AMS_th_k);
    [AMS_th_k_mean_max, idx] = max(AMS_th_k_mean);
    optimal_th = th(idx);
    
    AMS_lambda = [AMS_lambda, AMS_th_k_mean_max];
    optimal_th_lambda = [optimal_th_lambda, optimal_th];
end

%% ========================================================================
% Plot useful graphs
% =========================================================================
subplot(1, 2, 1),
semilogx(lambda, AMS_lambda, 'linewidth', 3)
%plot(lambda, AMS_lambda, 'linewidth', 3)
title('AMS vs. regularization', 'Fontsize', 14)
xlabel('Lambda', 'Fontsize', 12)
ylabel('AMS averaged over k-folds', 'Fontsize', 12)
grid
axis([min(lambda) max(lambda) 1 3])
subplot(1, 2, 2),
semilogx(lambda, optimal_th_lambda, 'linewidth', 3)
%plot(lambda, optimal_th_lambda, 'linewidth', 3)
title('Threshold vs. regularization', 'Fontsize', 14)
xlabel('Lambda', 'Fontsize', 12)
ylabel('Threshold', 'Fontsize', 12)
grid
axis([min(lambda) max(lambda) 0 1.3])
toc

% Display last CV AMS vs. threshold
AMS__th__k = AMS_th_k';
plot(th, AMS__th__k,'DisplayName','AMS__th__k', 'linewidth', 1), 
hold on
plot(th, AMS_th_k_mean, 'k', 'linewidth', 3,'DisplayName','AMS mean')
plot(th, AMS_th_k_std, 'b', 'linewidth', 3,'DisplayName','AMS std.dev.')
grid
title('AMS vs. threshold for k-folds CV', 'Fontsize', 16)
xlabel('threshold', 'Fontsize', 14)
ylabel('AMS', 'Fontsize', 14)




%% ========================================================================
% Training process for the test set
% =========================================================================
clc
tic

%-------------------------------------
% Set optimal paramters
%-------------------------------------
lambda      = 1e-5; % best lambda
threshold   = 0.8698; % best threshold
verbose     = 0;

[Nts, ~] = size(training_set);

Dtr = training_set;
Dte = test_set;

%-------------------------------------
% Normalize features
%-------------------------------------
for i=FEATURES
    mean_tr = mean(Dtr(:, i));
    std_tr = std(Dtr(:, i));
    Dtr(:, i) = (Dtr(:, i)-mean_tr)/std_tr;
    Dte(:, i) = (Dte(:, i)-mean_tr)/std_tr;
end

%-------------------------------------
% Apply kernel whole training dataset
%-------------------------------------
if kernel_en
    training_subset = vl_svmdataset(Dtr(:, FEATURES)', 'homkermap', hom);
else
    training_subset = Dtr(:, FEATURES)';
end
% using label +1 -1 produce much better result than +1 0 !!!!!!
labels = (Dtr(:, LABELS)*2)-ones(Nts, 1);

%-------------------------------------
% Train SVM
%-------------------------------------
[w b info] = vl_svmtrain(training_subset, labels, lambda, svm.config);


%-------------------------------------
% Apply kernel whole test dataset
%-------------------------------------
if kernel_en
    val_event_kernel = vl_homkermap(Dte(:, FEATURES)', hom.order, hom.kernel);
else
    val_event_kernel = Dte(:, FEATURES)';
end

%-------------------------------------
% Normalize output SVM
%-------------------------------------
float_prediction = val_event_kernel'*w + b;    
mean_f_p = mean(float_prediction);
std_f_p = std(float_prediction);
float_prediction_constrained = (float_prediction-mean_f_p)/std_f_p;

%----------------------------------
% Calualte AMS plus accuracy
%----------------------------------
prediction = float_prediction_constrained > threshold;
[AMS, ~, ~, ~] = AMS_metric(prediction, Dte(:, [WEIGHTS, LABELS]), verbose)

toc

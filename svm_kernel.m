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
% Setup toolbox SVM
% =========================================================================
addpath(genpath('vlfeat-0.9.19-bin'));
vl_setup

%% ========================================================================
% Split k-folds and normalization
% =========================================================================
load training_set.mat
D = training_set;

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
kernel      = 'homo'; % enable homogeneous kernel type
%kernel      = 'none'; % no kernel
%kernel      = 'rbf'; % rbf kernel, not applicable!
%--------------------------------------------------
% Kernel config
hom.kernel  = 'kchi2';
%hom.kernel  = 'kinters';
%hom.kernel  = 'kjs';
hom.order   = 1; 
%hom.window  = 'uniform';
hom.window  = 'rectangular';
hom.gamma   = 1;
%--------------------------------------------------
% SVM config
struct_values = {        
    'Epsilon';              1e-3; %slack margin
                            'Verbose';              
	'MaxNumIterations';     100000000;
    'BiasMultiplier';       1;
	'BiasLearningRate';     0.5; %sgd only
	'Loss';                 'HINGE'; %loss function HINGE=fastest
	'Solver';               'SGD';
	'Bias';                 0;
	};
svm = struct('config',struct_values);
%--------------------------------------------------    

AMS_lambda          = []; % store AMS vs lambda
optimal_th_lambda   = []; % store optimal threshold
w                   = []; % weights hyperplane for each fold
b                   = []; % k biasses
infos               = []; % store infos

for l=1:length(lambda) % for each regularizer
    AMS_th_k = []; % store AMS vs threshold
    PERF_th_k = []; % store accuracy of classification vs threshold

    for i=1:k % for each fold

        %-------------------------------------
        % Apply kernel to training set
        %-------------------------------------
        switch kernel
            case 'homo'
                training_subset = vl_svmdataset(cv_train_sets(:, FEATURES, i)', 'homkermap', hom);
            case 'rbf'
                % Out of memory. Type HELP MEMORY for your options.!!
                %Source: http://www.kernel-methods.net/matlab/kernels/rbf.m
                %{
                XX = cv_train_sets(:, FEATURES, i); 
                for j=1:size(XX,1)
                     K(j,j)=1;
                     for k=1:i-1
                         K(j,k)=exp(-norm(XX(j,:)-XX(k,:))^2/sig^2);
                         K(k,j)=K(j,k);
                     end
                end
                %}
                % Out of memory. Type HELP MEMORY for your options.!!
                XX = cv_train_sets(:, FEATURES, i);
                n=size(XX,1);
                K=XX*XX'/sig^2;
                d=diag(K);
                K=K-ones(n,1)*d'/2;
                K=K-d*ones(1,n)/2;
                K=exp(K);
                training_subset = K;
            otherwise
                training_subset = cv_train_sets(:, FEATURES, i)';
        end
        % using label +1 -1 produce much better result than +1 0 !!!!!!
        labels = (cv_train_sets(:, LABELS, i)*2)-ones(Nt, 1);

        %-------------------------------------
        % Train SVM
        %-------------------------------------
        [w(:, i, l) b(i, l) info] = vl_svmtrain(training_subset, labels, lambda(l), svm.config);
        infos = [infos, info];

        %-------------------------------------
        % Apply kernel to validation set
        %-------------------------------------
        switch kernel
            case 'homo'
                val_event_kernel = vl_homkermap(cv_val_sets(:, FEATURES, i)', hom.order, hom.kernel);            
            case 'rbf'
                % Out of memory. Type HELP MEMORY for your options.!!
                %Source: http://www.kernel-methods.net/matlab/kernels/rbf.m
                %{
                XX = cv_val_sets(:, FEATURES, i); 
                for j=1:size(XX,1)
                     K(j,j)=1;
                     for k=1:i-1
                         K(j,k)=exp(-norm(XX(j,:)-XX(k,:))^2/sig^2);
                         K(k,j)=K(j,k);
                     end
                end
                %}
                % Out of memory. Type HELP MEMORY for your options.!!
                XX = cv_val_sets(:, FEATURES, i);
                n=size(XX,1);
                K=XX*XX'/sig^2;
                d=diag(K);
                K=K-ones(n,1)*d'/2;
                K=K-d*ones(1,n)/2;
                K=exp(K);
                val_event_kernel = K;
            otherwise
                val_event_kernel = cv_val_sets(:, FEATURES, i)';
        end
        
        %-------------------------------------
        % Normalize the output of the SVM
        %-------------------------------------
        float_prediction = val_event_kernel'*w(:, i, l) + b(i, l);  
        mean_f_p = mean(float_prediction);
        std_f_p = std(float_prediction);
        float_prediction_constrained = (float_prediction-mean_f_p)/std_f_p;

        %-------------------------------------
        % Sweep output threshold and calculate accuracy plus AMS
        %-------------------------------------
        AMS_th = [];
        PERF_th = [];
        th = linspace(-5, 4, 200); % threshold palette
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
% Plot some useful graphs
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
figure,
AMS__th__k = AMS_th_k';
plot(th, AMS__th__k,'DisplayName','AMS__th__k', 'linewidth', 1), 
hold on
plot(th, AMS_th_k_mean, 'k', 'linewidth', 3,'DisplayName','AMS mean')
plot(th, AMS_th_k_std, 'b', 'linewidth', 3,'DisplayName','AMS std.dev.')
grid
title('AMS vs. threshold for k-folds CV', 'Fontsize', 16)
xlabel('Threshold', 'Fontsize', 16)
ylabel('AMS', 'Fontsize', 16)





%% ========================================================================
% Training process, Calculate AMS on the test set using the entire training set
% =========================================================================
load training_set.mat
load higgs_private_leaderboard.mat

kernel_en = 1;
%--------------------------------------------------
% Kernel config
hom.kernel  = 'kchi2';
%hom.kernel  = 'kinters';
%hom.kernel  = 'kjs';
hom.order   = 5; 
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

clc
tic
%-------------------------------------
% Set optimal parameters
%-------------------------------------
lambda      = 1e-5; % best lambda
threshold   = 0.8698; % best threshold
verbose     = 0;

[Nts, ~] = size(training_set);

Dtr = training_set;
Dte = higgs_private_leaderboard;

clear load training_set
clear higgs_private_leaderboard

%-------------------------------------
% normalization
%-------------------------------------
for i=FEATURES
    Dtr(:, i) = (Dtr(:, i)-mean_all(i))/std_all(i);
    Dte(:, i) = (Dte(:, i)-mean_all(i))/std_all(i);
end

%-------------------------------------
% Apply kernel whole training set
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

clear training_subset labels

%-------------------------------------
% Apply kernel to test set
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

%-------------------------------------
% Calculate prediction and AMS
%-------------------------------------
prediction = float_prediction_constrained > threshold;
[AMS, ~, ~, ~] = AMS_metric(prediction, Dte(:, [WEIGHTS, LABELS]), verbose)

toc
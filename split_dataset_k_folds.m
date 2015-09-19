%%=====================================================
%                HIGGS BOSON CHALLENGE 
%======================================================
%   University of Southampton
%   Msc Systems and Signal Processing
%   COMP6208 - Advanced Machine Learning
%   
%   Citraro L., Perodou A., Roullier B., Iyengar A.
%   Start: 12.02.2015 
%   End: 04.05.2015
%======================================================
%%
function [cv_train_sets, cv_val_sets] = split_dataset_k_folds(D, k, verbose)
%   split_dataset_k_folds:
%       Split the dataset D into k sub-set modifying the
%       weights and maintaining the number of signal
%       and background event in each subset (stratified k-folds).
%   inputs:
%       D: entire dataset [eventid, 30xfeatures, weights, labels]
%       k: number of folds k-fold crossvalidation
%       verbose: display information {on=1, off=0}
%   outputs:
%       cv_train_sets: training 3D array, [samples X features X k]
%       cv_val_sets: validation 3D array, [samples X features X k]

    [Ntot, M] = size(D);
    %{
    if M~=LABELS
        disp('Error: number of columns incorrect!')
    end
    %}
    
    WEIGHTS     = M-1;
    LABELS      = M;
    SIGNAL      = 1;
    BACKGROUND  = 0;
    WS          = 692;
    WB          = 411000;
    
    % Extract signals and backgrounds events
    S = D(any(D(:,LABELS)==SIGNAL, 2),:);
    B = D(any(D(:,LABELS)==BACKGROUND, 2),:);
    
    % Get number of samples
    [Ns, ~] = size(S);
    [Nb, ~] = size(B);
    
    % Compute sub-sub-fold size
    Ns_fold = floor(Ns/k);
    Nb_fold = floor(Nb/k);
    
    % Random permutation
    ps = randperm(Ns);
    pb = randperm(Nb);
    S = S(ps, :);
    B = B(pb, :);
    
    % Initialization of the 3D output arrays
    cv_train_sets = zeros((k-1)*(Ns_fold+Nb_fold), LABELS, k);
    cv_val_sets = zeros(Ns_fold+Nb_fold, LABELS, k);
    
    for i=1:k
        % Diagram splitting method
        % ----------------------------------------
        % |  1  |             k-1                | 
        % ----------------------------------------
        
        % get the first block (validation block)
        temp_s_val = S(1:Ns_fold, :);
        temp_b_val = B(1:Nb_fold, :);
        
        % Sum the weights
        ws = sum(temp_s_val(:, WEIGHTS));
        wb = sum(temp_b_val(:, WEIGHTS));
        
        % Modify the weights
        temp_s_val(:, WEIGHTS) = temp_s_val(:, WEIGHTS)*WS/ws;
        temp_b_val(:, WEIGHTS) = temp_b_val(:, WEIGHTS)*WB/wb;
        
        % Put in the same array
        temp_val = [temp_s_val; 
                    temp_b_val];
        
                
        % get the bigger block (training block)
        temp_s_train = S(Ns_fold+1:k*Ns_fold, :);
        temp_b_train = B(Nb_fold+1:k*Nb_fold, :);
        
        % Sum the weights
        ws = sum(temp_s_train(:, WEIGHTS));
        wb = sum(temp_b_train(:, WEIGHTS));
        
        % Modify the weights
        temp_s_train(:, WEIGHTS) = temp_s_train(:, WEIGHTS)*WS/ws;
        temp_b_train(:, WEIGHTS) = temp_b_train(:, WEIGHTS)*WB/wb;

        % Put in the same array
        temp_train = [temp_s_train;
                      temp_b_train];
        
        % Random permutation and storage
        ptt = randperm((k-1)*(Ns_fold+Nb_fold));   
        ptv = randperm(Ns_fold+Nb_fold); 
        cv_train_sets(:, :, i) = temp_train(ptt, :);
        cv_val_sets(:, :, i) = temp_val(ptv, :);                
 
        % Diagram splitting method
        % ----------------------------------------
        % |     |  1  |->                        | 
        % ----------------------------------------
        % Circle shifting the datasets and repeat
        S = circshift(S, [-Ns_fold,  0]);
        B = circshift(B, [-Nb_fold,  0]); 
    end
    
    if verbose
        disp('----------------------------------------------');
        disp('split_dataset_k_folds:');
        disp([sprintf('\t') 'samples : ', num2str(Ntot), 'x', num2str(M)]);
        
        disp([sprintf('\t') 'count signals : ', num2str(Ns)]);
        disp([sprintf('\t') 'count backgrounds : ', num2str(Nb)]);
        
        disp([sprintf('\t') 'k : ', num2str(k)]);
        disp([sprintf('\t') 'size folds : ', num2str(Ns_fold+Nb_fold)]);
        disp([sprintf('\t') 'size sub folds S : ', num2str(Ns_fold)]);
        disp([sprintf('\t') 'size sub folds B : ', num2str(Nb_fold)]);
        disp([sprintf('\t') 'dumped samples in S : ', num2str(Ns-k*Ns_fold)]);
        disp([sprintf('\t') 'dumped samples in B : ', num2str(Nb-k*Nb_fold)]);

        disp('----------------------------------------------');
    end
end
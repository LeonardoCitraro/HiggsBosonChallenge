%%=====================================================
%                HIGGS BOSON CHALLENGE 
%======================================================
%   University of Southampton
%   Msc Systems and Signal Processing
%   COMP6208 - Advanced Machine Learning
%   
%   Citraro L., Perodou A., Roullier B., Iyengar A.
%   Start: 17.03.2015 
%   End: 
%======================================================
%%
function [train_sets, val_sets] = stratified_bootstrapping_k_subset(D, k, n_train, n_val)
%   stratified_bootstrapping_k_subset:
%       Bootstrap the dataset in k training and validation
%       sets of n_train and n_val samples using a stratified
%       approach.
%   inputs:
%       D: entire dataset [eventid, 30xfeatures, weights, labels]
%       k: number of dataset to create
%       n_train, n_val: number of samples
%   outputs:
%       train_sets: training 3D array, [samples X features X k]
%       val_sets: validation 3D array, [samples X features X k]

    [Ntot, M] = size(D);
    
    WEIGHTS     = M-1;
    LABELS      = M;
    SIGNAL      = 1;
    BACKGROUND  = 0;
    WS          = 692; % sum of the weights signals
    WB          = 411000; % sum of the weights background
    
    % Split dataset in signal(3 kinds) and backgrounds
    S = D(any(D(:,LABELS)==SIGNAL, 2),:);
    S1 = S(any(S(:,WEIGHTS)<0.0016, 2),:);
    S2 = S(any(and(S(:,WEIGHTS)>0.0025,S(:,WEIGHTS)<0.0027), 2),:);
    S3 = S(any(S(:,WEIGHTS)>0.017, 2),:);
    B = D(any(D(:,LABELS)==BACKGROUND, 2),:);
    
    % Get number of samples
    [Ns, ~] = size(S);
    [Ns1, ~] = size(S1);
    [Ns2, ~] = size(S2);
    [Ns3, ~] = size(S3);
    [Nb, ~] = size(B);
    
    % Calculate number of samples for each stratified subset
    n_train_half_s = floor(n_train*Ns/(Nb+Ns));
    n_train_half_b = floor(n_train-n_train_half_s);
    n_train_half_s1 = floor(n_train_half_s*Ns1/(Ns1+Ns2+Ns3));
    n_train_half_s2 = floor(n_train_half_s*Ns2/(Ns1+Ns2+Ns3));
    n_train_half_s3 = floor(n_train_half_s*Ns3/(Ns1+Ns2+Ns3));
    n_train_sum_s = sum([n_train_half_s1, n_train_half_s2, n_train_half_s3]);
    n_train_sum = sum([n_train_half_b, n_train_sum_s]);
    
    n_val_half_s = floor(n_val*Ns/(Nb+Ns));
    n_val_half_b = floor(n_val-n_val_half_s);        
    n_val_half_s1 = floor(n_val_half_s*Ns1/(Ns1+Ns2+Ns3));
    n_val_half_s2 = floor(n_val_half_s*Ns2/(Ns1+Ns2+Ns3));
    n_val_half_s3 = floor(n_val_half_s*Ns3/(Ns1+Ns2+Ns3));
    n_val_sum_s = sum([n_val_half_s1, n_val_half_s2, n_val_half_s3]);
    n_val_sum = sum([n_val_half_b, n_val_sum_s]);
    
    % Initialization of the 3D output arrays
    train_sets = zeros(n_train_sum, LABELS, k);
    val_sets = zeros(n_val_sum, LABELS, k);
    
    for i=1:k
        % Random permutation
        ps = randperm(Ns);
        pb = randperm(Nb);
        S = S(ps, :);
        S1 = S(any(S(:,WEIGHTS)<0.0016, 2),:);
        S2 = S(any(and(S(:,WEIGHTS)>0.0025,S(:,WEIGHTS)<0.0027), 2),:);
        S3 = S(any(S(:,WEIGHTS)>0.017, 2),:);
        B = B(pb, :);
    
        % get one validation set
        temp_s_val = [  S1(1:n_val_half_s1, :);...
                        S2(1:n_val_half_s2, :);...
                        S3(1:n_val_half_s3, :)];
        temp_b_val = B(1:n_val_half_b, :);
        
        % Sum the weights
        ws = sum(temp_s_val(:, WEIGHTS));
        wb = sum(temp_b_val(:, WEIGHTS));
        
        % Modify the weights
        temp_s_val(:, WEIGHTS) = temp_s_val(:, WEIGHTS)*WS/ws;
        temp_b_val(:, WEIGHTS) = temp_b_val(:, WEIGHTS)*WB/wb;
        
        % Put in the same array
        temp_val = [temp_s_val; 
                    temp_b_val];
              
        % get training set (different from validation samples)
        temp_s_train = [  S1(n_val_half_s1+1:n_val_half_s1+n_train_half_s1, :);...
                          S2(n_val_half_s2+1:n_val_half_s2+n_train_half_s2, :);...
                          S3(n_val_half_s3+1:n_val_half_s3+n_train_half_s3, :)];
        temp_b_train = B(n_val_half_b+1:n_val_half_b+n_train_half_b, :);
        
        % Sum the weights
        ws = sum(temp_s_train(:, WEIGHTS));
        wb = sum(temp_b_train(:, WEIGHTS));
        
        % Modify the weights
        temp_s_train(:, WEIGHTS) = temp_s_train(:, WEIGHTS)*WS/ws;
        temp_b_train(:, WEIGHTS) = temp_b_train(:, WEIGHTS)*WB/wb;

        % Put in the same array
        temp_train = [temp_s_train;
                      temp_b_train];
                  
        % Shuffling
        ii_t = randperm(n_train_sum);
        temp_train = temp_train(ii_t, :);
        
        ii_v = randperm(n_val_sum);
        temp_val = temp_val(ii_v, :);
        
        % Store sets
        train_sets(:, :, i) = temp_train;
        val_sets(:, :, i) = temp_val;                
    end
end
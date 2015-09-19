function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y, train_w, val_w)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
%assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
%opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
AMS_store = [];
n = 1;
for i = 1 : numepochs
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    
    %{
    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        %str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        %str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
    %}
        
    % added by leo
    if opts.validation == 1  
        nn.testing = 1;    
        nn_train = nnff(nn, train_x, zeros(size(train_x,1), nn.size(end)));
        nn_val = nnff(nn, val_x, zeros(size(val_x,1), nn.size(end)));
        nn.testing = 0;

        float_prediction_val = nn_val.a{end};  
        mean_f_p_v = mean(float_prediction_val);
        std_f_p_v = std(float_prediction_val);
        float_prediction_constrained_val = (float_prediction_val-mean_f_p_v)/std_f_p_v;
        %float_prediction_constrained_val = nn_val.a{end};

        float_prediction_train = nn_train.a{end};  
        mean_f_p_t = mean(float_prediction_train);
        std_f_p_t = std(float_prediction_train);
        float_prediction_constrained_train = (float_prediction_train-mean_f_p_t)/std_f_p_t;
        %float_prediction_constrained_train =nn_train.a{end};

        AMS_th_v = [];  
        AMS_th_t = [];

        % Sweep output threshold
        th = linspace(-3, 3, 200);
        for j=th
            % Compute AMS with specific threshold
            prediction_t = float_prediction_constrained_train > j;
            prediction_v = float_prediction_constrained_val > j;

            [AMS, ~, ~, ~] = AMS_metric(prediction_v, [val_w, val_y], 0);
            AMS_th_v = [AMS_th_v, AMS];

            [AMS, ~, ~, ~] = AMS_metric(prediction_t, [train_w, train_y], 0);
            AMS_th_t = [AMS_th_t, AMS];
        end

        [AMS_train_max, idx_train_max] = max(AMS_th_t);
        [AMS_val_max, idx_val_max] = max(AMS_th_v);
        %idx_val_max = idx_train_max;
        %AMS_val_max = AMS_th_v(idx_train_max);

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ...
            ' AMS train: ' num2str(AMS_train_max) ' AMS val: ' num2str(AMS_val_max)...
            ' th train: ' num2str(th(idx_train_max))...
            ' th val: ' num2str(th(idx_val_max))])

    
        if AMS_val_max > nn.AMS_val
            nn.AMS_train = AMS_train_max;
            nn.TH_train = th(idx_train_max);
            nn.AMS_val = AMS_val_max;
            nn.TH_val = th(idx_val_max); 

            nn.nn_best = nn;   
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ' Model saved!'])
        end  
        
        AMS_store = [AMS_store, [AMS_train_max; AMS_val_max]];
        
        if ishandle(fhandle)
            nnupdatefiguresAMS(fhandle, AMS_store, i, opts);
        end        
    else
        loss = nneval(nn, loss, train_x, train_y);
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
    

    

        
    t = toc;   
    %disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
    
end
end


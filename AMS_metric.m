%%=====================================================
%                HIGGS BOSON CHALLENGE 
%======================================================
%   University of Southampton
%   Msc Systems and Signal Processing
%   COMP6208 - Advanced Machine Learning
%   
%   Citraro L., Perodou A., Roullier B., Iyengar A.
%   Start: 12.02.2015 
%   End: 
%======================================================
%%
function [AMS, s, b, NPtot] = AMS_metric(P, S, verbose)
%   AMS calculation:
%   inputs:
%       P: prediction vector [labels(s=1, b=0)]
%       S: solution array Nx2 [weigths, labels(s=1, b=0)]
%       verbose: display information {on=1, off=0}
%   outputs:
%       AMS: approximate median significance
%       s: sum of signal weights
%       b: sum of background weights

    NPtot = length(P);
    NStot = length(S);
    if NPtot ~= NStot
        disp('Error: Inputs have different lengths!');
        return
    end
    
    % this works because s=1 and b=0:
    % signal*signal*weight = weight
    % signal*background*weight = 0
    s = sum(P.*S(:, 2).*S(:, 1));
    
    % signal*NOT(signal)*weight = 0
    % signal*NOT(background)*weight = weight
    b = sum(P.*(-S(:, 2)+ones(NStot, 1)).*S(:, 1));
    
    breg = 10;
    AMS = sqrt(2*((s + b + breg)*log(1 + s/(b + breg))-s));
    
    
    if verbose
        disp('----------------------------------------------');
        disp('AMS_metric:');
        disp([sprintf('\t') 'samples : ', num2str(NPtot)]);
        
        NSsignal = sum(S(:, 2));
        NSbackground = NStot - NSsignal;
        NPsignal = sum(P);
        NPbackground = NPtot - NPsignal;
        disp([sprintf('\t') 'count signals in P: ', num2str(NPsignal)]);       
        disp([sprintf('\t') 'count backgrounds in P: ', num2str(NPbackground)]);
        disp([sprintf('\t') 'ratio backgrounds/signals in P: ', num2str(NPbackground/NPsignal)]);
        disp([sprintf('\t') 'count signals in S: ', num2str(NSsignal)]);
        disp([sprintf('\t') 'count backgrounds in S: ', num2str(NSbackground)]);
        disp([sprintf('\t') 'ratio backgrounds/signals in S: ', num2str(NSbackground/NSsignal)]);      
        
        Ns = sum(S(:, 1).*S(:, 2));
        Nb = sum(S(:, 1).*(-S(:, 2)+ones(NStot, 1)));
        disp([sprintf('\t') 'Ns: ', num2str(Ns)]);
        disp([sprintf('\t') 'Nb: ', num2str(Nb)]);
        disp([sprintf('\t') 'Nb/Ns: ', num2str(Nb/Ns)]);
        
        
        disp([sprintf('\t') 's: ', num2str(s)]);
        disp([sprintf('\t') 'b: ', num2str(b)]);
        disp([sprintf('\t') 'AMS: ', num2str(AMS)]);
        disp('----------------------------------------------');
    end
end

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
%% Testbench AMS_metric function
clc
clear all
% P=[prediction s=1 b=0]
% Number of signals=8
% Number of background = 3
% ratio 3/8 = 0.375
% Tot = 11
P = [1 1 0 0 1 1 1 1 1 1 0]';

% S=[weights; solution s=1 b=0]
% Number of signals=6
% Number of background = 5
% ratio 5/6 = 0.833
% Tot = 11
% Ns = 2+4+5+6+7+11= 35
% Nb = 1+3+8+9+10 = 31
S = [1 2 3 4 5 6 7 8 9 10 11;
     0 1 0 1 1 1 1 0 0 0 1]';
 
% s = 2+5+6+7 = 20;

% b = 1+8+9+10 = 28

% AMS = 3.008553588541159
AMS_testbench = sqrt(2*((48+10)*log(1+20/(28+10))-20))

[AMS, s, b, N] = AMS_metric(P, S, 1);

%%  ECE414 Makeene Learning - Project 2
%   By Jeffrey Shih

%% Linear Regression, Parameter Distribution - Data Generation

clc; clear all; close all;

% Model weights (we want to estimate these)
w0 = -0.3;
w1 = 0.5;
w = [w0 w1];

% Generate random data from uniform distribution U(x|-1,1) and add noise
% We want to see the effect on increasing # of observations N on
% the estimation for each of nTrials trials (so the data is N-by-nTrials)
nTrials = 6;
N = 100;
x = rand(N,nTrials)*2-1; % rand generates number taken uniformly from (0,1)
                        % so scale to (0,2), then subtract 1 for (-1,1)

% Find targets associated with data
t = linear_model_function(x,w0,w1);

% Generate Gaussian noise of std. dev 0.2 and add to targets            
sigma = 0.2;
noise = randn(N,nTrials)*sigma; % scale standard normal so that std. dev becomes 0.2
t = t + noise;


%% Linear Regression, Parameter Distribution - Estimating Model Parameters

% Noise parameter
beta = 1/sigma^2;

% Hyperparameter for prior
alpha = 2;

% Lambda is the regularization parameter
% See P153, bottom, for how it is defined in this example
lambda = alpha/beta;

% Generate initial estimates based on prior only (zero-mean, std.dev alpha)
w_est_prior = randn(2,nTrials)*alpha

% Estimate weights for model using REGULARIZED LEAST SQUARES
% We look at individual trials for each # of observations
w_est_matrix = []; % matrix to store all estimated weights, N*2-by-nTrials
for n = 1:N
    % Temporary array to store estimated weights for all trials of the same n
    w_est_trials = [];
    for nT = 1:nTrials
        % Basis functions for the design matrix, in this case 1 and x
        phi0 = ones(n,1);
        phi1 = x(1:n,nT);
        % Design matrix, see P142 (3.16)
        PHI = [phi0 phi1];
        % Expression for weights, see P145 (3.28)
        % Taking the Moore-Penrose pseudoinverse since PHI is not square
        % Slice targets such that we're only looking at THIS particular trial
        w_est = pinv(lambda*eye(length(w))+PHI'*PHI)*PHI'*t(1:n,nT);
        w_est_trials = [w_est_trials w_est];
    end
    % Examine this matrix at the end to see that as N goes up, estimates
    % of a0 and a1 become more acurate
    w_est_matrix = [w_est_matrix; w_est_trials];
end

w_est_matrix;

%% Linear Regression, Parameter Distribution - Plotting Figure 3.7

% Initialize plot
% We're plotting both n=1 and n=N so that publish displays 2 graphs
figure;
update_plot_3_7(N,N,nTrials,w,x,t,alpha,beta,w_est_prior,w_est_matrix);
f = figure;
update_plot_3_7(0,N,nTrials,w,x,t,alpha,beta,w_est_prior,w_est_matrix);

% Set up slider used to change n
slider = uicontrol('Parent',f,'Style','slider','Position',[10 50 20 340],...
              'value',0,'min',0,'max',N,'SliderStep',[1/(N-0) 1]);
bgcolor = f.Color;
slider_label1 = uicontrol('Parent',f,'Style','text','Position',[10,24,23,23],...
                'String','0','BackgroundColor',bgcolor);
slider_label2 = uicontrol('Parent',f,'Style','text','Position',[10,390,30,23],...
                'String',num2str(N),'BackgroundColor',bgcolor);

% Set slider callback to update plot, need to round the slider value
% because it might not be an integer
slider.Callback = @(es,ed) update_plot_3_7(round(es.Value),N,nTrials,w,x,t,alpha,beta,w_est_prior,w_est_matrix);


%% Testing

% Plot data space for all trials for n = 1
%{
for currTrial = 1:nTrials
    w0_est = w_est_matrix(1,currTrial)
    w1_est = w_est_matrix(1*2,currTrial)
    y_vec = w0_est + w1_est*x_vec;
    plot(x_vec,y_vec)
    title(['data space (N = ',num2str(1),')'])
    xlabel('\it x')
    ylabel('\it y')
    hold on
end
%}

figure
% Plot multivariate normal distribution for prior
m0 = zeros(length(w),1); % zero mean
S0 = alpha^(-1)*eye(length(w)); % variance for prior is governed by hyperparam 
w1 = -1:0.01:1;
w2 = -1:0.01:1;
[W1,W2] = meshgrid(w1,w2);
F = reshape(mvnpdf([W1(:) W2(:)],m0',S0),length(w2),length(w1));
imagesc(w1,w2,F)
%contour(w1,w2,F);
title('prior/posterior')
xlabel('\it w_{0}')
ylabel('\it w_{1}')

%% 
figure
% Plot multivariate normal distribution for posterior
n = 1;
nT = 1;
phi0 = ones(n,1);
phi1 = x(1:n,nT);
PHI = [phi0 phi1]
SN = pinv(pinv(S0) + beta*PHI'*PHI);
mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n,nT));
w1 = -1:0.1:1;
w2 = w1;
size(mN)
size(SN)
[W1,W2] = meshgrid(w1,w2);
F = reshape(mvnpdf([W1(:) W2(:)],mN',SN),length(w2),length(w1));
imagesc(F)
%contour(w1,w2,F);
title('prior/posterior')
xlabel('\it w_{0}')
ylabel('\it w_{1}')

%% 
    % Likelihood plot
    subplot(2,2,1)
    % Calculate design matrix
    currTrial = 1;
    phi0 = ones(n,1); % 0th basis function, 1
    phi1 = x(1:n,currTrial); % 1st basis function, x
    PHI = [phi0 phi1];
    % Match the equations on P93 (2.114) and P141 (3.10) for the parameters to use
    % This is also what we used for the likelihood to prove the posterior mean
    % and variance in class
    A = PHI;
    b = zeros(n,1);
    L_inv = beta^(-1)*eye(n);
    mN = A*w_est_matrix(n*2-1:n*2,currTrial) + b;
    SN = L_inv;
    w1 = -1:0.1:1;
    w2 = 0:0.1:1;
    [W1,W2] = meshgrid(w1,w2);
    FL = reshape(mvnpdf([W1(:) W2(:)],mL',SL),length(w2),length(w1));
    contour(w1,w2,FL);
    hold on;
    % Plot the truth
    scatter(w(1),w(2),'k+')
    % Plot labels
    title(['likelihood (n = ',num2str(n),')'])
    xlabel('\it w_{0}')
    ylabel('\it w_{1}')
    axis([-1 1 -1 1])
    hold off;
%%  ECE414 Makeene Learning - Project 2, Part II
%   By Jeffrey Shih

%% Linear Regression, Predictive Distribution - Data Generation

clc; clear all; close all;

% Generate random observations from uniform distribution U(x|0,1)
% We want to see the effect on increasing # of observations N on
% the predictive distribution
N = 100;
x = rand(N,1);

% Find targets associated with data and add Gaussian noise
sigma = 0.2; % standard deviation of Gaussian noise
beta = 1/sigma^2; % noise precision parameter, i.e. precision of observations
noise = randn(N,1)*sigma; % scale standard normal using desired std. dev
t = sin(2*pi*x) + noise;

% Prior parameters (see P153, zero-mean isotropic Gaussian with precision alpha)
% Since we're using 9 basis functions (i.e. 9 weights), the mean vector has
% 9 elements and the variance matrix is 9x9 (see Figure 3.8)
nWeights = 9;
alpha = 2; % hyperparameter for precision (see P153, 3.52)
S0 = alpha^(-1)*eye(nWeights);
m0 = zeros(nWeights,1);

% Parameters for Gaussian basis functions (see P139, 3.4)
u = linspace(-1,1,9); % locations of basis in input space
s = 0.2; % spatial scale of basis

%% Linear Regression, Predictive Distribution - Plotting Figure 3.8

close all;

% Initialize plot
% We're plotting both n=1 and n=N so that publish displays 2 graphs
figure;
update_plot_3_8(N,N,nWeights,x,t,beta,m0,S0,u,s)
figure;
update_plot_3_8(5,N,nWeights,x,t,beta,m0,S0,u,s)
figure;
update_plot_3_8(1,N,nWeights,x,t,beta,m0,S0,u,s)
f = figure;
update_plot_3_8(0,N,nWeights,x,t,beta,m0,S0,u,s)

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
slider.Callback = @(es,ed) update_plot_3_8(round(es.Value),N,nWeights,x,t,beta,m0,S0,u,s);

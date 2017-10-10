% Function used in conjunction with slider to update Figure 3.7 plots
% Estimate weights and plot likelihood/prior/posterior for increasing # of observations
%   prior - distribution on w conditioned on prior params
%   likelihood - distribution on (new) t conditioned on (previous) w
%   posterior - distribution on (new) w conditioned on (previous) t
function update_plot_3_7(n,N,nSamples,w,x,t,alpha,beta,w_est_prior,w_est_matrix,w_est_LS)
    
    % Change colormap
    colormap('jet')

    % Handle prior case (no observations)
    if n == 0
        
        % Empty likelihood subplot
        subplot(1,3,1,'replace')
        title(['likelihood (n = ',num2str(n),')'])
        xlabel('\it w_{0}')
        ylabel('\it w_{1}')
        axis([-1 1 -1 1])
        
        % Plot multivariate normal distribution for prior 
        subplot(1,3,2)
        m0 = zeros(length(w),1); % zero mean column vector, need to transpose when using mvnpdf
        S0 = alpha^(-1)*eye(length(w)); % variance for prior is governed by hyperparam alpha
        % Weight space
        w0 = -1:0.01:1;
        w1 = -1:0.01:1;
        [W0,W1] = meshgrid(w0,w1);
        % mvnpdf returns the distribution of w by accepting all possible
        % combinations of w0 and w1 as a 2-column matrix, then spits out a
        % column vector for the probabilities, so you need to reshape
        F0 = reshape(mvnpdf([W0(:) W1(:)],m0',S0),length(w1),length(w0));
        % Plot distribution image
        %contour(w0,w1,F0)
        imagesc(w0,w1,F0)
        set(gca,'YDir','normal') % unflip y-axis for imagesc
        title(['prior/posterior (n = ',num2str(n),')'])
        xlabel('\it w_{0}')
        ylabel('\it w_{1}')
        axis([-1 1 -1 1])
        
        % Plot weight estimates from prior (random with zero-mean)
        subplot(1,3,3)
        x_vec = -1:0.01:1;
        for currSample = 1:nSamples
            % Plot prior estimates
            w0_est = w_est_prior(1,currSample);
            w1_est = w_est_prior(2,currSample);
            y_vec = w0_est + w1_est*x_vec;
            plot(x_vec,y_vec,'r')
            hold on
        end
        title(['data space (n = ',num2str(n),')'])
        xlabel('\it x')
        ylabel('\it y')
        axis([-1 1 -1 1])
        
        hold off;
        return
    end
    
    % Now, we plot what happens with each additional observation
    
    % Likelihood Plot
    
    subplot(1,3,1)
    % Calculate probability of target for each w0, w1 given new observation
    w0 = -1:0.2:1;
    w1 = -1:0.2:1;
    % Meshgrid to avoid nested for loop
    % Since w0 is copied over row-wise such that there are length(w1) rows,
    % w1 represents the y-axis and we don't have to worry about transposing
    % the data
    [W0,W1] = meshgrid(w0,w1);
    phi0_n = 1; % 0th basis function, 1
    phi1_n = x(n); % 1st basis function, x, we only care about new obs
    % See P141, 3.10 for following expressions
    % The mean for each new observation is the linear combination of basis
    % functions evaluated for the observation and weights
    mL = phi0_n*W0 + phi1_n*W1;
    SL = beta^(-1);
    % Gaussian distribution for new target (P78, 2.42)
    % We evaluate the probability for all w0, w1 pairs
    FL = 1/(2*pi*SL)^(1/2)*exp(-1/(2*SL)*(t(n)-mL).^2)
    %FL = normpdf(t(n),mL,sqrt(SL)) # same result, normpdf takes std. dev
    % Plot distribution image
    imagesc(w0,w1,FL);
    set(gca,'YDir','normal') % unflip y-axis for imagesc
    hold on;
    % Plot the truth
    scatter(w(1),w(2),'w+','lineWidth',2)
    % Plot labels
    title(['likelihood (n = ',num2str(n),')'])
    xlabel('\it w_{0}')
    ylabel('\it w_{1}')
    axis([-1 1 -1 1])
    hold off;
    
    % Posterior plot
    
    subplot(1,3,2)
    % Prior parameters
    m0 = zeros(length(w),1); % zero mean
    S0 = alpha^(-1)*eye(length(w)); % variance for prior is governed by hyperparam
    % Design matrix for this n
    phi0 = ones(n,1); % 0th basis function, 1
    phi1 = x(1:n); % 1st basis function, x
    PHI = [phi0 phi1];
    % Posterior parameters
    SN = pinv(pinv(S0) + beta*PHI'*PHI);
    mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n)); % column vector, take tranpose when using mvnpdf
    % Weight space
    w0 = -1:0.01:1;
    w1 = -1:0.01:1;
    [W0,W1] = meshgrid(w0,w1);
    FN = reshape(mvnpdf([W0(:) W1(:)],mN',SN),length(w1),length(w0));
    % Plot distribution image
    imagesc(w0,w1,FN)
    set(gca,'YDir','normal') % unflip y-axis for imagesc
    hold on;
    % Plot the truth
    scatter(w(1),w(2),'w+','lineWidth',2)
    % Plot labels
    title(['prior/posterior (n = ',num2str(n),')'])
    xlabel('\it w_{0}')
    ylabel('\it w_{1}')
    axis([-1 1 -1 1])
    hold off;
    
    % Data space plots (posterior samples and RLS)
    
    subplot(1,3,3)
    % Plot actual observations
    x_obs = x(1:n);
    t_obs = t(1:n);
    scatter(x_obs,t_obs,'b')
    hold on
    % Plot all samples and RLS for the same n on the same figure
    x_vec = -1:0.01:1;
    % Posterior sample estimate
    for currSample = 1:nSamples
        w0_est = w_est_matrix(n*2-1,currSample);
        w1_est = w_est_matrix(n*2,currSample);
        y_vec = w0_est + w1_est*x_vec;
        plot(x_vec,y_vec,'r')
        hold on
    end
    % Regularized least squares weight estimate
    w0_est = w_est_LS(n*2-1);
    w1_est = w_est_LS(n*2);
    y_vec = w0_est + w1_est*x_vec;
    plot(x_vec,y_vec,'k')
    % Plot labels
    title(['data space (n = ',num2str(n),')'])
    xlabel('\it x')
    ylabel('\it y')
    axis([-1 1 -1 1.5])
    legend('Observations','Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','RLS')
    hold off
end
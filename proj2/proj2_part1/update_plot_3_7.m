% Function used in conjunction with slider to update Figure 3.7 plots
function update_plot_3_7(n,N,nTrials,w,x,t,alpha,beta,w_est_prior,w_est_matrix,w_est_matrix_LS)
    
    % Change colormap
    colormap('jet')

    % Handle prior case (no observations)
    if n == 0
        
        % Empty likelihood subplot
        subplot(2,2,1,'replace')
        title(['likelihood (n = ',num2str(n),')'])
        xlabel('\it w_{0}')
        ylabel('\it w_{1}')
        axis([-1 1 -1 1])
        
        % Plot multivariate normal distribution for prior
        subplot(2,2,2)
        m0 = zeros(length(w),1); % zero mean column vector, need to transpose when using mvnpdf
        S0 = alpha^(-1)*eye(length(w)); % variance for prior is governed by hyperparam alpha
        w0 = -1:0.01:1;
        w1 = -1:0.01:1;
        [W0,W1] = meshgrid(w0,w1);
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
        for i = 3:4
            subplot(2,2,i)
            x_vec = -1:0.01:1;
            for currTrial = 1:nTrials
                % Plot prior estimates
                w0_est = w_est_prior(1,currTrial);
                w1_est = w_est_prior(2,currTrial);
                y_vec = w0_est + w1_est*x_vec;
                plot(x_vec,y_vec,'r')
                hold on
            end
            if i == 3
                title(['data space (n = ',num2str(n),')'])
            else
                title(['data space RLS (n = ',num2str(n),')'])
            end
            xlabel('\it x')
            ylabel('\it y')
            axis([-1 1 -1 1])
            hold off;
        end
        
        hold off;
        return
    end
    
    % Now, we plot what happens with each additional observation
    
    % Likelihood Plot
    
    subplot(2,2,1)
    % Calculate probability of target for each w1, w2 given new observation
    currTrial = 1;
    w0 = -1:0.01:1;
    w1 = -1:0.01:1;
    %{
    % Stores the Gaussian probability values for all w1, w2 in FL
    % rows = length(w2) since w2 represents y-axis
    FL = ones(length(w1),length(w0));
    for i = 1:length(w1)
        for j = 1:length(w0)
            w_vec = [w0(j) w1(i)];
            phi0_n = 1; % 0th basis function, 1
            phi1_n = x(n,currTrial); % 1st basis function, x
            PHI_n = [phi0_n phi1_n];
            % See P141, 3.10 for following expressions
            mL = PHI_n*w_vec'; % mean for one observation
            SL = beta^(-1); % variance for one observation
            % Gaussian distribution for single target (P78, 2.42)
            FL(i,j) = 1/(2*pi*SL)^(1/2)*exp(-1/(2*SL)*(t(n,currTrial)-mL)^2);
        end
    end
    %}
    % Meshgrid to avoid nested for loop
    % Since w0 is copied over row-wise such that there are length(w1) rows,
    % w1 represents the y-axis and we don't have to worry about transposing
    % the data
    [W0,W1] = meshgrid(w0,w1);
    phi0_n = 1; % 0th basis function, 1
    phi1_n = x(n,currTrial); % 1st basis function, x
    % See P141, 3.10 for following expressions
    mL = phi0_n*W0 + phi1_n*W1; % mean for one observation
    SL = beta^(-1); % variance for one observation
    % Gaussian distribution for single target (P78, 2.42)
    % We evaluate the probability for all w0, w1 pairs
    FL = 1/(2*pi*SL)^(1/2)*exp(-1/(2*SL)*(t(n,currTrial)-mL).^2);
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
    
    subplot(2,2,2)
    % Prior parameters
    m0 = zeros(length(w),1); % zero mean
    S0 = alpha^(-1)*eye(length(w)); % variance for prior is governed by hyperparam
    % Design matrix for one particular trial of this n
    currTrial = 1;
    phi0 = ones(n,1); % 0th basis function, 1
    phi1 = x(1:n,currTrial); % 1st basis function, x
    PHI = [phi0 phi1];
    % Posterior parameters
    SN = pinv(pinv(S0) + beta*PHI'*PHI);
    mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n,currTrial)); % column vector, take tranpose when using mvnpdf
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
    
    % Data space plots (one for posterior mean estimate, one for RLS estimate)
    
    for i = 3:4
        subplot(2,2,i)
        % Plot actual observations for one trial
        currTrial = 1;
        x_obs = x(1:n,currTrial);
        t_obs = t(1:n,currTrial);
        scatter(x_obs,t_obs,'b')
        hold on
        % Plot all trials for the same n on the same figure
        x_vec = -1:0.01:1;
        for currTrial = 1:nTrials
            % Plot model estimates
            if i == 3 % Posterior mean weight estimates
                w0_est = w_est_matrix(n*2-1,currTrial);
                w1_est = w_est_matrix(n*2,currTrial);
            else % Regularized least squares weight estimates
                w0_est = w_est_matrix_LS(n*2-1,currTrial);
                w1_est = w_est_matrix_LS(n*2,currTrial);
            end
            y_vec = w0_est + w1_est*x_vec;
            plot(x_vec,y_vec,'r')
            hold on
        end
        % Plot labels
        if i == 3
            title(['data space (n = ',num2str(n),')'])
        else
            title(['data space RLS (n = ',num2str(n),')'])
        end
        xlabel('\it x')
        ylabel('\it y')
        axis([-1 1 -1 1])
        hold off
    end
end
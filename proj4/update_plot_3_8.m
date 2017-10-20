% Function used in conjunction with slider to update Figure 3.8 plots
% Calculate predictive distributions for increasing # of observations
% Includes both new and old method use Gaussian processes for regression
%   - Old method uses design matrix; new method uses kernel function
% Inputs:
%   - n - number of data points
%   - nWeights - number of weights/basis functions
%   - x - input data, has n points
%   - t - targets, n points
%   - beta - input noise precision (Gaussian)
%   - m0 - prior mean
%   - S0 - prior variance
%   - u - Gaussian basis function parameter, input space location of basis
%   - s - Gaussian basis function parameter, spatial scale for basis
%   - theta - the 4 Kernel function parameters
function update_plot_3_8(n,nWeights,x,t,beta,m0,S0,u,s,theta)

    % This is the "continuous" range of values that we're plotting the
    % our distribution for; different from x, which represents our observations
    x_truth = (0:0.01:1)'; % column vector
    
    % Only plot truth if no observations
    if n == 0
        for i = [1,2]
            subplot(1,2,i)
            plot(x_truth,sin(2*pi*x_truth),'g')
            if i == 1
                title(sprintf('Predictive Distribution, Design Matrix (n = %d)',n))
            elseif i == 2
                title(sprintf('Predictive Distribution, Kernel Function (n = %d)',n))
            end
            xlabel('\it x')
            ylabel('\it t')
            axis([0 1 -1.5 1.5])
            legend('truth')
        end
        return
    end

    % METHOD 1: GAUSSIAN PROCESS REGRESSION USING DESIGN MATRIX

    % Populate design matrix by evaluating all basis functions for each xn
    % that we have already seen (see P142, 3.16)
    % We use this to find the posterior parameters below
    for m = 1:nWeights
        PHI(:,m) = gaussian_basis_function(x,u(m),s); % PHI(:,m) ensures that columns are stacked next to each other
    end
    
    % Posterior parameters (see P153, 3.50, 3.51)
    SN = pinv(pinv(S0) + beta*PHI'*PHI);
    mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n));
    
    % New design matrix for each possible value of x_truth, used to form
    % our distribution curve
    for m = 1:nWeights
        PHI_truth(:,m) = gaussian_basis_function(x_truth,u(m),s);
    end
    
    % Predictive distribution parameters (see P156, 3.58, 3.59)
    % Calculate mean and variance for each point in input x_truth for
    % continuous looking graph
    for i = 1:length(x_truth)
        phi_x = PHI_truth(i,:)'; % P146, phi(x) is an M-dim COLUMN vector with elements
                                % phi_j(x), so we're just looking at the basis
                                % function evaluated for one observation
        %mP_i(i,:) = mN'*phi_x;
        SP(i,:) = 1/beta + phi_x'*SN*phi_x; % SP(i,:) ensures that rows are stacked below each other to column vec
    end
    % Can also do mP as follows:
    mP = (mN'*PHI_truth')'; % transpose again to make column vector

    % METHOD 2: GAUSSIAN PROCESS REGRESSION USING KERNEL FUNCTION

    % Covariance matrix (6.62)
    for i = 1:length(x)
        for j = 1:length(x)
            % For kroneckerDelta with numeric inputs, use the eq function instead.
            C_N(i,j) = kernel_function(x(i),x(j),theta) + beta^(-1)*eq(i,j);
        end
    end

    % Calculate predictive distribution over x_truth
    % i.e. what is the predicted target value for each point in x_truth?
    % Need new C_N+1, k, c => m_N+1, S_N+1 for each point in x_truth
    % See P307-308, 6.65, 6.66, 6.67
    for i = 1:length(x_truth)
        for j = 1:length(x)
            k(j,:) = kernel_function(x(j),x_truth(i),theta); % ensures that k is column vector
        end
        c = kernel_function(x_truth(i),x_truth(i),theta) + beta^(-1);
        C_Nplus1 = [C_N, k; k', c];
        m_Nplus1(i,:) = k'*(C_N\t); % use C_N\t instead of inv(C_N)*t, faster
        S_Nplus1(i,:) = c-k'*(C_N\k); % matrix multiplication is associative, so do division first
    end
    
    % Plot figures

    for i = [1,2]

        subplot(1,2,i)

        % Truth (green line)
        plot(x_truth,sin(2*pi*x_truth),'g')
        hold on

        % Observations (blue circles)
        scatter(x(1:n),t(1:n),'b')
        hold on

        % Mean of Predictive Distribution (red line)
        % This curve represents our model estimate
        if i == 1
            plot(x_truth,mP,'r')
        elseif i == 2
            plot(x_truth,m_Nplus1,'r')
        end
        hold on
        
        % Std. Dev of Predictive Distribution (red shaded region)
        % Don't forget to take the root of SP
        if i == 1
            top_y = mP+sqrt(SP); % upper bound for std.dev
            bot_y = mP-sqrt(SP); % lower bound
        elseif i == 2
            top_y = m_Nplus1+sqrt(S_Nplus1); % upper bound for std.dev
            bot_y = m_Nplus1-sqrt(S_Nplus1); % lower bound
        end
        % Draw polygon, start at bot and go from left to right,
        % then move to top and go from right to left
        % Transpose so that they're all row vectors (easier to visualize)
        X = [x_truth' fliplr(x_truth')];
        Y = [bot_y' fliplr(top_y')];
        color = [1 0.4 0.6];
        h = fill(X,Y,color);
        set(h,'EdgeColor','none') % remove edges/borders from fill
        alpha(0.3) % make filled region slightly transparent
        %uistack(h,'bottom') % moves this shading behind all other plots
        hold on
        
        % Plot labels
        if i == 1
            title(sprintf('Predictive Distribution, Design Matrix, n = %d',n))
        elseif i == 2
            title(sprintf('Predictive Distribution, Kernel Function, n = %d',n))
        end
        xlabel('\it x')
        ylabel('\it t')
        axis([0 1 -1.5 1.5])
        legend('truth','observations','predictive mean','predictive std. dev')
        hold off
    end
end
% Function used in conjunction with slider to update Figure 3.8 plots
% Calculate predictive distributions for increasing # of observations
function update_plot_3_8(n,N,nWeights,x,t,beta,m0,S0,u,s)

    % This is the "continuous" range of values that we're plotting the
    % our distribution for; different from x, which represents our observations
    x_truth = (0:0.01:1)'; % column vector
    
    % Only plot truth if no observations
    if n == 0
        plot(x_truth,sin(2*pi*x_truth),'g')
        title(sprintf('n = %d',n))
        xlabel('\it x')
        ylabel('\it t')
        axis([0 1 -1.5 1.5])
        legend('truth')
        return
    end

    % Populate design matrix by evaluating all basis functions for each xn
    % that we have already seen (see P142, 3.16)
    % We use this to find the posterior parameters below
    PHI = [];
    for m = 1:nWeights
        PHI = [PHI gaussian_basis_function(x(1:n),u(m),s)];
    end
    
    % Posterior parameters (see P153, 3.50, 3.51)
    SN = pinv(pinv(S0) + beta*PHI'*PHI);
    mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n));
    
    % New design matrix for each possible value of x_truth, used to form
    % our distribution curve
    PHI_truth = [];
    for m = 1:nWeights
        PHI_truth = [PHI_truth gaussian_basis_function(x_truth,u(m),s)];
    end
    
    % Predictive distribution parameters (see P156, 3.58, 3.59)
    % Calculate mean and variance for each point in input x_truth for
    % continuous looking graph
    %mP = [];
    SP = [];
    for i = 1:length(x_truth)
        phi_x = PHI_truth(i,:)'; % P146, phi(x) is an M-dim COLUMN vector with elements
                                % phi_j(x), so we're just looking at the basis
                                % function evaluated for one observation
        %mP_i = mN'*phi_x;
        SP_i = 1/beta + phi_x'*SN*phi_x;
        %mP = [mP; mP_i];
        SP = [SP; SP_i];
    end
    % Can also do mP as follows:
    y_pred = (mN'*PHI_truth')'; % transpose again to make column vector

    % Plot figures

    % Truth (green line)
    plot(x_truth,sin(2*pi*x_truth),'g')
    hold on

    % Observations (blue circles)
    scatter(x(1:n),t(1:n),'b')
    hold on

    % Mean of Predictive Distribution (red line)
    % This curve represents our model estimate
    plot(x_truth,y_pred,'r')
    hold on
    
    % Std. Dev of Predictive Distribution (red shaded region)
    % Don't forget to take the root of SP
    top_y = y_pred+sqrt(SP); % upper bound for std.dev
    bot_y = y_pred-sqrt(SP); % lower bound
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
    title(sprintf('n = %d',n))
    xlabel('\it x')
    ylabel('\it t')
    axis([0 1 -1.5 1.5])
    legend('truth','observations','predictive mean','predictive std. dev')
    hold off
end
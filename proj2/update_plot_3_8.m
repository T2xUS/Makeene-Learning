% Function used in conjunction with slider to update Figure 3.8 plots
% Calculate predictive distributions for increasing # of observations
function update_plot_3_8(n,N,nWeights,x,t,beta,m0,S0,u,s)

    % Populate design matrix by evaluate all basis functions for each xn
    % (see P142, 3.16)
    PHI = [];
    for m = 1:nWeights
        PHI = [PHI gaussian_basis_function(x(1:n),u(m),s)];
    end
    
    % Posterior parameters (see P153, 3.50, 3.51)
    SN = pinv(pinv(S0) + beta*PHI'*PHI);
    mN = SN*(pinv(S0)*m0 + beta*PHI'*t(1:n));

    % Predictive distribution parameters (see P156, 3.58, 3.59)
    % Go through each target and find the mean and variance associated with it
    mP = [];
    SP = [];
    for i = 1:n
        phi_x = PHI(i,:)'; % P146, phi(x) is an M-dim COLUMN vector with elements
                           % phi_j(x), so we're just looking at the basis
                           % function evaluated for one observation
        mP_i = mN'*phi_x;
        SP_i = 1/beta + phi_x'*SN*phi_x;
        mP = [mP; mP_i];
        SP = [SP; SP_i];
    end

    % Plot figures

    % Truth (green line)
    x_truth = 0:0.01:1;
    plot(x_truth,sin(2*pi*x_truth),'g')
    hold on

    % Observations (blue circles)
    scatter(x(1:n),t(1:n),'b')
    hold on

    % Mean of Predictive Distribution (red line)
    scatter(x(1:n),mP,'r+')
    hold on
    % Sort the numbers before you plot line so it doesn't look like a mess
    [sorted_x, sorted_i] = sort(x(1:n));
    plot(sorted_x,mP(sorted_i),'r')
    hold on

    % Std. Dev of Predictive Distribution (red shaded region)
    % Don't forget to take the root of SP
    %plot(sorted_x,mP(sorted_i)+sqrt(SP(sorted_i)),'k')
    %plot(sorted_x,mP(sorted_i)-sqrt(SP(sorted_i)),'k')
    top_y = mP(sorted_i)+sqrt(SP(sorted_i)); % upper bound for std.dev
    bot_y = mP(sorted_i)-sqrt(SP(sorted_i)); % lower bound
    % Draw polygon, start at bot and go from left to right,
    % then move to top and go from right to left
    % Transpose so that they're all row vectors (easier to visualize)
    X = [sorted_x' fliplr(sorted_x')];
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
    % Don't include extra legend entries if unnecessary (will cause issues)
    if n == 0
        legend('truth')
    else
        legend('truth','observations','predictive mean','predictive model','predictive std. dev')
    end
    hold off
end
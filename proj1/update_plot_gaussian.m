% Function used in conjunction with slider to update Gaussian posterior plot
function update_plot_gaussian(i,N,x_vec,s,mu_ML_norm,sigma,mu_o,sigma_o)
    % Update parameters
    % Since the update parameter contains the ML estimate, I decided to
    % average out the ML estimates for this particular N across all trials,
    % not sure if this is the proper way to do it
    mu_ML_avg = mean(mu_ML_norm(:,i),1);
    mu_N = sigma^2/(N(i)*sigma_o(s)^2+sigma^2)*mu_o(s) + ...
            N(i)*sigma_o(s)^2/(N(i)*sigma_o(s)^2+sigma^2)*mu_ML_avg;
    sigma_N = (1/sigma_o(s)^2+N(i)/sigma^2)^(-1/2); % -1/2 power b/c sqrt
    % Update distribution
    p_post = pdf('normal',x_vec,mu_N,sigma_N);
    % "Plot" graph
    plot(x_vec,p_post)
    title(sprintf('Gaussian Posterior Density (\\mu_{0}=%.3f, \\sigma_{0}=%.3f)',mu_o(s),sigma_o(s)))
    xlabel('X')
    axis([-1 1 0 3])
    legend(['N = ' num2str(N(i))])
    %drawnow
    % Capture plot frame
    M(i) = getframe(gcf);
end
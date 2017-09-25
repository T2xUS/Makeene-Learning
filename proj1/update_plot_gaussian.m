% Function used in conjunction with slider to update Gaussian posterior plot
function update_plot_gaussian(i,N,mean_vec,s,mu_ML_norm,sigma,mu_o,sigma_o)
    % Update parameters
    mu_N = sigma^2/(N(i)*sigma_o(s)^2+sigma^2)*mu_o(s) + ...
            N(i)*sigma_o(s)^2/(N(i)*sigma_o(s)^2+sigma^2)*mu_ML_norm(i);
    sigma_N = (1/sigma_o(s)^2+N(i)/sigma^2)^(-1/2); % -1/2 power b/c sqrt
    % Update distribution
    p_post = pdf('normal',mean_vec,mu_N,sigma_N);
    % "Plot" graph
    plot(mean_vec,p_post)
    title(sprintf('Gaussian Posterior Density (\\mu_{0}=%.3f, \\sigma_{0}=%.3f)',mu_o(s),sigma_o(s)))
    xlabel('Mean')
    ylabel('Density')
    axis([-1 1 0 15])
    legend(['N = ' num2str(N(i))])
end